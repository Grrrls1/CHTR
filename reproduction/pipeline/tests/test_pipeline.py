import json
from pathlib import Path
import subprocess
import sys
import tempfile
import struct
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import states, validate_points, render, iou, png, write_records, digest, arrays
from decode_vs_npz import decode
from render_and_filter import evaluate
from validate_dataset import validate

PIPE=Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.d=Path(self.tmp.name)

    def tearDown(self): self.tmp.cleanup()

    def cli(self,name,*args,ok=True):
        proc=subprocess.run([sys.executable,str(PIPE/(name+'.py'))]+list(map(str,args)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if ok: self.assertEqual(proc.returncode,0,proc.stdout.decode())
        else: self.assertNotEqual(proc.returncode,0)
        return proc

    def native(self,data,lengths,cursors):
        p=self.d/'native.npz'
        np.savez(str(p),strokes_data=np.asarray(data,float),round_length=np.asarray(lengths,dtype=np.int64),init_cursors=np.asarray(cursors),image_size=256,init_width=.01)
        return p

    def test_axis_width_round_reset_and_move(self):
        source=self.native([[0,.5,.5,0,.5,.1,2],[1,.5,.5,0,.25,.2,.5],[0,.5,.5,.5,0,.1,1],[0,.5,.5,0,.5,.1,1]],[3,1],[[.25,.25],[.5,.5]])
        z=decode(source,dict(raster_size=128,min_window_size=32,cursor_type='next'))
        self.assertEqual(len(z['subpath_offsets']),4)
        first=z['points'][:z['subpath_offsets'][1]]
        np.testing.assert_allclose(first[0,:2],[64,64]); np.testing.assert_allclose(first[-1,:2],[96,64])
        self.assertEqual(first[0,2],1); self.assertEqual(first[-1,2],7)
        third=z['points'][z['subpath_offsets'][2]:]
        np.testing.assert_allclose(third[0,:3],[128,128,1])
        validate_points(z['points'],z['subpath_offsets'])

    def test_malformed_and_empty_native(self):
        p=self.native([[1,.5,.5,0,.5,.1,1]],[1],[[.5,.5]])
        with self.assertRaises(ValueError): decode(p,dict(raster_size=128,min_window_size=32,cursor_type='next'))
        p=self.native([[0,.5,.5,0,.5,.1,1]],[2],[[.5,.5]])
        with self.assertRaises(ValueError): decode(p,dict(raster_size=128,min_window_size=32,cursor_type='next'))

    def test_no_cross_subpath_lines_and_end_states(self):
        p=states(np.array([[20,20,4,0,0,0],[40,20,4,0,0,0],[180,180,4,0,0,0],[200,180,4,0,0,0]]),np.array([0,2,4]))
        image=render(p,np.array([0,2,4]))
        self.assertEqual(image[100,100],0)
        np.testing.assert_array_equal(p[1,3:],[0,1,0]); np.testing.assert_array_equal(p[-1,3:],[0,0,1])
        self.assertEqual(iou(np.zeros((2,2)),np.zeros((2,2))),0)

    def test_qc_rejects_bad_fidelity(self):
        p=states(np.array([[30,30,6,0,0,0],[60,30,8,0,0,0]]),np.array([0,2]))
        z=dict(points=p,subpath_offsets=np.array([0,2]))
        im=render(p,z['subpath_offsets'])
        _,q=evaluate(z,im>=128,np.full((256,256),255,np.uint8))
        self.assertEqual(q['qc_status'],'rejected')

    def test_sources_reference_review_and_export(self):
        image=self.d/'source.png'; im=np.full((50,100),255,np.uint8); im[20:30,20:80]=0; png(image,im)
        src=self.d/'source.jsonl'; write_records(src,[dict(sample_id='a',image_path=str(image),character='一',writer_id='w1',source='fixture')])
        self.cli('prepare_sources','--manifest',src,'--output',self.d/'normalized')
        norm=json.loads((self.d/'normalized/manifest.jsonl').read_text())
        self.assertEqual(norm['transform']['resized_size'],[240,120])
        mask=np.zeros((1,256,256),np.uint8); mask[0,30:40,20:100]=1
        npz=self.d/'mask.npz'; np.savez(str(npz),extractnet_strokes=mask,stroke_labels=np.array([0]),reference_single_image=mask,stroke_label=np.array([0]))
        refs=self.d/'refs.jsonl'; write_records(refs,[dict(character='一',npz_path=str(npz),source='fixture',category_system='StrokeExtraction-24')])
        self.cli('build_reference_bank','--manifest',refs,'--output',self.d/'references')
        extracted=self.d/'extracted.jsonl'; write_records(extracted,[dict(norm,extracted_path=str(npz))])
        self.cli('review_strokes','--manifest',extracted,'--output',self.d/'preview')
        self.cli('export_vs_inputs','--manifest',self.d/'preview/manifest.jsonl','--output',self.d/'blocked',ok=False)
        decisions=json.loads((self.d/'preview/decisions.template.json').read_text()); decisions['a'].update(status='approved',reviewer='test-fixture')
        dec=self.d/'decisions.json'; dec.write_text(json.dumps(decisions))
        self.cli('review_strokes','--manifest',extracted,'--output',self.d/'approved','--decisions',dec)
        self.cli('export_vs_inputs','--manifest',self.d/'approved/manifest.jsonl','--output',self.d/'vs')
        self.assertTrue((self.d/'vs/a_stroke000.png').exists())
        decisions['a']['source_sha256']='bad'; dec.write_text(json.dumps(decisions))
        self.cli('review_strokes','--manifest',extracted,'--output',self.d/'stale','--decisions',dec,ok=False)

    def test_pack_validate_and_tamper(self):
        from pack_dataset import pack_group
        p=states(np.array([[40,40,10,0,0,0],[70,40,10,0,0,0]],np.float32),np.array([0,2]))
        path=self.d/'trajectory.npz'; np.savez(str(path),points=p,subpath_offsets=np.array([0,2]))
        image=png(self.d/'image.png',255-render(p,np.array([0,2])))
        reviewed=self.d/'mask.npz'; np.savez(str(reviewed),extractnet_strokes=(render(p,np.array([0,2]))>=128)[None].astype(np.uint8),stroke_labels=np.array([0]))
        row=dict(sample_id='a',character='一',source='test',writer_id='w',image_path=image,expected_strokes=1,stroke_index=0,stroke_label=0,review_status='approved',mask_review=[dict(reviewer='fixture')],trajectory_review_status='approved',trajectory_review=[dict(reviewer='fixture')],qc=dict(qc_status='passed',iou=1.,upstream_iou=1.),trajectory_path=str(path),trajectory_sha256=digest(path),reviewed_path=str(reviewed),reviewed_sha256=digest(reviewed),input_png=image,input_sha256=digest(image))
        row.update(upstream_render_path=image,upstream_render_sha256=digest(image))
        dest=self.d/'packed'; dest.mkdir(); meta=pack_group([row],dest)
        self.assertTrue(validate([meta])['valid'])
        other=dict(meta,sample_id='b',split='test' if meta['split']=='train' else 'train')
        self.assertFalse(validate([meta,other])['valid'])
        meta['npz_sha256']='bad'; self.assertFalse(validate([meta])['valid'])
        row['trajectory_review_status']='pending'
        with self.assertRaises(ValueError): pack_group([row],dest)

    def test_gnt_import_and_truncation(self):
        from prepare_sources import expand_sources
        im=np.full((10,20),255,np.uint8); im[3:6,4:15]=0
        gnt=self.d/'writer.gnt'; gnt.write_bytes(struct.pack('<I',210)+'一'.encode('gbk')+struct.pack('<HH',20,10)+im.tobytes())
        row=dict(gnt_path=str(gnt),writer_id='writer',source='fixture')
        result=expand_sources([row],self.d/'rasters')
        self.assertEqual(result[0]['character'],'一'); self.assertEqual(result[0]['gnt_offset'],0)
        gnt.write_bytes(gnt.read_bytes()[:-1])
        with self.assertRaises(ValueError): expand_sources([row],self.d/'bad')

    def test_trajectory_review_reverses_width_and_rechecks(self):
        p=states(np.array([[30,30,4,0,0,0],[70,30,10,0,0,0]],np.float32),np.array([0,2]))
        path=self.d/'points.npz'; np.savez(str(path),points=p,subpath_offsets=np.array([0,2]))
        image=png(self.d/'image.png',255-render(p,np.array([0,2])))
        row=dict(stroke_id='a_stroke000',trajectory_path=str(path),input_png=image,upstream_render_path=image,qc=dict(threshold=.95,fidelity_threshold=.95))
        manifest=self.d/'qc.jsonl'; write_records(manifest,[row])
        decisions=self.d/'decision.json'; decisions.write_text(json.dumps(dict(a_stroke000=dict(source_sha256=digest(path),status='approved',reviewer='fixture',subpaths=[dict(index=0,reverse=True)]))))
        self.cli('review_trajectories','--manifest',manifest,'--decisions',decisions,'--output',self.d/'reviewed')
        z=arrays(self.d/'reviewed/a_stroke000/reviewed.npz')
        np.testing.assert_allclose(z['points'][0,:3],[70,30,10]); np.testing.assert_array_equal(z['points'][-1,3:],[0,0,1])
        decisions.write_text(json.dumps(dict(a_stroke000=dict(source_sha256='wrong',status='approved',reviewer='fixture'))))
        self.cli('review_trajectories','--manifest',manifest,'--decisions',decisions,'--output',self.d/'stale',ok=False)

    def test_pipeline_dry_run_and_resume(self):
        script=self.d/'config.json'; manifest=self.d/'empty.jsonl'; manifest.write_text('')
        cfg=dict(report=str(self.d/'run.json'),stages=[dict(id='normalize',script='prepare_sources',python=sys.executable,args=['--manifest',str(manifest),'--output',str(self.d/'normalized')],inputs=[str(manifest)],outputs=[str(self.d/'normalized/manifest.jsonl')])])
        script.write_text(json.dumps(cfg))
        self.cli('run_pipeline','--config',script,'--dry-run'); self.assertFalse((self.d/'normalized').exists())
        self.cli('run_pipeline','--config',script)
        self.cli('run_pipeline','--config',script,'--resume')


if __name__=='__main__': unittest.main()
