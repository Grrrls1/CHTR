from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent))
from models.visualization.mask_types import TYPE_COLORS,color_masks,type_entries,write_gallery
from models.scripts.export_mask_overlays import add_npz_fields


class MaskTypeVisualizationTests(unittest.TestCase):
    def test_palette_type_consistency_and_overlap(self):
        self.assertEqual(len(TYPE_COLORS),24);self.assertEqual(len(set(TYPE_COLORS)),24)
        masks=np.zeros((2,3,3),np.float32);masks[0,0,:2]=1;masks[1,:2,1]=1
        image=np.full((3,3),255,np.uint8)
        drawn=color_masks(masks,[1,2],image,annotate=False)
        c=np.asarray(TYPE_COLORS,np.float32)[:,::-1]
        np.testing.assert_array_equal(drawn[0,0],np.rint(.85*c[1]+.15*255).astype(np.uint8))
        np.testing.assert_array_equal(drawn[1,1],np.rint(.85*c[2]+.15*255).astype(np.uint8))
        np.testing.assert_array_equal(drawn[0,1],np.rint(.85*(c[1]+c[2])/2+.15*255).astype(np.uint8))
        np.testing.assert_array_equal(drawn[2,2],[255,255,255])
        same=color_masks(masks,[1,1],image,annotate=False)
        np.testing.assert_array_equal(same[0,1],same[0,0])

    def test_entries_and_input_validation(self):
        masks=np.ones((1,3,3),np.float32);probs=np.zeros((1,24));probs[0,2]=.9
        entries=type_entries([2],masks,{2:'竖'},probs)
        self.assertEqual(entries[0]['type_name'],'竖');self.assertEqual(entries[0]['stroke_number'],1)
        self.assertEqual(entries[0]['mask_pixels'],9);self.assertEqual(entries[0]['type_confidence'],.9)
        with self.assertRaises(ValueError):color_masks(masks,[24],np.ones((3,3),np.uint8))
        with self.assertRaises(ValueError):color_masks(masks*2,[2],np.ones((3,3),np.uint8))

    def test_npz_preserves_original_trajectory_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'prediction.npz';points=np.arange(12,dtype=np.float32).reshape(2,6)
            offsets=np.array([0,2],np.int64)
            np.savez_compressed(path,points=points,stroke_offsets=offsets,mode=np.array('image_only'))
            add_npz_fields(path,dict(stroke_masks=np.ones((1,3,3),np.float32)))
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['points'],points);np.testing.assert_array_equal(z['stroke_offsets'],offsets)
                self.assertEqual(str(z['mode']),'image_only');self.assertEqual(z['stroke_masks'].shape,(1,3,3))

    def test_legend_html_is_complete_and_escaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);names={i:'类型'+str(i) for i in range(24)}
            write_gallery(folder,[dict(sample_id='a',character='<一>')],names)
            text=(folder/'index.html').read_text()
            self.assertIn('&lt;一&gt;',text);self.assertIn('predicted types',text.lower())
            self.assertTrue((folder/'mask_type_legend.json').is_file())


if __name__=='__main__':unittest.main()
