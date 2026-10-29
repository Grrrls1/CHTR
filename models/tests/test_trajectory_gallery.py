import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from models.visualization.trajectory_gallery import trajectory_svg, pack_paths, write_character_gallery, write_prediction_gallery, write_compact_gallery


class TrajectoryGalleryTests(unittest.TestCase):
    def test_stroke_boundaries_and_single_point(self):
        paths = [np.array([[10,20,3],[30,40,4]],np.float32),np.array([[90,100,5]],np.float32)]
        points, offsets = pack_paths(paths)
        self.assertEqual(offsets.tolist(),[0,2,3])
        np.testing.assert_array_equal(points,np.concatenate(paths))
        svg = ET.fromstring(trajectory_svg(paths))
        ns = {'s':'http://www.w3.org/2000/svg'}
        lines = svg.findall('s:line',ns)
        # Only the first stroke has a segment: never connect (30,40) to (90,100).
        self.assertEqual(len(lines),1)
        self.assertEqual(lines[0].attrib['x2'],'30.000')
        self.assertEqual(svg.attrib['data-total'],'3')
        self.assertEqual(len(svg.findall('s:g',ns)),2)

    def test_duplicate_points_and_empty_prediction_are_renderable(self):
        ET.fromstring(trajectory_svg([np.array([[255,0,1],[255,0,1]],np.float32)]))
        self.assertIn('未预测到笔画',trajectory_svg([]))
        points, offsets = pack_paths([])
        self.assertEqual(points.shape,(0,3))
        self.assertEqual(offsets.tolist(),[0])
        with tempfile.TemporaryDirectory() as tmp:
            write_prediction_gallery(tmp,np.full((256,256),255,np.uint8),[])
            self.assertTrue((Path(tmp)/'index.html').is_file())

    def test_partial_gallery_is_not_marked_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = np.array([[10,20,3]],np.float32)
            np.savez(folder/'stroke001.npz',ground_truth_xyw_px=path,predicted_xyw_px=path)
            write_character_gallery(folder,'<test>',expected_strokes=2)
            with np.load(folder/'character_trajectory.npz') as data:
                self.assertFalse(bool(data['complete_character']))
                self.assertEqual(data['stroke_indices'].tolist(),[1])
                self.assertEqual(data['stroke_offsets'].tolist(),[0,1])
            self.assertIn('&lt;test&gt;', (folder/'index.html').read_text())
            self.assertIn('data-player', (folder/'index.html').read_text())

    def test_new_gallery_has_only_whole_character_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=np.array([[10,20,3],[40,50,4]],np.float32)
            write_compact_gallery(tmp,'example',np.full((256,256),255,np.uint8),[path,path],[path])
            self.assertEqual({p.name for p in Path(tmp).iterdir()},{'index.html','character_trajectory.npz'})
            self.assertNotIn('<article>',(Path(tmp)/'index.html').read_text())
            with np.load(Path(tmp)/'character_trajectory.npz') as data:
                self.assertEqual(data['stroke_offsets'].tolist(),[0,2])
                self.assertEqual(data['ground_truth_stroke_offsets'].tolist(),[0,2,4])


if __name__ == '__main__':
    unittest.main()
