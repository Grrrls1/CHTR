import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import digest
from import_calligraphy_dataset import normalize
from import_ccse_kaiti_sequence import rasterize, match_ccse_records
from link_calligraphy_sequence import make_links, diagnostic_iou
from verify_sequence_import import verify
from import_makemeahanzi import render_record
from link_template_sources import make_links as make_multi_source_links


class NewImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_calligraphy_normalization_and_provenance(self):
        source = self.root / '42.jpg'
        image = np.full((40, 60), 255, np.uint8)
        image[10:30, 15:45] = 0
        cv2.imwrite(str(source), image)
        row = normalize((source, '一', self.root, 8))
        result = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
        self.assertEqual(result.shape, (256, 256))
        self.assertLess(result[128, 128], 20)
        self.assertEqual(result[0, 0], 255)
        self.assertEqual(row['sample_id'], 'ccd_u04E00_42')
        self.assertEqual(row['source_sha256'], digest(source))
        self.assertFalse(row['pack_eligible'])
        self.assertEqual(row['provenance_status'], 'writer_and_style_unknown')
        image[:] = 0
        cv2.imwrite(str(source), image)
        with self.assertRaisesRegex(ValueError, 'dark background'):
            normalize((source, '一', self.root, 8))

    def test_svg_direction_medians_and_missing_categories(self):
        # The first path is near y=900 (top); the second is near y=100 (bottom).
        record = dict(character='一', image_id='0001',
                      strokes=['M 100 880 L 300 880 L 300 920 L 100 920 Z',
                               'M 100 80 L 300 80 L 300 120 L 100 120 Z'],
                      medians=[[[100, 900], [300, 900]], [[100, 100], [300, 100]]],
                      category_id=[8, 8])
        row = rasterize(record, self.root)
        self.assertEqual(row['ordered_strokes_sha256'], digest(row['ordered_strokes_path']))
        with np.load(row['ordered_strokes_path'], allow_pickle=False) as z:
            masks = z['ordered_stroke_masks']
            self.assertEqual(masks.shape, (2, 256, 256))
            self.assertLess(np.where(masks[0])[0].mean(), np.where(masks[1])[0].mean())
            np.testing.assert_allclose(z['median_000'], [[25, 0], [75, 0]])
            np.testing.assert_allclose(z['median_001'], [[25, 200], [75, 200]])
            np.testing.assert_array_equal(z['stroke_order'], [0, 1])
            self.assertEqual(str(z['category_system']), 'CCSE-25')
        self.assertGreater(diagnostic_iou(row['preview_path'], row['preview_path']), .99)
        record.update(character='丁', image_id='0002', category_id=[8])
        partial = rasterize(record, self.root)
        self.assertEqual(partial['category_status'], 'unknown_due_to_upstream_missing_annotations')
        with np.load(partial['ordered_strokes_path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['stroke_label'], [-1, -1])
            np.testing.assert_array_equal(z['raw_partial_stroke_label'], [8])

    def test_official_makemeahanzi_unlabeled_raster(self):
        record = dict(character='一',
                      strokes=['M 100 880 L 300 880 L 300 920 L 100 920 Z'],
                      medians=[[[100, 900], [300, 900]]])
        row = render_record(record, self.root)
        self.assertEqual(row['ordered_strokes_sha256'], digest(row['ordered_strokes_path']))
        self.assertFalse(row['strokeextraction_24_compatible'])
        with np.load(row['ordered_strokes_path'], allow_pickle=False) as z:
            self.assertEqual(z['ordered_stroke_masks'].shape, (1, 256, 256))
            self.assertGreater(z['ordered_stroke_masks'].sum(), 0)
            np.testing.assert_array_equal(z['stroke_label'], [-1])
            np.testing.assert_allclose(z['median_000'], [[25, 0], [75, 0]])
            self.assertEqual(str(z['category_system']), 'unlabeled')

    def test_multi_source_candidates_keep_categories_separate(self):
        call = [dict(schema='chtr-local-v1', sample_id='x', character='一',
                     image_path='/x.png', source_image_path='/x.jpg', source_sha256='h',
                     provenance_status='writer_and_style_unknown', pack_eligible=False)]
        ccse = [dict(character='一', ordered_strokes_path='/c.npz', ordered_strokes_sha256='c',
                     preview_path='/c.png', stroke_count=1, category_system='CCSE-25',
                     category_status='complete', strokeextraction_24_compatible=False)]
        mmah = [dict(character='一', ordered_strokes_path='/m.npz', ordered_strokes_sha256='m',
                     preview_path='/m.png', stroke_count=1, category_system='unlabeled',
                     strokeextraction_24_compatible=False)]
        rhsedb = [dict(character='一', reference_path='/r.npz', reference_sha256='r',
                       stroke_count=1, source_split='train', category_system='StrokeExtraction-24')]
        rows, summary = make_multi_source_links(call, ccse, mmah, rhsedb, shared_geometry={"一"})
        self.assertEqual(len(rows[0]['template_candidates']), 3)
        self.assertEqual([c['category_system'] for c in rows[0]['template_candidates']],
                         ['CCSE-25', 'unlabeled', 'StrokeExtraction-24'])
        self.assertEqual([c['strokeextraction_24_compatible'] for c in rows[0]['template_candidates']],
                         [False, False, True])
        self.assertEqual(summary['no_template_images'], 0)
        self.assertTrue(rows[0]['template_candidates'][1]['same_geometry_as_lee_xeo'])
        self.assertEqual(rows[0]['review_status'], 'pending')

    def test_ccse_alignment_by_character_and_class_multiset(self):
        records = [dict(character='一', image_id='not-coco-id', category_id=[8, 1]),
                   dict(character='丁', image_id='also-not-coco-id', category_id=[25])]
        images = {12: dict(id=12, font='丁'), 99: dict(id=99, font='一')}
        annotations = [dict(image_id=99, category_id=1),
                       dict(image_id=99, category_id=8), dict(image_id=12, category_id=25)]
        matched, mismatches = match_ccse_records(records, images, annotations)
        self.assertEqual(matched['一']['id'], 99)
        self.assertEqual(mismatches, [])
        annotations[0]['category_id'] = 2
        _, mismatches = match_ccse_records(records, images, annotations)
        self.assertEqual(mismatches[0]['character'], '一')
        with self.assertRaisesRegex(ValueError, 'character sets differ'):
            match_ccse_records(records, {12: images[12]}, annotations)

    def test_container_hash_and_category_map_verification(self):
        record = dict(character='一', image_id='0001',
                      strokes=['M 100 880 L 300 880 L 300 920 L 100 920 Z'],
                      medians=[[[100, 900], [300, 900]]], category_id=[8])
        generated = self.root / 'generated'
        generated.mkdir()
        row = rasterize(record, generated)
        manifest = generated / 'manifest.jsonl'
        manifest.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
        graphics = self.root / 'graphics.json'
        graphics.write_text(json.dumps([record], ensure_ascii=False), encoding='utf-8')
        categories = [dict(id=i, name=str(i)) for i in range(1, 26)]
        (generated / 'category_map.json').write_text(json.dumps(dict(category_system='CCSE-25', categories=categories)))
        ccse = self.root / 'ccse' / 'annotations'
        ccse.mkdir(parents=True)
        for split in ('train', 'val', 'test'):
            (ccse / ('instances_%s2021.json' % split)).write_text(json.dumps(dict(categories=categories)))
        report = verify(manifest, graphics, ccse.parent)
        self.assertEqual(report['verified_npz'], 1)
        self.assertTrue(report['category_map_matches_all_ccse_splits'])
        categories[0]['name'] = 'changed'
        (ccse / 'instances_val2021.json').write_text(json.dumps(dict(categories=categories)))
        with self.assertRaisesRegex(ValueError, 'category map differs'):
            verify(manifest, graphics, ccse.parent)
        (ccse / 'instances_val2021.json').write_text(json.dumps(dict(categories=[dict(id=i, name=str(i)) for i in range(1, 26)])))
        Path(row['ordered_strokes_path']).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'NPZ hash mismatch'):
            verify(manifest, graphics, ccse.parent)

    def test_character_join_never_uses_image_ids_or_claims_registration(self):
        base = dict(schema='chtr-local-v1', pack_eligible=False,
                    provenance_status='writer_and_style_unknown', source_sha256='a',
                    image_path='/a.png', source_image_path='/a.jpg')
        calligraphy = [dict(base, character='一', sample_id='a'),
                       dict(base, character='丁', sample_id='b'),
                       dict(base, character='七', sample_id='c')]
        seq = [dict(character='一', sample_id='template1', image_id='9999',
                    category_system='CCSE-25', category_status='complete',
                    strokeextraction_24_compatible=False, ordered_strokes_path='/t.npz',
                    ordered_strokes_sha256='h', preview_path='/t.png',
                    ccse_image_path='/c.jpg', ccse_split='test', stroke_count=2),
               dict(character='丁', sample_id='template2', image_id='a',
                    category_system='CCSE-25', category_status='unknown_due_to_upstream_missing_annotations',
                    strokeextraction_24_compatible=False, ordered_strokes_path='/u.npz',
                    ordered_strokes_sha256='i', preview_path='/u.png',
                    ccse_image_path='/d.jpg', ccse_split='train', stroke_count=3)]
        joined, summary = make_links(calligraphy, seq)
        self.assertEqual([r['match_status'] for r in joined],
                         ['candidate_review_required', 'candidate_labels_incomplete', 'no_same_character_template'])
        self.assertEqual(joined[0]['template_sample_id'], 'template1')
        self.assertEqual(joined[0]['registration_status'], 'not_evaluated')
        self.assertFalse(joined[0]['strokeextraction_24_compatible'])
        self.assertNotIn('template_sample_id', joined[2])
        self.assertEqual(summary['unmatched_calligraphy_characters'], 1)
        self.assertEqual(summary['template_characters_with_incomplete_labels'], 1)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            make_links(calligraphy, seq + [seq[0]])
        with self.assertRaisesRegex(ValueError, 'provenance'):
            make_links([dict(calligraphy[0], pack_eligible=True)], seq)


if __name__ == '__main__':
    unittest.main()
