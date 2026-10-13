"""Portability and full-wrapper checks with synthetic inputs only."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import demo
import screen
import run_screen
from bootstrap_screen import build_boxes
from execute_pipeline import freeze_eligible_manifest


class PortableChecks(unittest.TestCase):
    def test_equal_boxes_cover_and_are_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);_,receptor=demo.fixtures(folder/'inputs')
            result=build_boxes(receptor,folder/'run')
            self.assertEqual(result['heavy_atoms'],result['covered_heavy_atoms'])
            with (folder/'run'/'site_jobs.csv').open(encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),8)
            self.assertEqual(len({(row['size_x'],row['size_y'],row['size_z']) for row in rows}),1)
            self.assertTrue(all(row['receptor']=='receptor.pdbqt' for row in rows))
            with self.assertRaisesRegex(ValueError,'Frozen boxes changed'):build_boxes(receptor,folder/'run',9)

    def test_manifest_paths_survive_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);manifest,_=demo.fixtures(folder/'inputs')
            frozen,info=freeze_eligible_manifest(manifest,folder/'run')
            self.assertEqual(info['prepared_ligands'],2)
            with frozen.open(encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
            self.assertTrue(all((frozen.parent/row['pdbqt']).is_file() for row in rows))

    def test_one_command_exports_candidates_and_structures(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'example'
            self.assertEqual(demo.main(['--output-dir',str(output)]),0)
            root=output/'run'
            with (root/'results'/'synthetic_results.csv').open(encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['candidate_id'],'SYN001')
            self.assertEqual(rows[0]['predicted_hit_box_count_8'],'8')
            self.assertTrue(all((root/'results'/row['structure_file']).is_file() for row in rows))
            summary=json.loads((root/'ranking_summary.json').read_text())
            self.assertEqual(summary['successful_jobs'],16)
            self.assertFalse(summary['quality_annotations_supplied'])
            self.assertFalse((root/'results'/'results.csv').exists())
            self.assertEqual(summary['evidence_type'],'synthetic_software_test')
            self.assertTrue(all(row['scientific_candidate']=='False' for row in rows))


if __name__=='__main__':unittest.main()
