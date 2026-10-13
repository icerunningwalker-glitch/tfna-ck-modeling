"""Finite-charge guards and transparent, immutable eligibility manifests."""
import csv
from pathlib import Path
import tempfile
import unittest

from execute_pipeline import freeze_eligible_manifest
from prepare_library import audit_pdbqt


class PreparationGuards(unittest.TestCase):
    def test_nonfinite_charge_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ligand.pdbqt'
            for charge in ('nan','inf','-inf'):
                atom=list(' '*80)
                atom[0:6]='ATOM  '
                for first,last in ((30,38),(38,46),(46,54)):
                    atom[first:last]=f'{0:8.3f}'
                atom[70:76]=f'{charge:>6}'
                atom[77]='C'
                path.write_text('ROOT\n'+''.join(atom)+'\nENDROOT\n',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'Nonfinite PDBQT charge'):
                    audit_pdbqt(path,1,allow_planar=True)

    def write_manifest(self,path,rows):
        with path.open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['identifier','name','pdbqt','status','detail'])
            writer.writeheader();writer.writerows(rows)

    def test_failed_source_id_is_retained_and_unranked(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            source=folder/'source.csv'
            self.write_manifest(source,[dict(identifier='A',name='valid',pdbqt='A.pdbqt',status='prepared',detail=''),
                                        dict(identifier='B',name='failed',pdbqt='',status='failed',detail='source stereo not preserved')])
            prepared,evidence=freeze_eligible_manifest(source,folder/'outputs',expected_ligands=2)
            self.assertEqual(evidence['requested_jobs'],16)
            self.assertEqual(evidence['docking_jobs'],8)
            self.assertEqual(evidence['excluded_ids'],['B'])
            with prepared.open(encoding='utf-8-sig',newline='') as handle:
                self.assertEqual([row['identifier'] for row in csv.DictReader(handle)],['A'])
            self.assertIn('source stereo not preserved',(folder/'outputs'/'preparation_exclusions.csv').read_text(encoding='utf-8-sig'))
            self.assertEqual(freeze_eligible_manifest(source,folder/'outputs',2)[1],evidence)

    def test_changed_eligibility_and_duplicate_ids_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);source=folder/'source.csv'
            rows=[dict(identifier='A',name='a',pdbqt='A.pdbqt',status='prepared',detail=''),
                  dict(identifier='B',name='b',pdbqt='',status='failed',detail='unprepared')]
            self.write_manifest(source,rows)
            freeze_eligible_manifest(source,folder/'outputs',2)
            frozen=(folder/'outputs'/'prepared_ligand_manifest.csv').read_bytes()
            rows[1].update(status='prepared',pdbqt='B.pdbqt',detail='')
            self.write_manifest(source,rows)
            with self.assertRaisesRegex(ValueError,'Frozen screening manifest changed'):
                freeze_eligible_manifest(source,folder/'outputs',2)
            self.assertEqual((folder/'outputs'/'prepared_ligand_manifest.csv').read_bytes(),frozen)
            rows[1]['identifier']='A';self.write_manifest(source,rows)
            with self.assertRaisesRegex(ValueError,'all unique source catalog IDs'):
                freeze_eligible_manifest(source,folder/'other',2)


if __name__=='__main__':unittest.main()
