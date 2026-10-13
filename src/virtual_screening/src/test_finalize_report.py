import unittest
from finalize_report import join_rankings


class FinalReportChecks(unittest.TestCase):
    def setUp(self):
        self.row=dict(rank='1',ligand_id='A',name='test',successful_boxes='8',complete='True')
        self.catalog={'A':dict(cas='testCAS',formula='C',molecular_weight='12')}
        self.quality={'A':dict(identifier='A',name='test',geometrywarnings='severe_bond_stretch',fallback_used='True')}

    def test_quality_warning_and_catalog_are_retained(self):
        result=join_rankings([self.row],self.catalog,self.quality)[0]
        self.assertEqual(result['geometrywarnings'],'severe_bond_stretch')
        self.assertEqual(result['cas'],'testCAS')
        self.assertEqual(result['fallback_used'],'True')

    def test_missing_quality_annotation_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Missing source or conformer'):
            join_rankings([self.row],self.catalog,{})

    def test_partial_ligand_cannot_be_presented_as_final(self):
        self.row.update(successful_boxes='7',complete='False')
        with self.assertRaisesRegex(ValueError,'eight-box-complete'):
            join_rankings([self.row],self.catalog,self.quality)


if __name__=='__main__':unittest.main()
