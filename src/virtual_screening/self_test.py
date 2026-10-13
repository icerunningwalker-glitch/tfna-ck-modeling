"""Run bundled software tests without launching real screening or installing tools."""
import argparse
import importlib.util
import json
import sys
import unittest
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-chemistry',action='store_true')
    args=parser.parse_args(argv)
    modules=['test_run_screen','test_preparation_guards','test_finalize_report','test_portable']
    available=importlib.util.find_spec('rdkit') is not None
    if args.require_chemistry and not available:parser.error('RDKit is required for chemistry tests')
    if available:modules+=['test_chemistry_geometry_checks','test_rigid_conformer_fallback']
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    print(json.dumps(dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                         chemistry_tests=available,optional_chemistry_skipped=not available)))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
