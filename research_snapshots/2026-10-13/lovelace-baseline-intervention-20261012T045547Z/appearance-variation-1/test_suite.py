"""Nonconfirmation test suite; never launches the registered generation."""
import unittest
if __name__=='__main__':
    names=('test_variation_solver','test_variation_pipeline','test_verify_variation')
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in names)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

