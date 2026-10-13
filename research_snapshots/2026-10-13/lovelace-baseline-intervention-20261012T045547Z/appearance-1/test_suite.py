"""Load exact A4 targets despite historical sys.path mutation."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent


def from_exact_file(name,filename):
    path=ROOT/filename
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve()!=path.resolve():
        raise RuntimeError('substitute target in tests')
    return module


def suite():
    for name in ('appearance_solver','run_appearance','verify_appearance','execute_appearance'):
        from_exact_file(name,name+'.py')
    result=unittest.TestSuite()
    for filename in ('test_appearance_solver.py','test_run_appearance.py','test_verify_appearance.py','test_execute_appearance.py'):
        module=from_exact_file('a4_exact_'+filename[:-3],filename)
        result.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    return result


if __name__=='__main__':
    outcome=unittest.TextTestRunner(verbosity=2).run(suite())
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
