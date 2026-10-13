"""Load exact phase targets despite historical modules prepending sys.path."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def from_exact_file(name, filename):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError('test imported a substitute target')
    return module


def suite():
    for name in ('ablation_solver', 'run_ablation', 'verify_ablation'):
        from_exact_file(name, name + '.py')
    result = unittest.TestSuite()
    for filename in ('test_ablation_solver.py', 'test_run_ablation.py', 'test_verify_ablation.py'):
        module = from_exact_file('exact_phase_' + filename[:-3], filename)
        result.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    return result


if __name__ == '__main__':
    outcome = unittest.TextTestRunner(verbosity=2).run(suite())
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
