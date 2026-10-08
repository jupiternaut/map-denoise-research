"""Implementation checks and two-object timing only, before calibration lock."""
import importlib
import io
import platform
import sys
import time
import unittest
import numpy as np
import experiment as e

def main():
    stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(e.ROOT),pattern='test_*.py')
    outcome=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True)
    e.dump(e.ROOT/'B0/RESULTS.json',dict(created=e.now(),passed=outcome.wasSuccessful(),tests=outcome.testsRun,failures=len(outcome.failures),errors=len(outcome.errors),transcript=stream.getvalue(),python=sys.executable,platform=platform.platform(),numpy=np.__version__))
    assert outcome.wasSuccessful()
    import new_fixtures as f
    timings=[]
    for world in f.object_specs('calibration')[:2]:
        t=time.perf_counter();cams=f.cameras();images=f.render_object(world,cams);render=time.perf_counter()-t
        t=time.perf_counter();a=e.predictor.estimate_aux(images[0],images[1],cams[0],cams[1]);fit=time.perf_counter()-t
        t=time.perf_counter();r=e.predictor.predict_curve(a,cams[0],cams[2],images[2],np.arange(450.,901.,2.),600.);score=time.perf_counter()-t
        timings.append(dict(render_seconds=render,aux_seconds=fit,fold_score_seconds=score,shape=list(images.shape),raw_valid=r['valid']))
    e.dump(e.ROOT/'B0/TIMING.json',dict(objects=2,known_calibration_only=True,measurements=timings,scientific_rules_unchanged=True))
    print(timings)

if __name__=='__main__':main()
