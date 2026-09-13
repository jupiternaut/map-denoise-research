"""Known-geometry diagnostic ONLY, not the fitted-model confirmation experiment.

Run from project root with -m exploration_v19.track_b.oracle_diagnostic.
Saves aggregate JSON only; never writes the large simulated arrays.
"""
import argparse
import importlib.util
import json
from math import erf, log, sqrt
from pathlib import Path
import time
import numpy as np
from scipy.special import expit
from scipy.stats import wasserstein_distance

spec = importlib.util.spec_from_file_location("v19_track_b_operator", Path(__file__).with_name("operator.py"))
op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)


def experiment(n=200000, seed=9193201):
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    upper = rng.random(n) < .2
    truth = 2.*upper-1.
    measured = truth + rng.normal(0., 2., n)
    x = np.c_[np.ones(n), np.zeros((n, 2))]
    results = []
    true_count_mass = float(upper.mean())
    for assumed_sigma in (1., 2., 4.):
        # Exactly correct posterior ONLY when assumed_sigma equals 2 mm.
        q = expit(log(.2/.8) + 2.*measured/assumed_sigma**2)
        model = dict(k=2, means=np.array([-1., 1.]), slope=np.zeros(2),
                     posterior=np.c_[1-q, q], family="known_geometry_diagnostic",
                     true_parameters_given=True, assumed_sigma=assumed_sigma)
        for name, out in op.run(x, measured, assumed_sigma, model).items():
            row = dict(method=name, assumed_sigma_mm=assumed_sigma,
                       information_condition="true means/prior, exact posterior" if assumed_sigma == 2. else "true means/prior, misspecified sigma",
                       posterior_upper_mass=float(q.mean()), posterior_brier=float(np.mean((q-upper)**2)))
            if name == "weighted_measure":
                heights, weights = out["support_heights"], out["support_weights"]
                predicted_mass = float(weights[:, 1].mean())
                row.update(upper_mass=predicted_mass,
                           population_marginal_w1_mm=2.*abs(predicted_mass-.2),
                           realized_latent_marginal_w1_mm=2.*abs(predicted_mass-true_count_mass),
                           true_surface_distance_mm=0.,
                           integral_source_rms_mm=float(np.sqrt(np.mean(np.sum(weights*(heights-truth[:, None])**2, axis=1)))),
                           deterministic_source_rms_mm=None,
                           expected_mislabel_fraction=float(np.mean(np.where(upper, weights[:, 0], weights[:, 1]))))
            else:
                pred = out["prediction"]
                groups = out["groups"]
                row.update(upper_mass=None if groups is None else float(groups.mean()),
                           population_marginal_w1_mm=float(wasserstein_distance(pred, [-1., 1.], v_weights=[.8, .2])),
                           realized_latent_marginal_w1_mm=float(wasserstein_distance(pred, [-1., 1.], v_weights=[1-true_count_mass, true_count_mass])),
                           true_surface_distance_mm=float(np.mean(np.minimum(abs(pred+1), abs(pred-1)))),
                           deterministic_source_rms_mm=float(np.sqrt(np.mean((pred-truth)**2))),
                           integral_source_rms_mm=None,
                           expected_mislabel_fraction=None if groups is None else float(np.mean(groups != upper)))
            if row["upper_mass"] is not None:
                row["upper_mass_error_to_population"] = abs(row["upper_mass"]-.2)
                row["upper_mass_error_to_realized_latent"] = abs(row["upper_mass"]-true_count_mass)
            results.append(row)
    phi = lambda z: .5*(1+erf(z/sqrt(2)))
    threshold = 2.*log(4.)
    fn = phi((threshold-1)/2)
    fp = 1-phi((threshold+1)/2)
    analytic = dict(map_threshold_mm=threshold, map_upper_mass=.2*(1-fn)+.8*fp,
                    map_mislabel_fraction=.2*fn+.8*fp,
                    map_source_rms_mm=2.*sqrt(.2*fn+.8*fp))
    return dict(evidence_type="ORACLE-ONLY mechanism diagnostic; no fitted-model gain claim",
                n=n, seed=seed, true_means_mm=[-1., 1.], true_upper_probability=.2, true_sigma_mm=2.,
                realized_true_upper_mass=true_count_mass,
                interpretation=["True surfaces and prior are supplied, not estimated.",
                                "True-surface error zero is tautological for on-surface outputs.",
                                "Weighted integral RMS is not a deterministic point-estimate RMS.",
                                "A posterior-mass quota can be accurate internally while physically wrong under sigma mismatch.",
                                "Sample marginal mass is not physical surface area."],
                exact_posterior_map_analytic=analytic, results=results,
                seconds=time.perf_counter()-started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = experiment()
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(dict(output=str(args.output), n=result["n"], rows=len(result["results"]), seconds=result["seconds"])))
