"""Read-only exact summary of verified view-bias decisions."""
from collections import defaultdict
from fractions import Fraction as Q
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/view-bias-v1"
DEST = ROOT / "outputs/analysis-v1"

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def summarize(rows):
    offsets = [r for r in rows if Q(r["initial_ae_mm"]) != 0]
    correct = [r for r in rows if Q(r["initial_ae_mm"]) == 0]
    negative = [r for r in offsets if r["case"] == "equal_full_negative"]
    errors = [Q(r["ae_mm"]) for r in offsets]
    initial = [Q(r["initial_ae_mm"]) for r in offsets]
    mean = sum(errors, Q(0)) / len(offsets)
    mse = sum((x*x for x in errors), Q(0)) / len(offsets)
    before = sum(initial, Q(0)) / len(offsets)
    before_mse = sum((x*x for x in initial), Q(0)) / len(offsets)
    observations = {}
    for r in rows:
        old = observations.setdefault(r["observation_id"], r["covered"])
        if old != r["covered"]:
            raise ValueError("coverage changes with the initial state")
    return {
        "n_offset": len(offsets),
        "improved": sum(e < i for e, i in zip(errors, initial)),
        "same": sum(e == i for e, i in zip(errors, initial)),
        "harm": sum(e > i for e, i in zip(errors, initial)),
        "mae_exact_model_mm": str(mean),
        "mae_model_mm": float(mean),
        "mse_exact_model_mm2": str(mse),
        "mse_model_mm2": float(mse),
        "mae_reduction_percent_vs_keep": float(100*(before-mean)/before),
        "mse_reduction_percent_vs_keep": float(100*(before_mse-mse)/before_mse),
        "n_correct": len(correct),
        "correct_kept": sum(Q(r["output"]) == Q(r["incumbent"]) for r in correct),
        "correct_harm": sum(Q(r["ae_mm"]) > 0 for r in correct),
        "n_negative_offset": len(negative),
        "negative_offset_kept": sum(Q(r["output"]) == Q(r["incumbent"]) for r in negative),
        "n_observations": len(observations),
        "true_depth_covered": sum(observations.values()),
    }

def grouped(rows, keys):
    bins = defaultdict(list)
    for row in rows:
        bins[tuple(row[k] for k in keys)].append(row)
    return [{**dict(zip(keys, values)), **summarize(subset)} for values, subset in bins.items()]

def main():
    if load(RUN/"verification/VERIFY.json")["status"] != "PASS":
        raise RuntimeError("independent verification must PASS before analysis")
    rows = load(RUN/"evaluation/ROWS.json")
    if len(rows) != 5832 or len({(r["task_id"], r["method"]) for r in rows}) != 5832:
        raise ValueError("wrong complete fixed population")
    overall = grouped(rows, ("beta","method"))
    by_case = grouped(rows, ("beta","method","case"))
    by_seed = grouped(rows, ("beta","method","seed"))
    seed_stats = []
    for row in overall:
        seeds = [r for r in by_seed if r["beta"] == row["beta"] and r["method"] == row["method"]]
        seed_stats.append({"beta":row["beta"],"method":row["method"],"noise_seeds":len(seeds),
                          "mae_mean_model_mm":statistics.mean(r["mae_model_mm"] for r in seeds),
                          "mae_sd_across_seeds_model_mm":statistics.stdev(r["mae_model_mm"] for r in seeds),
                          "mse_sd_across_seeds_model_mm2":statistics.stdev(r["mse_model_mm2"] for r in seeds)})
    indexed = {(r["task_id"],r["method"]):r for r in rows}
    paired = []
    for beta in ("1/200","1/100","1/50"):
        tids = [r["task_id"] for r in rows if r["beta"] == beta and r["method"] == "keep" and Q(r["initial_ae_mm"]) > 0]
        for baseline, method in (("channel_wide","global_shared_wide"),
                                 ("shared_strict","global_shared_wide")):
            gains = [Q(indexed[t,baseline]["ae_mm"])-Q(indexed[t,method]["ae_mm"]) for t in tids]
            paired.append({"beta":beta,"baseline":baseline,"method":method,"n_offset":len(tids),
                           "method_better":sum(x>0 for x in gains),"same":sum(x==0 for x in gains),
                           "method_worse":sum(x<0 for x in gains),
                           "mae_gain_exact_model_mm":str(sum(gains,Q(0))/len(gains)),
                           "mae_gain_model_mm":float(sum(gains,Q(0))/len(gains))})
    legal_claim = []
    for beta in ("1/200","1/100","1/50"):
        g = next(r for r in overall if r["beta"] == beta and r["method"] == "global_shared_wide")
        c = next(r for r in overall if r["beta"] == beta and r["method"] == "channel_wide")
        safety = g["true_depth_covered"] == 54 and g["correct_kept"] == 54 and g["harm"] == 0
        legal_claim.append({"beta":beta,"safety_passed":safety,
                            "increment_passed":g["improved"]>c["improved"] and Q(g["mae_exact_model_mm"])<Q(c["mae_exact_model_mm"]),
                            "C1_this_beta":safety and g["improved"]>c["improved"] and Q(g["mae_exact_model_mm"])<Q(c["mae_exact_model_mm"])})
    DEST.mkdir(parents=True, exist_ok=False)
    for name, values in (("OVERALL",overall),("BY_CASE",by_case),("BY_SEED",by_seed),
                         ("SEED_STATS",seed_stats),("PAIRED",paired),("CLAIM_CHECKS",legal_claim)):
        save(DEST/(name+".json"),values)
        table(DEST/(name+".csv"),values)
    save(DEST/"ANALYSIS_SCOPE.json",{"source_run":str(RUN),"new_method_runs":0,
         "betas":["1/200","1/100","1/50"],"base_observations":54,"shifted_observations":162,
         "independent_real_scenes":0,"C1_supported":any(r["C1_this_beta"] for r in legal_claim),
         "same_source_note":"beta treatments and noise replicates are not independent scene samples",
         "reference_note":"raw historical images were not clipped; shift then clip is a common observation treatment"})
    print(json.dumps({"overall":overall,"paired":paired,"claims":legal_claim},ensure_ascii=False))

if __name__ == "__main__":
    main()

