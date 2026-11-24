"""Compare completed training exports using validation returns and publish one model."""
import argparse
import csv
import json
import shutil
from pathlib import Path

from experiment import ROOT, digest, save_json


def select(runs, out):
    selected = []
    for profile, run in runs.items():
        data = run/"export/results"
        summary = json.loads((data/"training_summary.json").read_text())
        model_file = run/"export/models/best_model.zip"
        assert digest(model_file) == summary["model_sha256"]
        with (data/"training_progress.csv").open() as f:
            history = list(csv.DictReader(f))
        best = max(history, key=lambda r: float(r["validation_mean"]))
        assert int(best["step"]) == summary["selected_step"]
        assert float(best["validation_mean"]) == summary["selected_validation_mean"]
        selected.append(dict(profile=profile, steps=summary["steps"], updates=summary["updates"],
                             training_seconds=summary["seconds_learning"],
                             step_cost=1 if profile=="step-cost" else 0,
                             train_freq=summary["config"]["train_freq"],
                             best_step=int(best["step"]), validation_mean=float(best["validation_mean"]),
                             best_landed=int(best["validation_landings"]),
                             maximum_landed=max(int(r["validation_landings"]) for r in history),
                             model_sha256=summary["model_sha256"]))
    winner = max(selected, key=lambda r:r["validation_mean"])
    results = out/"results"
    models = out/"models"
    models.mkdir(parents=True,exist_ok=True)
    folder_names = {"sparse":"pilot","standard":"standard","step-cost":"step_cost"}
    for entry in selected:
        profile = entry["profile"]
        source = runs[profile]/"export"
        folder = "." if entry is winner else folder_names[profile]
        dst = results/folder
        dst.mkdir(parents=True,exist_ok=True)
        for filename in ["training_summary.json","training_episodes.csv","training_progress.csv","validation_episodes.csv"]:
            shutil.copyfile(source/"results"/filename, dst/filename)
        entry["logs"] = folder
        if entry is winner:
            shutil.copyfile(source/"models/best_model.zip",models/"best_model.zip")
    record = dict(selection="Highest mean original return on 10 fixed validation seeds across all runs and checkpoints. Earlier step wins within-run ties. Cross-run ties follow CLI profile order: sparse, standard, step-cost.",
                  selected_profile=winner["profile"], total_training_steps=sum(r["steps"] for r in selected),
                  runs=selected)
    save_json(results/"experiments.json",record)
    print(f"Selected {winner['profile']} at {winner['best_step']:,} steps; validation mean {winner['validation_mean']:.2f}.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot",type=Path,default=ROOT/"runs/pilot")
    p.add_argument("--standard",type=Path,default=ROOT/"runs/reproduction")
    p.add_argument("--step-cost",type=Path,default=ROOT/"runs/step_cost")
    p.add_argument("--out",type=Path,default=ROOT/"runs/selected")
    a=p.parse_args()
    select({"sparse":a.pilot,"standard":a.standard,"step-cost":a.step_cost},a.out)
