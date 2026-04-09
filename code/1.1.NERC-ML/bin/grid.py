#!/usr/bin/env python3

import os
import re
import csv
import json
import shutil
import argparse
import itertools
from pathlib import Path

import paths
from extract_features import extract_features
from train import train
from predict import predict

import sys
sys.path.append(paths.UTIL)
from evaluator import evaluate


FULL_GRIDS = {
    "CRF": {
        "algorithm": ["lbfgs"],
        "feature.minfreq": [1, 2, 3],
        "max_iterations": [50, 100, 200],
        "c1": [0.01, 0.1, 0.5, 1.0],
        "c2": [0.01, 0.1, 0.5, 1.0],
        "epsilon": [1e-5, 1e-4],
    },
    "MEM": {
        "C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        "solver": ["lbfgs", "liblinear", "saga"],
        "max_iter": [500, 1000, 1500, 3000],
    },
    "SVM": {
        "C": [0.1, 1.0, 5.0, 10.0],
        "kernel": ["linear", "rbf", "poly"],
        "degree": [2, 3],
        "gamma": [0.001, 0.01, 0.1, 1.0],
    },
}

FAST_GRIDS = {
    "CRF": {
        "algorithm": ["lbfgs"],
        "feature.minfreq": [1, 2],
        "max_iterations": [100],
        "c1": [0.01, 0.1, 0.5],
        "c2": [0.01, 0.1, 0.5],
        "epsilon": [1e-5],
    },
    "MEM": {
        "C": [0.5, 1.0, 5.0],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [1000, 1500],
    },
    "SVM": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
        "degree": [2, 3],
        "gamma": [0.01, 0.1],
    },
}


def ensure_features_exist(do_extract: bool) -> None:
    os.makedirs(paths.PREPROCESS, exist_ok=True)

    train_feat = os.path.join(paths.PREPROCESS, "train.feat")
    devel_feat = os.path.join(paths.PREPROCESS, "devel.feat")

    if do_extract or not (os.path.exists(train_feat) and os.path.exists(devel_feat)):
        print("Extracting features for train...")
        extract_features(os.path.join(paths.DATA, "train.xml"), train_feat)

        print("Extracting features for devel...")
        extract_features(os.path.join(paths.DATA, "devel.xml"), devel_feat)


def grid_to_combinations(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for combination in itertools.product(*values):
        combos.append({k: str(v) for k, v in zip(keys, combination)})
    return combos


def maybe_filter_invalid_combos(model: str, combos: list[dict]) -> list[dict]:
    filtered = []
    for p in combos:
        if model == "SVM":
            try:
                float(p["gamma"])
            except Exception:
                continue
        filtered.append(p)
    return filtered


def sanitize_value(v: str) -> str:
    return str(v).replace("/", "_").replace(" ", "_").replace(".", "p")


def experiment_name(model: str, params: dict) -> str:
    parts = [model]
    for k in sorted(params.keys()):
        parts.append(f"{k}-{sanitize_value(params[k])}")
    return "__".join(parts)


def parse_stats_file(stats_file: str) -> tuple[float | None, float | None]:
    macro_f1 = None
    micro_f1 = None

    if not os.path.exists(stats_file):
        return macro_f1, micro_f1

    with open(stats_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()

            if s.startswith("M.avg"):
                percents = re.findall(r"(\d+(?:\.\d+)?)%", s)
                if percents:
                    macro_f1 = float(percents[-1])

            elif s.startswith("m.avg"):
                if s.startswith("m.avg(no class)"):
                    continue
                percents = re.findall(r"(\d+(?:\.\d+)?)%", s)
                if percents:
                    micro_f1 = float(percents[-1])

    return macro_f1, micro_f1


def safe_remove(path: str) -> None:
    if os.path.isfile(path):
        os.remove(path)


def safe_remove_model_files(model_file: str, model_type: str) -> None:
    safe_remove(model_file)
    if model_type in {"MEM", "SVM"}:
        safe_remove(model_file + ".idx")


def copy_best_model(temp_model_file: str, best_model_file: str, model_type: str) -> None:
    shutil.copy2(temp_model_file, best_model_file)
    if model_type in {"MEM", "SVM"}:
        shutil.copy2(temp_model_file + ".idx", best_model_file + ".idx")


def remove_previous_best(best_model_file: str, model_type: str) -> None:
    safe_remove(best_model_file)
    if model_type in {"MEM", "SVM"}:
        safe_remove(best_model_file + ".idx")


def run_one_experiment(model: str, params: dict, temp_dir: str) -> dict:
    exp_name = experiment_name(model, params)

    temp_model_file = os.path.join(temp_dir, f"{exp_name}.{model}")
    temp_pred_file = os.path.join(temp_dir, f"{exp_name}.out")
    temp_stats_file = os.path.join(temp_dir, f"{exp_name}.stats")

    train_feat = os.path.join(paths.PREPROCESS, "train.feat")
    devel_feat = os.path.join(paths.PREPROCESS, "devel.feat")
    devel_xml = os.path.join(paths.DATA, "devel.xml")

    print(f"\n=== Running {exp_name} ===")
    print(json.dumps(params, ensure_ascii=False))

    try:
        train(train_feat, params, temp_model_file)
        predict(devel_feat, temp_model_file, temp_pred_file)
        evaluate("NER", devel_xml, temp_pred_file, temp_stats_file)

        macro_f1, micro_f1 = parse_stats_file(temp_stats_file)

        return {
            "status": "ok",
            "experiment": exp_name,
            "params": params,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "temp_model_file": temp_model_file,
            "temp_pred_file": temp_pred_file,
            "temp_stats_file": temp_stats_file,
        }

    except Exception as e:
        return {
            "status": f"failed: {type(e).__name__}: {e}",
            "experiment": exp_name,
            "params": params,
            "macro_f1": None,
            "micro_f1": None,
            "temp_model_file": temp_model_file,
            "temp_pred_file": temp_pred_file,
            "temp_stats_file": temp_stats_file,
        }


def save_results_csv(results: list[dict], csv_file: str) -> None:
    rows = []
    for r in results:
        rows.append({
            "experiment": r["experiment"],
            "params": json.dumps(r["params"], ensure_ascii=False, sort_keys=True),
            "macro_f1": r["macro_f1"],
            "micro_f1": r["micro_f1"],
            "status": r["status"],
        })

    rows.sort(
        key=lambda x: (
            x["micro_f1"] is None,
            -(x["micro_f1"] or -1),
            -(x["macro_f1"] or -1),
        )
    )

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "params", "macro_f1", "micro_f1", "status"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Grid search that keeps only the best model")
    parser.add_argument("--model", required=True, choices=["CRF", "MEM", "SVM"])
    parser.add_argument("--grid", default="fast", choices=["fast", "full"])
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--outdir", default="gridsearch_best")
    parser.add_argument("--metric", default="micro", choices=["micro", "macro"])
    parser.add_argument("--keep-csv", action="store_true")
    args = parser.parse_args()

    ensure_features_exist(do_extract=args.extract)

    grid = FAST_GRIDS[args.model] if args.grid == "fast" else FULL_GRIDS[args.model]
    combos = maybe_filter_invalid_combos(args.model, grid_to_combinations(grid))

    out_root = os.path.join(args.outdir, args.model.lower())
    temp_dir = os.path.join(out_root, "temp")
    best_dir = os.path.join(out_root, "best")

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    best_model_file = os.path.join(best_dir, f"best_model.{args.model}")
    best_stats_file = os.path.join(best_dir, f"best_model.stats")
    best_pred_file = os.path.join(best_dir, f"best_model.out")
    summary_file = os.path.join(best_dir, f"best_{args.model.lower()}_config.json")
    csv_file = os.path.join(out_root, f"{args.model.lower()}_{args.grid}_results.csv")

    print(f"Model: {args.model}")
    print(f"Grid: {args.grid}")
    print(f"Metric: {args.metric}")
    print(f"Combinations: {len(combos)}")

    results = []
    best_result = None
    best_score = None

    for i, params in enumerate(combos, start=1):
        print(f"\n[{i}/{len(combos)}]")
        result = run_one_experiment(args.model, params, temp_dir)
        results.append(result)

        if result["status"] == "ok":
            score = result["micro_f1"] if args.metric == "micro" else result["macro_f1"]

            if score is not None and (best_score is None or score > best_score):
                print(f"New best {args.metric} F1: {score:.2f}")

                remove_previous_best(best_model_file, args.model)

                copy_best_model(result["temp_model_file"], best_model_file, args.model)
                shutil.copy2(result["temp_stats_file"], best_stats_file)
                shutil.copy2(result["temp_pred_file"], best_pred_file)

                best_score = score
                best_result = {
                    "model": args.model,
                    "grid": args.grid,
                    "selection_metric": args.metric,
                    "best_score": score,
                    "micro_f1": result["micro_f1"],
                    "macro_f1": result["macro_f1"],
                    "params": result["params"],
                    "experiment": result["experiment"],
                    "model_file": best_model_file,
                    "stats_file": best_stats_file,
                    "prediction_file": best_pred_file,
                }

        # remove temporary files for this run
        safe_remove_model_files(result["temp_model_file"], args.model)
        safe_remove(result["temp_pred_file"])
        safe_remove(result["temp_stats_file"])

    if args.keep_csv:
        save_results_csv(results, csv_file)

    if best_result is not None:
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(best_result, f, indent=2, ensure_ascii=False)

        print("\n=== BEST CONFIGURATION ===")
        print(json.dumps(best_result, indent=2, ensure_ascii=False))
    else:
        print("\nNo successful runs.")

    # remove temp directory if empty
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass


if __name__ == "__main__":
    main()