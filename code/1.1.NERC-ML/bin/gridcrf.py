#!/usr/bin/env python3

import itertools
import os
import shutil
import sys

from train import train
from predict import predict

import paths
sys.path.append(paths.UTIL)
from evaluator import evaluate


def parse_stats_f1(stats_file):
    with open(stats_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("m.avg "):
                parts = line.strip().split()
                return float(parts[-1].replace("%", ""))
    raise ValueError(f"Could not find m.avg line in {stats_file}")


def print_stats(stats_file):
    with open(stats_file, "r", encoding="utf-8") as f:
        print(f.read())


def cleanup_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def build_param_combinations():
    combinations = []

    minfreq_values = [1, 2]
    max_iterations_values = [50, 100]
    c1_values = [0.0, 0.1]
    c2_values = [0.1, 1.0]
    epsilon_values = [1e-5]

    for minfreq, maxit, c1, c2, eps in itertools.product(
        minfreq_values,
        max_iterations_values,
        c1_values,
        c2_values,
        epsilon_values
    ):
        combinations.append({
            "algorithm": "lbfgs",
            "feature.minfreq": str(minfreq),
            "max_iterations": str(maxit),
            "c1": str(c1),
            "c2": str(c2),
            "epsilon": str(eps),
        })

    return combinations


def main():
    train_feat = os.path.join(paths.PREPROCESS, "train.feat")
    devel_feat = os.path.join(paths.PREPROCESS, "devel.feat")
    devel_xml = os.path.join(paths.DATA, "devel.xml")

    os.makedirs(paths.MODELS, exist_ok=True)
    os.makedirs(paths.RESULTS, exist_ok=True)

    best_f1 = -1.0
    best_params = None

    best_model_path = os.path.join(paths.MODELS, "best-model.CRF")
    best_out_path = os.path.join(paths.RESULTS, "best-devel-CRF.out")
    best_stats_path = os.path.join(paths.RESULTS, "best-devel-CRF.stats")

    temp_model_path = os.path.join(paths.MODELS, "temp-model.CRF")
    temp_out_path = os.path.join(paths.RESULTS, "temp-devel-CRF.out")
    temp_stats_path = os.path.join(paths.RESULTS, "temp-devel-CRF.stats")

    combinations = build_param_combinations()
    print(f"Testing {len(combinations)} CRF hyperparameter combinations...\n")

    for i, params in enumerate(combinations, start=1):
        print("=" * 70)
        print(f"[{i}/{len(combinations)}]")
        print(f"Training CRF with params: {params}")

        cleanup_if_exists(temp_model_path)
        cleanup_if_exists(temp_out_path)
        cleanup_if_exists(temp_stats_path)

        try:
            train(train_feat, params, temp_model_path)
            predict(devel_feat, temp_model_path, temp_out_path)
            evaluate("NER", devel_xml, temp_out_path, temp_stats_path)

            f1 = parse_stats_f1(temp_stats_path)
            print(f"m.avg F1 = {f1:.2f}")

            if f1 > best_f1:
                best_f1 = f1
                best_params = params.copy()

                shutil.copyfile(temp_model_path, best_model_path)
                shutil.copyfile(temp_out_path, best_out_path)
                shutil.copyfile(temp_stats_path, best_stats_path)

                print("New best model found and saved.")

        except Exception as e:
            print(f"Failed for params: {params}")
            print(f"Error: {e}")

    print("\n" + "=" * 70)
    print("GRID SEARCH FINISHED")
    print("=" * 70)

    if best_params is None:
        print("No valid model was trained.")
        sys.exit(1)

    print("Best parameters:")
    print(best_params)
    print(f"Best m.avg F1: {best_f1:.2f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Best stats saved to: {best_stats_path}")

    print("\nBest .stats:\n")
    print_stats(best_stats_path)


if __name__ == "__main__":
    main()