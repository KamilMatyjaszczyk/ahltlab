#! /bin/bash

set -e

AHLT=../../..
PRE=../preprocessed
EXP=../experiment_results/all_gridsearch
MODELS="$EXP/models"
RESULTS="$EXP/results"

mkdir -p "$MODELS" "$RESULTS"

rm -f "$MODELS"/*.MEM "$MODELS"/*.SVM "$MODELS"/*.idx
rm -f "$RESULTS"/*.out "$RESULTS"/*.stats

for C in 0.1 1 10 100 1000; do
  echo "Training MEM model C=$C ..."
  py -3.12 train.py "$PRE/train.feat" "$MODELS/model-$C.MEM" C=$C
done

for C in 0.1 1 10 100 1000; do
  echo "Training SVM model C=$C ..."
  py -3.12 train.py "$PRE/train.feat" "$MODELS/model-$C.SVM" C=$C &
done
wait

for C in 0.1 1 10 100 1000; do
  echo "Running MEM model C=$C ..."
  py -3.12 predict.py \
    "$PRE/devel.feat" \
    "$MODELS/model-$C.MEM" \
    "$RESULTS/devel-MEM-$C.out"

  echo "Evaluating MEM results C=$C ..."
  py -3.12 "$AHLT/util/evaluator.py" DDI \
    "$AHLT/data/devel.xml" \
    "$RESULTS/devel-MEM-$C.out" \
    "$RESULTS/devel-MEM-$C.stats"

  echo "Running SVM model C=$C ..."
  py -3.12 predict.py \
    "$PRE/devel.feat" \
    "$MODELS/model-$C.SVM" \
    "$RESULTS/devel-SVM-$C.out"

  echo "Evaluating SVM results C=$C ..."
  py -3.12 "$AHLT/util/evaluator.py" DDI \
    "$AHLT/data/devel.xml" \
    "$RESULTS/devel-SVM-$C.out" \
    "$RESULTS/devel-SVM-$C.stats"
done

echo "Finished. Results stored in $RESULTS"