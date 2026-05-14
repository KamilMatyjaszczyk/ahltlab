#! /bin/bash

AHLT=../../..

rm -rf *.out *.stats *.MEM *.SVM *.idx

for C in 0.1 1 10 100 1000; do
  echo "Training MEM model $C ..."
  python3 train.py train.feat model-$C.MEM C=$C
done

for C in 0.1 1 10 100 1000; do
  echo "Training SVM model $C ..."
  python3 train.py train.feat model-$C.SVM C=$C &
done
wait

for C in 0.1 1 10 100 1000; do
  echo "Running MEM model $C ..."
  python3 predict.py devel.feat model-$C.MEM devel-MEM-$C.out

  echo "Evaluating MEM results $C ..."
  python3 $AHLT/util/evaluator.py DDI $AHLT/data/devel.xml devel-MEM-$C.out devel-MEM-$C.stats

  echo "Running SVM model $C ..."
  python3 predict.py devel.feat model-$C.SVM devel-SVM-$C.out

  echo "Evaluating SVM results $C..."
  python3 $AHLT/util/evaluator.py DDI $AHLT/data/devel.xml devel-SVM-$C.out devel-SVM-$C.stats
done
