#!/bin/bash
for tt in 1 2 4 8 16 32
do
    export OMP_NUM_THREADS=$tt
    echo ""
    echo ""
    echo "number of threads $tt"
    _build/opt/deeplearning/torch/th.llar deeplearning/torch/layers/test/test_HSM_speed.lua
done
