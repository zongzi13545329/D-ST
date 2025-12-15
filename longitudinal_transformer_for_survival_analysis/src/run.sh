#!/bin/bash

# 不需要 conda activate 也不需要 cd

COMMON_ARGS="--dropout 0.25 --augment --reduce_lr --batch_size 8 --tpe --step_ahead"

# # OHTS baseline
# python train.py --results_dir results --dataset OHTS --model image --dropout 0.25 --augment --reduce_lr --batch_size 64

# # OHTS LSTA
# python train.py --results_dir results --dataset OHTS --model LTSA $COMMON_ARGS

# OHTS SF deformable variations
# for spatial in false true; do
#     for temporal in true false; do
#         args=""
#         if [ "$spatial" == "true" ]; then
#             args="$args --use_deformable_spatial"
#         fi
#         if [ "$temporal" == "true" ]; then
#             args="$args --use_deformable_temporal"
#         fi

#         echo "Running: python train.py --results_dir results --dataset OHTS --model SF $COMMON_ARGS $args"
#         python train.py --results_dir results --dataset OHTS --model SF $COMMON_ARGS $args
#     done
# done

# # AREDS baseline
# python train.py --results_dir results --dataset AREDS --model image --dropout 0.25 --augment --reduce_lr --batch_size 32

# # AREDS LSTA
# python train.py --results_dir results --dataset AREDS --model LTSA $COMMON_ARGS

# AREDS SF deformable variations
for spatial in false true; do
    for temporal in false true; do
        args=""
        if [ "$spatial" == "true" ]; then
            args="$args --use_deformable_spatial"
        fi
        if [ "$temporal" == "true" ]; then
            args="$args --use_deformable_temporal"
        fi

        echo "Running: python train.py --results_dir results --dataset AREDS --model SF $COMMON_ARGS $args"
        python train.py --results_dir results --dataset AREDS --model SF $COMMON_ARGS $args
    done
done