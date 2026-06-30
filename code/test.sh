#!/usr/bin/env bash
# EvSTVSR test entry point.

# ---- Adobe240 (x4 space, x8 time) ----
CUDA_VISIBLE_DEVICES=0 python test.py \
  --yml_file configs/test_adobe.yml \
  --exp_name evstvsr \
  --save_vis

# ---- GoPro (x4 space, x8 time) ----
# CUDA_VISIBLE_DEVICES=0 python test.py \
#   --yml_file configs/test_gopro.yml \
#   --exp_name evstvsr \
#   --save_vis
