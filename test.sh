#!/usr/bin/env bash
# EvSTVSR test entry point.
# Put the pretrained weights at ./saved_checkpoints/evstvsr.pth (see README).
# The same Adobe-trained model is used for both Adobe and GoPro.
# --save_vis additionally writes LR / output / GT / error-map images (and flow/event maps
# if vis_flow/vis_ev are enabled in the config). Results go to ./results/<exp_name>/.

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
