# EvSTVSR: Event Guided Space-Time Video Super-Resolution

## Installation

Tested with Python 3.7 and PyTorch 1.13 (CUDA 11.7) on NVIDIA GPUs.

```bash
conda create -n evstvsr python=3.7 -y
conda activate evstvsr

# install PyTorch (matched to your CUDA), e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt

# build the correlation (cost-volume) CUDA extension used by the flow network
cd correlation_package
python setup.py build develop      # needs nvcc; see "nvcc setting.md"
cd ..
```

## Pretrained weights

Following the paper, a single model is trained on Adobe240 and evaluated on **both**
Adobe240 and GoPro. Download the checkpoint and put it at `./saved_checkpoints/evstvsr.pth`:

**Download:** [Google Drive](https://drive.google.com/file/d/1gKDC3ZA8tdcoC1giBSL04WdZ5UDDlleh/view?usp=sharing)

(Use `--checkpoint PATH` to point to a different weights file.)

## Data

Set the dataset paths (`dataroot_GT` / `dataroot_LQ`) in `configs/test_adobe.yml` and
`configs/test_gopro.yml`. The scenes to evaluate are listed in
`dataset/adobe_folder_test.txt` and `dataset/gopro_folder_test.txt`.

Expected layout (one folder per scene, plus precomputed event voxel grids):

```
data/
├── Adobe240/frame/test/<scene>/...        # high-fps RGB frames (.png)
├── Adobe240/frame/test_volt/<scene>/...    # event voxel grids (.npz)
├── GoPro/GOPRO_Large_all/test/<scene>/...
└── GoPro/GOPRO_Large_all/test_volt/<scene>/...
```

Following the paper, LR frames are bilinearly down-sampled (×4) and events are simulated
with [vid2e](https://github.com/uzh-rpg/rpg_vid2e) and converted to voxel grids (16 bins).
See `dataset/adobe_dataset_ev_from_raw.py` for the exact file naming.

## Testing

```bash
# Adobe240
python test.py --yml_file configs/test_adobe.yml --save_vis

# GoPro
python test.py --yml_file configs/test_gopro.yml --save_vis
```

`test.sh` contains the same commands. Useful flags:

- `--checkpoint PATH` — use a specific weights file (default `./saved_checkpoints/evstvsr.pth`).
- `--save_vis` — also save the LR / prediction / GT / error-map images (and flow/event maps
  if `vis_flow` / `vis_ev` are enabled in the config).
- `--exp_name NAME` — name of the output subfolder.

PSNR/SSIM are printed to the log, and visual results are written to
`./results/<exp_name>/test_<dataset>/`.
