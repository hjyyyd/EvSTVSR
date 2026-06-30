# EvSTVSR: Event Guided Space-Time Video Super-Resolution

Official implementation of our AAAI 2025 paper:

> **EvSTVSR: Event Guided Space-Time Video Super-Resolution**
> Haojie Yan, Zhan Lu, Zehao Chen, De Ma, Huajin Tang, Qian Zheng, Gang Pan

Given two low-resolution RGB frames and the events between them, EvSTVSR jointly performs
video frame interpolation and spatial super-resolution (here: ×8 in time, ×4 in space).

## Links

- [Paper(AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/32983)

## Abstract
In the domain of space-time video super-resolution, it is typically challenging to handle complex motions (including large and nonlinear motions) and varying illumination scenes due to the lack of inter-frame information. Leveraging the dense temporal information provided by event signals offers a promising solution. Traditional event-based methods typically rely on multiple images, using motion estimation and compensation, which can introduce errors. Accumulated errors from multiple frames often lead to artifacts and blurriness in the output. To mitigate these issues, we propose EvSTVSR, a method that uses fewer adjacent frames and integrates dense temporal information from events to guide alignment. Additionally, we introduce a coordinate-based feature fusion upsampling module to achieve spatial super-resolution. Experimental results demonstrate that our method not only outperforms existing RGB-based approaches but also excels in handling large motion scenarios.

## Citation

```bibtex
@inproceedings{yan2025evstvsr,
  title     = {EvSTVSR: Event Guided Space-Time Video Super-Resolution},
  author    = {Yan, Haojie and Lu, Zhan and Chen, Zehao and Ma, De and Tang, Huajin and Zheng, Qian and Pan, Gang},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025},
}
```

## Acknowledgments

This code builds on [VideoINR](https://github.com/Picsart-AI-Research/VideoINR-Continuous-Space-Time-Super-Resolution),
[Zooming Slow-Mo](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020) and
[CBMNet](https://github.com/intelpro/CBMNet). Thanks to the authors for releasing their code.
