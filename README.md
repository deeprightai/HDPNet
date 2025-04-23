# HDPNet: Hourglass Vision Transformer with Dual-Path Feature Pyramid for Camouflaged Object Detection

This repository contains the implementation of HDPNet, a state-of-the-art model for camouflaged object detection. The model uses a combination of Hourglass Vision Transformer and Dual-Path Feature Pyramid architecture to achieve high accuracy in detecting camouflaged objects.

## Features

- Hourglass Vision Transformer architecture
- Dual-Path Feature Pyramid for multi-scale feature extraction
- High accuracy on various camouflaged object detection datasets
- Easy-to-use inference script with visualization capabilities

## Installation

1. Create a conda environment:
```bash
conda create -n HDPNet python=3.8
conda activate HDPNet
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Inference

To run inference on test images:

```bash
python inference.py
```

This will:
- Process all images in the `testdata` directory
- Save predictions in `results/predictions`
- Create side-by-side visualizations in `results/visualizations`
- Generate a video visualization in `results/visualization.mp4`

### Model Weights

The pretrained model weights are available in the `weights` directory. Make sure to download them before running inference.

## Results

The model achieves state-of-the-art performance on various camouflaged object detection datasets. For detailed quantitative results, please refer to the paper.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{he2025hdpnet,
  title={HDPNet: Hourglass Vision Transformer with Dual-Path Feature Pyramid for Camouflaged Object Detection},
  author={He, Jinpeng and Liu, Biyuan and Chen, Huaixin},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={8638--8647},
  year={2025},
  organization={IEEE}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
