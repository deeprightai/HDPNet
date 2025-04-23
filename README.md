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

## Model Weights

Download the pretrained model weights from the following link:
- [HDPNet Weights](https://drive.google.com/file/d/1-2X3Y4Z5A6B7C8D9E0F1G2H3I4J5K6L7/view?usp=sharing)

After downloading:
1. Create a `weights` directory in the project root:
```bash
mkdir weights
```

2. Place the downloaded weights file in the `weights` directory:
```bash
mv /path/to/downloaded/weights.pth weights/
```

## Usage

### Inference

To run inference on test images:

1. Place your test images in the `testdata` directory:
```bash
mkdir -p testdata
# Copy your images to testdata/
```

2. Run the inference script:
```bash
python inference.py
```

This will:
- Process all images in the `testdata` directory
- Save predictions in `results/predictions`
- Create side-by-side visualizations in `results/visualizations`
- Generate a video visualization in `results/visualization.mp4`

### Example Results

Here are some example results showing the original images and their predictions:

![Example 1](results/visualizations/camourflage_00122.jpg)
*Left: Original Image, Right: Prediction*

![Example 2](results/visualizations/camourflage_00364.jpg)
*Left: Original Image, Right: Prediction*

### Input/Output Format

- **Input**: RGB images (any size, will be resized to 384x384)
- **Output**: 
  - Binary mask predictions (0-255 grayscale)
  - Side-by-side visualizations
  - Video compilation of results

### Directory Structure

```
HDPNet/
├── weights/              # Model weights
├── testdata/            # Input test images
├── results/
│   ├── predictions/     # Binary mask predictions
│   ├── visualizations/  # Side-by-side visualizations
│   └── visualization.mp4  # Video compilation
├── inference.py         # Inference script
└── requirements.txt     # Dependencies
```

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
