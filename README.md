# vit-medical README.md

# ViT Medical

ViT Medical is a project that implements a Vision Transformer (ViT) model for traditional Chinese medicine classification. This repository provides tools for training and inference using the ViT architecture, leveraging the Transformers library.

## Features

- Implementation of the Vision Transformer model tailored for medical datasets.
- Pre-training and inference scripts for easy model training and evaluation.
- Configurable training parameters through YAML configuration files.
- Utilities for dataset loading and preprocessing.

## Project Structure

```
vit-medical
├── src
│   ├── vit_medical
│   │   ├── __init__.py
│   │   ├── model
│   │   │   ├── __init__.py
│   │   │   ├── modeling_vit.py
│   │   │   └── vit_modeling_config.py
│   │   ├── data
│   │   │   ├── __init__.py
│   │   │   └── dataset_utils.py
│   │   └── utils
│   │       ├── __init__.py
│   │       └── training_utils.py
├── scripts
│   ├── vit_pt.py
│   └── inference.py
├── configs
│   └── training_config.yaml
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

### Pre-training the ViT Model

To pre-train the ViT model, use the following command:

```
python scripts/vit_pt.py --config configs/training_config.yaml
```
### dataset_download
```
from datasets import load_dataset
Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("wubingheng/vit-medical-classifaction2.0")
```
### Train

To train model, run:

```
accelerate launch --config_file accelerate_configs/single_gpu.yaml ./scripts/vit_pt.py --config configs/training_config.yaml

```

## Configuration

Training parameters can be adjusted in the `configs/training_config.yaml` file. This includes settings such as learning rate, batch size, and number of training epochs.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## train result
![W B Chart 2025_6_4 14_00_12](https://github.com/user-attachments/assets/bb3b5d8c-483f-4c84-839a-4de1d0a1316c)
![W B Chart 2025_6_4 14_00_36](https://github.com/user-attachments/assets/deffee00-757d-4ab4-a4c3-91dd3b3b1953)
![W B Chart 2025_6_4 14_00_45](https://github.com/user-attachments/assets/4cdb1970-a900-4d4e-bb35-856dbfeb14a7)


