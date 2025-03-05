# vit-medical README.md

# ViT Medical

ViT Medical is a project that implements a Vision Transformer (ViT) model for medical image analysis. This repository provides tools for training and inference using the ViT architecture, leveraging the Transformers library.

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
## datase
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("wubingheng/vit-medical-classifaction2.0")



### Inference

To perform inference with a trained model, run:

```
python scripts/inference.py --model_path <path_to_trained_model>
```

## Configuration

Training parameters can be adjusted in the `configs/training_config.yaml` file. This includes settings such as learning rate, batch size, and number of training epochs.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

