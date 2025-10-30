# [InstanceSegFrac-TMI-2025]
Welcome to the official repository for the InstanceSegFrac.

**Code and model will be released on acceptance of the paper!**

## News & Updates

## Models Available

## Getting Started
To get started, follow these steps:

1. Clone the Repository
   ```bash
   git clone 
   ```
2. Create and Activate a Virtual Environment
    ```bash
    conda create -n InstanceSegFrac python=3.12
    conda activate InstanceSegFrac
   ```
3. Install Pytorch: Follow the instructions [here](https://pytorch.org/get-started/locally/):
   ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
5. Install the Repository 
   ```bash 
    cd InstanceSegFrac
    pip install -e .
   ```

## Data Preparation
We follow the [nnU-Net V2](https://github.com/MIC-DKFZ/nnUNet?tab=readme-ov-file) guideline for data preparation, detailed below and accessible [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).


## Citation
If you utilize the baselines in this repository for your research, please consider citing the relevant papers.


## Acknowledgements

We would like to acknowledge the contributions of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and the authors of the baseline models: [LightM-UNet](https://github.com/mrblankness/lightm-unet), [MedNeXT](https://github.com/MIC-DKFZ/MedNeXt), and [SAMed](https://github.com/hitachinsk/SAMed). This repository builds upon their foundational code and work.

