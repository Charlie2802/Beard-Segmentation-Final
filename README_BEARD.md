# Beard Segmentation

## Overview

Beard Segmentation is a project focused on developing and training segmentation models specifically for detecting and segmenting beards in images. This repository provides a comprehensive framework for preprocessing data, training models, and evaluating their performance. The project leverages the power of tensorflow-gpu to facilitate efficient model training and inference.

## Features

- Data preprocessing and albumentations
- Training and evaluation scripts
- Support for multiple segmentation models
- Customizable training configurations
- Future Work

## Installation

### Prerequisites

- Python >= 3.8
- CUDA == 12.2

### Setup
### Clone the Repository

```bash
git clone https://github.com/Charlie2802/Beard-Segmentation
cd beard-segmentation
```

- To install the required dependencies, navigate to your project directory and run:
```bash
pip install -r requirements.txt

```


## Training Data



The `TRAIN_DATASETS` folder includes the following folders:

- `train_4_graycrop/`: Images and their corresponding masks are cropped using `pre_process_graycrop.py` in the `PRE-PROCESS` folder, which utilizes `yolov8n-fact.pt` (a face detection model) to remove irrelevant background information. It also converts the images into grayscale.
- `train_5_Color/`: Similar to `train_4_graycrop`, except that images are in color.
- `train_6_graymask/`: Uses `graymasks.py` in the `PRE-PROCESS` folder, which applies a Sobel filter to find edges and performs preprocessing operations to generate grayscale masks, aiming for precise details in beard segmentation.
- `train_7_trimap/`: Uses `trimap.py` in the `PRE-PROCESS` folder to create trimaps of the masks, which highlight the border of the mask, indicating areas where the model needs to be more certain about the presence of a beard.
- `train_8_harshir/`: Utilizes `graymask_harshsir` in the `PRE-PROCESS` folder, which calculates the first non-zero element column-wise from both top and bottom in the Sobel image and makes everything white in between. This aims to find more granular and precise hair details for better masks.
- `train_10/`: Combines the Sobel filter and trimap techniques to enhance fine details using image matting loss, representing future work for improved beard segmentation.

## Models 


The `MODELS_ARCHITECTURE` folder includes the following folders:

- `model_1_unet`: A basic U-Net model (`model_1_unet.py`) designed for general beard segmentation tasks.

- `model_2_aspp`: Incorporates Atrous Spatial Pyramid Pooling (ASPP) (`model_2_ASPP.py`), which uses dilated convolutions at multiple scales to capture contextual information effectively.

- `model_3_latestmodel`: Utilizes depthwise separable convolutions, attention mechanisms (`model_3_latestmodel.py`), and pyramid pooling to enhance feature extraction and spatial context, crucial for accurate image segmentation.

- `model_4_avinash`: Based on depthwise separable convolutions and upsampling layers, this model focuses on feature extraction and spatial context capture for precise image segmentation.

- `model_5_google`: Integrates separable convolutions, bottleneck blocks, ASPP modules, SE blocks, and dense blocks to capture detailed features and spatial contexts. Designed for high-resolution inputs (224x224x3).

- `model_6_matting`: A refinement model that refines segmentation outputs using concatenated input and encoder-decoder model outputs with convolutional layers in Keras. This model architecture aims to improve segmentation accuracy by focusing on finer details. This represents ongoing work to enhance segmentation quality.

## Loss Functions

The `TRAINING_FILES` folder includes the script `losses.py`:

### Overview

This scipt explores different loss functions for the task of beard segmentation in images. The goal was to address challenges such as class imbalance(images of people with no beard), boundary segmentation ( finer segmentation) , and detection in diverse skin tones.

### Loss Functions Used

- **Binary Cross-Entropy (BCE) Loss**: Standard loss function for binary classification tasks.
- **Dice Loss**: Effective for handling class imbalance and small objects.
- **Lovasz Hinge Loss**: Optimizes IoU-based metrics, improving segmentation accuracy.
- **Boundary Loss**: Focuses on predicting sharp boundaries between objects.
- **Matting Loss**: Utilizes Sobel filters for capturing fine details and minimizing boundary errors.


### Results

The combination of Matting Loss and BCE Loss with weights [0.3, 0.7] respectively yielded the best results. Matting Loss, leveraging Sobel filters, significantly enhanced the segmentation quality by capturing intricate beard features and improving boundary predictions. This approach effectively addressed challenges in detecting beards across diverse demographics, including images of women and individuals with darker skin tones.

### Future Work

Future work will focus on integrating Matting Blocks with Matting Loss to further refine segmentation outputs and explore additional enhancements for better performance.



## Results

The `RESULTS` folder includes the following RESULTS

- `results_rewat`: This was the previously trained model, which served as the benchmark. The newer model should perform better and be smaller than this.

- `results_model_gray_224_gamma_corrected`: These results are for the basic U-Net model. The images were preprocessed by adaptive gamma correction (this code is present in `test.py` in the TRAINING FILES folder). The main reason for this preprocessing was to eliminate the bias of dark skin and shadow, which was getting detected as beard.

- `results_model_color_aspp`: These results were obtained with the `aspp.py` model.

- `results_model_color_224_google_inv_block`: This set of results was produced using the Google model (`google.py`) without any different loss functions applied. This performed better than all the previosuly used models. 

- `results_model_color_224_google_matting_bce`: This set of results is from the Google model (`google.py`) on color + crop images. This was the best result achieved, but it still detects beards on females. If implemented with dice loss along with the above-mentioned two losses, this problem will likely be resolved.


## Future Work

Future work will focus on integrating Matting Blocks with Matting Loss and creating a refinement model (`image_matting_model.py`) with a basic encoder-decoder model. This model will also use trimaps on edges to further segment images, refining the segmentation outputs. Additionally, a combination of dice loss (for non-bearded individuals) with BCE and Matting Loss, with suitable weights, will be explored.

Other potential avenues include examining hair segmentation GitHub repositories and their datasets to understand how they have carefully curated their datasets. These insights can be used to explore additional enhancements for better performance.


