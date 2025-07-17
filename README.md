# NEURAL STYLE TRANSFER

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: ABHISHEK SINGH RAJPUT

*INTERN ID*: CT04DG3397

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH


# Neural Style Transfer

## Overview

Neural Style Transfer is a fascinating technique in deep learning that blends the content of one image with the artistic style of another. This project implements a neural style transfer model in Python using PyTorch and torchvision libraries. The script takes a **content image** (e.g., a photograph) and a **style image** (e.g., a famous painting) and generates a new image that preserves the original content but is rendered in the artistic style of the style image.

This approach leverages convolutional neural networks (CNNs), specifically a pretrained VGG19 model, to extract feature representations of content and style, and then iteratively updates a generated image to minimize the difference from these features, resulting in visually compelling stylized images.

## Features

* Utilizes PyTorch and torchvision for deep learning and image processing.
* Uses the pretrained VGG19 network to extract style and content features.
* Supports GPU acceleration if available (automatically uses CPU otherwise).
* Customizable number of optimization steps for better stylization quality.
* Saves the output image automatically after processing.
* Provides informative loss updates during optimization.


## Installation & Setup

### Prerequisites

Before running the script, ensure you have the following installed:

* Python 3.7 or higher
* PyTorch (compatible with your Python version and CUDA if using GPU)
* torchvision
* Pillow (PIL)
* numpy

You can install the necessary packages via pip:

```bash
pip install torch torchvision pillow numpy
```

Make sure you have a stable internet connection for the first run, as the VGG19 pretrained model weights will be downloaded automatically.

## Usage

1. **Prepare Images:**
   Place your content image and style image files in the project directory or specify their full paths in the script.

2. **Run the Script:**
   Execute the Python script:

```bash
python style_transfer.py
```

3. **Process:**
   The script will load the images, initialize the neural style transfer process, and run iterative optimization (default 300 steps) to blend style and content.

4. **Output:**
   After completion, the stylized image will be saved as `output.jpg` in the current directory. You can open this file to see the result.


## How It Works

* **Content Extraction:**
  The pretrained VGG19 network is used to extract high-level content features from the content image, particularly from intermediate convolutional layers.

* **Style Extraction:**
  The style image’s textures and patterns are encoded through gram matrices computed from various convolutional layers of VGG19, capturing the style statistics.

* **Optimization:**
  A copy of the content image is treated as a trainable tensor, and gradient descent (LBFGS optimizer) is applied to minimize a loss function that combines both content and style loss components.

* **Loss Components:**

  * *Content Loss:* Measures the difference between the generated image and content image features.
  * *Style Loss:* Measures the difference between the gram matrices of the style image and the generated image.

By minimizing these losses, the output image gradually acquires the artistic style while maintaining the content structure.


## Important Notes & Tips

* **Image Sizes:**
  Larger images provide better detail but require more GPU memory and processing time. Typical sizes are 400x400 or 512x512 pixels.

* **Performance:**
  Using a GPU dramatically speeds up processing. If running on CPU, the process will be slower but still functional.

* **Parameter Tweaking:**
  You can change the number of optimization steps, content/style weights, or image sizes in the script to experiment with different results.

* **Input Format:**
  Ensure the input images are in a readable format (JPEG, PNG, etc.) and accessible at the specified path.

* **Dependencies:**
  Keep your PyTorch and torchvision versions updated to avoid compatibility warnings or errors.


## Common Issues

* **FileNotFoundError:**
  Make sure your content and style image paths are correct.

* **Runtime Errors related to backward():**
  These may arise from PyTorch’s autograd system if the computation graph is modified in place. The provided script is designed to avoid such issues.

* **Warnings about deprecated parameters:**
  These come from torchvision model loading. Using updated weights arguments is recommended.

## Future Enhancements

* Support for multiple style images (style blending).
* Real-time style transfer using optimized models.
* Adding a GUI interface or web app for easier use.
* Batch processing of multiple images.
* Video style transfer capabilities.

#STYLE

![Image](https://github.com/user-attachments/assets/1426ce90-0401-4784-94ca-e6954da9a766)
