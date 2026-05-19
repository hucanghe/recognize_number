# Handwritten Digit Recognition

Recognize a handwritten digit from a jpg-photo using a trained MNIST model.

## Neural Network Architecture

The neural network used has the following model structure:
Linear(28 * 28, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 10)


## Requirements

- Tested with Python 3.13

## Included Files

- 10 test image files (*.jpg) are included

## Usage

### Basic usage:
```bash
py recog_num.py photo_8.jpg
py recog_num.py photo_9.jpg --weights weights.pth
