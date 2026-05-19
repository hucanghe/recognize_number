Recognize a handwritten digit from a jpg-photo using a trained MNIST model.

The neural network used has a model of:

Linear(28 * 28, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 10)

Tested with python 3.13.

10 test image files (*.jpg) are included.

Usage:
  py recog_num.py photo_8.jpg
  
  py recog_num.py photo_9.jpg --weights weights.pth
