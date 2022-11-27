# unetTracker

Code to train and deploy a U-Net to track objects in images, videos, or camera feeds.

The U-Net does image segmentation, but it is used here to track the position of different object in space (e.g., body parts).

The user follows a series Jupyter notebooks which illustrate how to do the main steps involved in setting up a project, preparing a dataset, training a neural network and deploying the model.

Labeling of images to train the network can be done within a jupyter notebook. 

The project uses PyTorch as a deep learning framework. Training and inference will be much faster on GPU.


* [Installation](documentation/install.md)
* [Getting started](documentation/getting_started.md)

