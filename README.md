# unetTracker

This repository contains code to train a U-Net to track objects or body parts in images, videos, or camera feeds. 

For each image, the model outputs x and y coordinates together with a probability that the object is in the image.

Here is an animation to give you a sense of what unetTracker does. The unetTracker model for this animation was trained with 120 labeled images from a video.

![Example](documentation/images/tracking_animation.gif)


There is a series of Jupyter notebooks that get you through the steps required to create your dataset of labeled images, train your neural network, assess how good it is, and improve it with problematic images.  

Once you have a model that performs as well as needed, you can use your trained model to process other videos.

The repository contains two main parts: 
1. The unetTracker python package. The code is located in the `unetTracker` folder.
2. A series of notebooks that shows you how to train, evaluate, and use your model. The notebooks are in the `notebooks` folders.

You can run the notebook using a Jupyter server running on a local computer. You will need access to a GPU to train your network and possibly do inference on additional videos. If you don't have access to a GPU on your local computer, you can [use Google Colab](documentation/colab.md) for the steps that require hardware acceleration. 

You can perform live tracking using a webcam if you have a local GPU. But I recommend starting with videos instead of webcam. 

Under the hood, the project uses PyTorch as a deep learning framework.

We have developed this code on computers running Ubuntu 20.04.

* [Local installation](documentation/install.md)
* [unetTracker in a Docker container](documentation/docker.md)
* [Google Colab](documentation/colab.md)
* [Getting started](documentation/getting_started.md)





