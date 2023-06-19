# unetTracker

Code to train a U-Net to track objects in images, videos, or camera feeds. 

For each image, we get a x,y coordinate for each tracked object together with a probability that the object is in the image.

There is a series Jupyter notebooks that get you through the steps required to train your neural network, assess how good it is, improve it with difficult images.  You can then use your trained network to process other video.

The repository contains two main parts: 
1. The unetTracker python package. The code is located in the `unetTracker` folder.
2. A series of notebooks that shows you how to train, evaluate and use your model. The notebooks are in the `notebooks` folders.

You can run the notebook using a Jupyter server. The server will need to have access to a GPU. 

You can train your model using a live webcam as input or using saved video files. 

Under the hood, the project uses PyTorch as a deep learning framework.

We have develop this code on computers running Ubuntu 20.04.

* [Local installation](documentation/install.md)
* [unetTracker in a Docker container](documentation/docker.md)
* [Google colab](documentation/colab.md)
* [Getting started](documentation/getting_started.md)

