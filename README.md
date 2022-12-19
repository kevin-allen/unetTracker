# unetTracker

Code to train a U-Net to track objects in images, videos, or camera feeds. For each image, we get a x,y coordinate for each tracked object together with a probability that the object is in the image.

There is a series Jupyter notebooks that get you through the steps required to train your network, assess how good it is, improve it, process videos and export it.

You can run a jupyter server on a computer with a GPU and go through the jupyter notebooks from a remote computer.

You can work from saved videos or from a camera.

The project uses PyTorch as a deep learning framework. You will need access to a GPU to train your model.

We have develop this code on computers running Ubuntu 20.04.

* [Local installation](documentation/install.md)
* [unetTracker in a Docker container](documentation/docker.md)
* [Google colab](documentation/colab.md)
* [Getting started](documentation/getting_started.md)

