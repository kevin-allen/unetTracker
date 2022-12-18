# unetTracker

Code to train and deploy a U-Net to track objects in images, videos, or camera feeds. For each image, we get a coordinate for each tracked object if they are visible in the image.

There is a series Jupyter notebooks that get you through the steps required to train your network, assess how good it is, improve it, and process videos.

You can work from saved videos or from a camera.

The project uses PyTorch as a deep learning framework. You will need access to a GPU to train your model.

We have develop this code on computers running Ubuntu 20.04.

* [Local installation](documentation/install.md)
* [unetTracker in a Docker container](documentation/docker.md)
* [Google colab](documentation/colab.md)
* [Getting started](documentation/getting_started.md)

