# unetTracker

This repository contains code to train a U-Net to track objects or body parts in images, videos, or camera feeds. 

For each image, the model outputs x and y coordinates together with a probability that the object is in the image.

Here is an animation to give you a sense of what unetTracker does. The unetTracker model for this animation was trained with 120 labeled images from a video.

![Example](documentation/images/tracking_animation.gif)


A series of Jupyter notebooks get you through the steps required to create your dataset of labeled images, train your neural network, assess how good it is, and improve it with problematic images.  

Once you have a model that performs as well as needed, you can use your trained model to process other videos.

The repository contains two main parts: 
1. The unetTracker python package. The code is located in the `unetTracker` folder.
2. A series of notebooks that shows you how to train, evaluate, and use your model. Use the notebooks in the folder `tracking_project_notebooks` for a step-by-step introduction on how to create your dataset, train the model on Google Colab, and label videos.

You can run the notebook using a Jupyter server running on a local computer. You will need access to a GPU to train your network and possibly do inference on additional videos. If you don't have access to a GPU on your local computer, you can [use Google Colab](documentation/colab.md) for the steps that require hardware acceleration. 

Under the hood, the project uses PyTorch as a deep learning framework.


* [Local installation](documentation/install.md)
* [Tracking project example](documentation/tracking_project_example.md)
* [Google Colab](documentation/colab.md)


## Model

I opted for a U-Net as a model architecture. This model was developed to perform image segmentation of biomedical images. The model outputs 2D arrays of the same height and width as the input images. I repurposed the model to output small blobs at the location of the tracked objects. 

https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

<img src="documentation/images/u-net-architecture.png" width="1000"/>

The model is defined in the `model.py` file.






