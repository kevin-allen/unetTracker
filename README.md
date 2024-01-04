# unetTracker

This repository contains code to train a U-Net to track objects or body parts in images, videos, or camera feeds. 

For each image, the model outputs x and y coordinates together with a probability that the object is in the image.

Here is an animation to give you a sense of what unetTracker does. The unetTracker model for this animation was trained with 120 labeled images from a video.

![Example](documentation/images/tracking_animation.gif)

A series of Jupyter Notebooks get you through the steps required to create your dataset of labeled images, train your neural network, assess how good it is, and improve it with problematic images.  

Once you have a model that performs as well as needed, you can use your trained model to process other videos.

The repository contains two main parts: 
1. The unetTracker python package. The code is located in the `unetTracker` folder.
2. A series of Jupyter Notebooks that shows you how to train, evaluate, and use your model. These Notebooks are located in the folders with `notebooks` in their names.

## Required hardware
  
You will need to have access to a GPU/TPU to train your network and make inferences. 

If you don't have a computer with a GPU, you use Google Colab to work with unetTracker. Use the Notebooks located in the folder `tracking_project_notebooks_colab` for a step-by-step introduction to creating your project and datasets, training the model, and labeling videos. See the instructions on how to [use Google Colab](documentation/colab.md).
  
More information on getting going with your project can be found [here](https://github.com/kevin-allen/unetTracker/blob/main/documentation/tracking_project_example.md).

If you have a computer with a GPU, you can run the Notebook using a Jupyter server running on a local computer. You can use the Notebooks of the `notebooks` directory.

Under the hood, the project uses PyTorch as a deep learning framework.

## Useful information

* [Local installation](documentation/install.md)
* [Tracking project example](documentation/tracking_project_example.md)
* [Google Colab](documentation/colab.md)
* [Speed up inference time](documentation/speed_up.md)


## Model

I opted for a U-Net as a model architecture. This model was developed to perform image segmentation of biomedical images. The model outputs 2D arrays of the same height and width as the input images. I repurposed the model to output small blobs at the location of the tracked objects. 

https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

<img src="documentation/images/u-net-architecture.png" width="1000"/>

The model is defined in the `model.py` file.

