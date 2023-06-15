# Getting started with unetTracker

Once the unetTracker python package is installed in your python environment, all you need to do is to follow a series of Jupyter notebooks located in the `notebooks` directory of the unetTracker repository. 

I would recommend to create a directory for your project that will include the model, dataset and jupyter notebooks to train the model.

For example, in my case I created a directory called openFieldMouse in my Documents and create a subfolder for the notebooks there. I then copied all the notebooks from the unetTracker repository.

```
cd ~/Documents
mkdir openFieldMouse
cd openFieldMouse/
mkdir notebooks
cd notebooks/
cp ~/repo/unetTracker/notebooks/* . 
```

The name of the notebooks starts with 01, 02, 03, etc. 

Just go through the notebooks following the numerical order. You will have to adapt the parameters and configuration to you own need.

For a few notebook indices (e.g., starting with 01), there is a 'a' and 'b' versions. The 'a' version is used if you want to work with data from a live 'camera' and the 'b' version is used if you want to work from a video file.


# Model

I opted for a U-Net a main architecture. This is a model that was developped to perform image segementation of biomedical images. 

https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

<img src="images/u-net-architecture.png" width="1000"/>

The model is defined in the `model.py` file.



