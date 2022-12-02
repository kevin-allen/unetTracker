## Installation


### NVIDIA GPU and drivers

You will need to have a GPU and GPU drivers loaded in order to use the GPU to train your network.

You can get instructions from [NVIDIA](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) to do this.


### Create your virtual environment

If you have a conda environment already activated, do `conda deactivate` and make sure you are not in a virtual environment.

Create a virtual environment using venv and install the required packages.

Here the virtual environment will be called `torch`.

```
cd ~
sudo apt-get install pip
python3 -m pip install --user --upgrade pip
python3 -m venv torch
source torch/bin/activate
```

Make sure your `torch` environment is activated for all the remaining installation steps.

To activate the `torch` environment...
```
cd 
source torch/bin/activate
```


### Install ipwidgets, ipyevents and jupyterlab

Within the `torch` environment, run the following.

```
pip3 install ipyevents ipywidgets jupyterlab
```


### Add a jupyter extension for ipyevents

You will need to create a jupyter extensions that allows us to create a small GUI within the notebook. 

The instructions can be found here: [ipyevents](https://github.com/mwcraig/ipyevents) package.


#### nodejs

On my Unbuntu computer, I had to first install a recent nodejs. I got the instructions [here](https://github.com/nodesource/distributions).

```
sudo apt-get install curl
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash - &&\
sudo apt-get install -y nodejs
```

You can double-check that the installation worked.

```
node -v
```

You should have version 16 installed.


Now build the jupyter lab extension

#### Build the ipyevents jupyterlab extension

```
jupyter labextension install @jupyter-widgets/jupyterlab-manager ipyevents
```

If you have a warning: `ValueError: Please install nodejs >=12.0.0 before continuing`

Remove the current version of nodejs and reinstall the one above.



### Install pytorch and some other libraries

```
pip3 install torch torchvision matplotlib pandas tqdm  imgaug albumentations 
```

### Clone the unet-tracker repository

```
mkdir ~/repo
cd ~/repo
git clone https://github.com/kevin-allen/unetTracker.git
```

### Install the unet-tracker package

```
cd unetTracker
python3 -m pip install -e .
```

To test that the package is installed. 

```
ipython
```
```
from unetTracker.trackingProject import TrackingProject
```



### Run jupyter lab


You will need to run `source torch/bin/activate` each time you open a terminal to activate your virtual environment. 

You can decide to activate it by default by putting `source torch/bin/activate` at the end of your `bash.rc` file.

Start jupyter lab and get going with the notebooks in the unet-tracker repository.

If you are working on a desktop machine, just run this command to start the jupyter lab server.

```
jupyter lab
```


### Run jupyter lab on a remote server

If you are working on a remote server, run this on the server

```
jupyter lab --no-browser
```

On your local machine, run

```
ssh -N -L 8889:localhost:8889 kevin@a230-pc89
```

Then paste the address of the jupyter server into your browser. It should look like `http://localhost:8889/lab?token=d0427240ec80edfab108b8a0e69a3d8sdfasdfasdfasdfa6`


You will have to make sure that the jupyter lab extensions are enable (click on the icon that looks like a puzzel piece called extension manager, far left of the jupyter lab window).


### ipyevents

If you have problem using the GUI part of unetTracker, make sure the ipyevents are working. [Test available here](ipyevent_test.md)

