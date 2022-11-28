## Installation

You can create a new Python environment and run the Jupyter notebook from within this environmnt. 



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
pip3 install torch torchvision torchaudio matplotlib pandas ipywidgets tqdm jupyterlab imgaug albumentations ipyevents
```


You will need to run `source torch/bin/activate` each time you open a terminal to activate your virtual environment. 
You can decide to activate it by default by putting `source torch/bin/activate` at the end of your `bash.rc` file.


### Clone the unet-tracker repository

```
cd ~/repo
git clone https://github.com/kevin-allen/unet-tracker.git
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

### Add a jupyter extension for ipyevents

You will need to create a jupyter extensions that allows us to create a small GUI within the notebook. 

The instructions can be found here: [ipyevents](https://github.com/mwcraig/ipyevents) package.

On my Unbuntu computer, I had to first install nodejs. I got the instructions [here](https://github.com/nodesource/distributions).

```
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash - &&\
sudo apt-get install -y nodejs
```

You can double-check that the installation worked.

```
node -v
```

Then you should be ready to build the jupyter lab extension

```
jupyter labextension install @jupyter-widgets/jupyterlab-manager ipyevents
```


### Run jupyter lab

Start jupyter lab and get going with the notebooks in the unet-tracker repository.

If you are working on a desktop machine, just run this command to start the jupyter lab server.

```
jupyter lab
```

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





