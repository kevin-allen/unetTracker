## Installation

You can create a new Python environment and run the Jupyter notebook from within this environmnt. 



### Create your virtual environment

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
git clone https://github.com/kevin-allen/unet-tracker.git
```

### Install the unet-tracker package

```
python3 -m pip install -e unetTracker/
```

To test that the package is installed. 

```
ipython
```
```
from unetTracker.trackingProject import TrackingProject
```

### Add jupyter extension for ipyevents

You will need to create a jupyter extensions that allows us to create a small GUI within the notebook. 

The instructions can be found here: [ipyevents](https://github.com/mwcraig/ipyevents) package.

```
jupyter labextension install @jupyter-widgets/jupyterlab-manager ipyevents
```

You will have to make sure that the jupyter lab extensions are enable (click on the icon that looks like a puzzel piece called extension manager, far left of the jupyter lab window).


### Run jupyter lab

Start jupyter lab and get going with the notebooks in the unet-tracker repository.

```
jupyter lab
```





