# unet-tracker

Code to train and deploy a neural network to track objects in images, videos, or camera feeds.

The user can follow a few Jupyter notebooks that will instruct how to set up your project, prepare a dataset, train your neural network and deploy your model to detect object in images (or videos).

The project uses PyTorch as a deep learning framework. Training should be done on GPUs. Inference can be done using the CPU (slower), a GPU on a desktop or a Jetson single-board computer.

## Installation

You can create a new Python environment and run the Jupyter notebook from within this environmnt. Alternatively, you can download a docker container and run it on your computer. I would suggest option 1 if you want to do some code development. Option 2 is great if you just want to get going quickly with you model.


### Option 1: Create a Python environment and run a Jupyter lab server

#### Clone the unet-tracker repository

```
git clone https://github.com/kevin-allen/unet-tracker.git
```

#### Create your virtual environment

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

#### Add jupyter extension for ipyevents

You will need to create a jupyter extensions that allows us to create a small GUI within the notebook. 

The instructions can be found here: [ipyevents](https://github.com/mwcraig/ipyevents) package.

```
jupyter labextension install @jupyter-widgets/jupyterlab-manager ipyevents
```

You will have to make sure that the jupyter lab extensions are enable (click on the icon that looks like a puzzel piece called extension manager, far left of the jupyter lab window).


#### Run jupyter lab

Start jupyter lab and get going with the notebooks in the unet-tracker repository.

```
jupyter lab
```


### Option 2: Run the code in a Docker container.



