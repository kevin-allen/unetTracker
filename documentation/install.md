# Windows installation

### Install Anaconda

1. Download the Anaconda installation file from https://docs.anaconda.com/free/anaconda/install/windows/
2. Double-click on the downloaded file to install Anaconda. Use the default settings and note the installation directory

### Create a conda environment in which you will use unetTracker

1. Open Anaconda.Navigator
2. Click on Environments
3. Click create
4. Enter the name "torch" to create an environment named torch. Select python 3.11.5

### Install torch

1. Open Anaconda powershell prompt
2. `conda activate torch`
3. `conda install pytorch torchvision torchaudio cpuonly -c pytorch`

### Test torch installation
1. Open Anaconda powershell prompt
2. `conda activate torch`
3. `python`
4. In the python interpreter, run `import torch`

### Install jupyter lab in your torch environment
1. Open Anaconda powershell prompt
2. `conda activate torch`
3. `conda install -c conda-forge jupyterlab`
4. Close and restart Anaconda powershell prompt and activate the torch environment
5. `jupyter lab`
6. 

### Install git for windows

1. Download the installer
2. Double-click on the installer
3. Choose the default installation.

### Get unetTracker repository

1. Open a git terminal
2. Create a repo directory
3. cd repo
4. git clone https://github.com/kevin-allen/unetTracker.git


### Instan unetTracker

1. Start Anaconda powershell prompt and activate your torch environment
2. Install requirements
```
conda install ipywidgets, ipympl, matplotlib, pandas, tqdm, imgaug
conda install -c conda-forge albumentations
```

3. Install unetTracker
```
cd repo/unetTracker
python -m pip install -e .
```
https://drive.google.com/file/d/1PT66JhMFxMxS90nMWFE9JkDnvkfJgSK9/view?usp=sharing


# Installation on Ubuntu


### NVIDIA GPU and drivers

If you have a Nivida GPU, you will need to install the drivers for it and install the CUDA library.

To do this, you can get instructions from [NVIDIA](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html).

### Create your virtual environment

You can install unetTracker and its dependencies into a pip virtual environment. 

If you already have a conda environment, do `conda deactivate` and ensure you are not in a conda virtual environment.

Create a virtual environment using `venv` and install the required packages.

Here, the virtual environment will be called `torch`.

```
cd ~
sudo apt-get install pip
python3 -m pip install --user --upgrade pip
python3 -m venv torch
source torch/bin/activate
```

If this stalls, check that your proxy is set properly. For instance, at the DKFZ, you would use this code.

```
echo $https_proxy
export https_proxy=www-int2.inet.dkfz-heidelberg.de:80
```

Ensure your `torch` environment is activated for all the remaining installation steps.

To activate the `torch` environment...
```
cd 
source torch/bin/activate
```

### Clone the unetTracker repository

```
mkdir ~/repo
cd ~/repo
git clone https://github.com/kevin-allen/unetTracker.git
```

### Install the unetTracker package

```
cd unetTracker
pip3 install -r requirements.txt 
python3 -m pip install -e .
```

### Testing your installation

You should now be able to import unetTracker and torch from Python.

In a terminal, get a python terminal
```
ipython
```

Within python...

```
import torch
import unetTracker
```

Check if a Cuda device (GPU) is available.

```
torch.cuda.is_available()
```

If you have a Cuda-supported GPU installed correctly, this should return True.


### PyTorch for Jetson

If you are using a Nvidia Jetson instead of a Linux PC, you probably want to install Pytorch as indicated on the [Nvidia website](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)


### Run jupyter lab


You must run `source torch/bin/activate` each time you open a terminal to activate your virtual environment. 

You can decide to activate it by default by putting `source torch/bin/activate` at the end of your `bash.rc` file.

Start Hupyter lab and get going with the notebooks in the unetTracker repository.

If you are working on a desktop machine, just run this command to start the Jupyter lab server.

```
jupyter lab
```

If you run the following line, you should see that ipywidgets is in the list of selected packages.

```
Jupyter --version
```



### Run jupyter lab on a remote server

If you are working on a remote server, run this on the server.

```
jupyter lab --no-browser
```

On your local machine, run

```
ssh -N -L 8889:localhost:8889 kevin@a230-pc89
```

Then paste the address of the Jupyter server into your browser. It should look like `http://localhost:8889/lab?token=d0427240ec80edfab108b8a0e69a3d8sdfasdfasdfasdfa6`


You will have to make sure that the jupyter lab extensions are enabled (click on the icon that looks like a puzzle piece called extension manager, far left of the jupyter lab window).

