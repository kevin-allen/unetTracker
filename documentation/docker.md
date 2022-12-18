# Running unetTracker in a Docker container

## Install docker

Follow the steps here to install docker : https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

Whatever you do, don't use snap to install docker. This version will not give you access to the GPU.

You can follow these steps to run docker without having to use sudo all the time: https://docs.docker.com/engine/install/linux-postinstall/

You should be able to run this command if all goes well, and get the nvidia-smi output.


## Desktop computer and Jetson

We need to build the image using a different image as the base depending on whether we are creating the image for a Jetson or a desktop computer.

The Docker file for the desktop Ubuntu computer is in `unetTracker/Docker` and the one for Jetson is in `unetTracker/Docker_jetson`.


## Create a docker image with pytorch and unetTracker (Desktop computer)

Run this to make sure that docker has access to your GPU.

```
docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

I created a simple Dockerfile in the `unetTracker/Docker` directory. You can use it to build your own image.

You might want to check that the base image at the top of the Docker file is suitable for your computer.

```
cd ~/repo/unetTracker/Docker
docker build -t unettracker:latest .
```

You should see the image running `docker images`.

## Create a docker image with pytorch and unetTracker (Jetson computer)

I created a simple Dockerfile in the `unetTracker/Docker` directory. You can use it to build your own image.

You might want to check that the base image at the top of the Docker file is suitable for your Jetpack.

You can run `cat /etc/` to check which version of the JetPack you have installed. This will determine which base image you can use.

```
cd ~/repo/unetTracker/Docker_jetson
docker build -t unettracker:latest .
```





## Runing the docker image and accessing the file system of the host computer

Running the image will launch a jupyter lab server that you can access at http://localhost:8888 using your browser.

The password is `unet`.


When within your Docker image, you might want to save your work and access data located on the host computer running the container. You can mount these file system using the --volume argument. The change you make within these mounted directories will be preserved when you shut down the container.

In the example below, I am accessing 2 directories on the host machine.

* My python code that I might want to modify
* A directory with the data of my tracking projects (unetTracker project, datasets, etc.).

On Desktop
```
docker run --gpus all -it --rm --device /dev/video0 -p 8888:8888 --volume /home/kevin/Documents/trackingProjects:/home/kevin/Documents/trackingProjects --volume /home/kevin/repo:/usr/src/app/repo --shm-size 10G  unettracker
```

On Jetson
```
docker run --runtime nvidia  -it --rm -p 8888:8888  --device /dev/video0 --shm-size 16G   --volume /home/kevin/Documents/trackingProjects:/home/kevin/Documents/trackingProjects --volume /home/kevin/repo:/usr/src/app/repo  unettracker
```

## Install UnetTracker in the container

Run this in a jupyter notebook.

```
! cd /usr/src/app/repo/unetTracker && ! python3 -m pip install -e .
```
