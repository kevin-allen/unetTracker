# Running unetTracker in a Docker container

## Install docker

Follow the steps here to install docker : https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

Whatever you do, don't use snap to install docker.

You should be able to run this command if all goes well, and get the nvidia-smi output.

```
sudo docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

## Create a docker image with pytorch and unetTracker

I created a simple Dockerfile in the unetTracker repository. You can use it to build your own image.

```
cd ~/repo/unetTracker
docker build -t unettracker:latest .
```

You should see the image running `docker images`.

## Runing the docker image


```
docker run --gpus all -it --rm --shm-size 10G -p 8888:8888 unettracker
```

To add access to a camera.
 
```
docker run --gpus all -it --rm --device /dev/video0  --shm-size 10G   -p 8888:8888 unettracker 
```

## Runing the docker image and accessing the file system of the host computer

When within your Docker image, you might want to save your work and access data located on the host computer running the container. You can mount these file system using the --volume argument. The change you make within these mounted directories will be preserved when you shut down the container.

```
docker run --gpus all -it --rm --device /dev/video0 -p 8888:8888 --volume /home/kevin/Documents/trackingProjects:/home/kevin/Documents/trackingProjects --volume /home/kevin/repo:/usr/src/app/repo --shm-size 10G  unettracker
```
