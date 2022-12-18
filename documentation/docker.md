# Running unetTracker in a Docker container

## Install docker

Follow the steps here to install docker : https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

Whatever you do, don't use snap to install docker.

You should be able to run this command if all goes well, and get the nvidia-smi output.

```
sudo docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

## Create a docker image with pytorch and unetTracker


```
cd ~/repo/unetTracker
docker build -t unettracker:latest .
```

You should see the image running `docker images`.

## Runing the docker image

```
docker run --gpus all -it --rm  -p 8888:8888 unettracker
```
