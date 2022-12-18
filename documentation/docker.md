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
docker run --gpus all -it --rm --shm-size 10G -p 8888:8888 unettracker
```

To add access to a camera.
 
```
docker run --gpus all -it --rm --device /dev/video0  --shm-size 10G   -p 8888:8888 unettracker 
```

To add access to some directories on the host.

```
docker run --gpus all -it --rm --device /dev/video0 -p 8888:8888 --volume /home/kevin/Documents/trackingProjects:/home/kevin/Documents/trackingProjects  --shm-size 10G  unettracker
```


##
trt_ts_module = torch_tensorrt.compile(traced_model,  
    inputs = [torch_tensorrt.Input([2, 3, 480, 640], dtype=torch.float)], # Datatype of input tensor. Allowed options torch.(float|half|int8|int32|bool),
    enabled_precisions = {torch.half}) # Run with FP16)
