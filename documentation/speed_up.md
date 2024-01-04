# Speed up inference time

One limitation of using a U-net to process images is that these models are not particularly fast when processing larger images. If you want to perform position detection in real-time or if you have long videos to process, this might be an issue.

Below are two strategies to speed up the inference time. 

## Reduce the model size

By default, the number of convolution kernels in the different layers of the u-net is set to 64, 128, 256, 512. One strategy is to use smaller values, for example, 32, 64, 128, 256. This will reduce the number of computations needed to process an image and speed up the inference. 

You can control the number of features in a model by setting the argument `unet_features=[32,64,128,256]` when creating your project with the function 

```
project = TrackingProject(name="finger_tracker",
                          root_folder = root_path,
                          object_list=["f1","f2","f3","f4"],
                          unet_features=[32,64,128,256],
                          target_radius=6)
```

If you already have a working project and want to change the number of features, you can edit the variable `'unet_features' in the `config.yalm` file. You must retrain the model from scratch if you change these values.


## Combine two models working on full size images or cropped images

A second strategy applies if the objects you are tracking are found in a small region of the images. For example, when tracking a small mouse in a large maze. 

This strategy is to have two models, one processing full-frame images and one processing cropped images. When we know where the tracked objects are in the image, we process only cropped images, which is faster. 

With cropped images of 200x200 pixels, you can process more than 100 frames a second with an NVIDIA RTX4080 GPU.

The algorithm starts by searching for the tracked object using full-frame images. Once the objects are located, we can track the objects by using cropped images centered on the last known position of the tracked object. 

1. Train a model using full-frame images.
2. Create a model to process cropped images. You can use the Notebook called `notebooks/15_cropped_network.ipynb` to create a project and labeled dataset for cropped images.
3. Train the cropped model like you did for the full-frame model.
4. You can use the code in `notebooks/17_track_large_cropped.ipynb` to track your object using a combination of the 2 models.
