# Running unetTracker on Google Colab

Google Colab is a good option for training neural networks if you don't have access to a local GPU. Unfortunately, some components of the graphical user interface of unetTracker are currently not working in Google Colab. 

This means that you will need to do some steps locally on your computer, then upload your project directory to your Google Drive and train your network in Google Colab. You can also use your model to track objects in videos in Google Colab.

Most steps for preparing your project and datasets can be done on any recent computer.

Training your neural network requires a GPU. Rapid inference (processing video with your neural network) also requires a GPU.

## Steps to perform on your local computer without GPU (using CPU)

0. Create your project
1. Generate a dataset
2. Inspect the dataset
3. Calculate values for data normalization
4. Create training and validation datasets
5. Create your data augmentation pipeline
11. Label videos from a video you have performed tracking on

## Steps to perform on Google Colab (or with a computer equipped with a GPU)

6. Train your network
8. Evaluate your network
10. Track objects in videos with your network
