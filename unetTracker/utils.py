import os
from unetTracker.coordinatesFromSegmentationMask import CoordinatesFromSegmentationMask
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch


def extract_object_position_from_video(project,transform,model,device,video_fn,
                                       blobMinArea=30,nFrames=None,startFrameIndex=0,
                                       BGR2RGBTransformation=False,plotData=False):
    """
    Function to extract the position of objects in a video
    Arguments
    project: your unet project object
    transform: tranformation to apply, usually the same as for the validation set. You don't want to apply rotation, flip, etc. Only normalization
    model: model for the inference
    device: on which device the model is
    video_fn: path of the video file 
    blobMinArea: minimal size of the blob that will be considered, goes to the openCV function.
    nFrames: number of frames to process. Will start at frame 0 and go up to nFrame. If not given the whole file is processed
    BGR2RGBTransformation: whether to apply a BGR to RGB transformation. This should be the same as when you load images in your dataset class
    plotData: plot the images used as model input and the output of the model. Only use this for debugging as it will be slow.
    
    
    Return
    Pandas DataFrame with the object position within each video frame
    For each object, there is 3 columns in the data frame (x,y,probability).
    """
    
    detector = CoordinatesFromSegmentationMask(minArea=blobMinArea)
    
    if not os.path.exists(video_fn):  
        raise IOError("Video file does not exist:",video_fn)

    cap = cv2.VideoCapture(video_fn)

    if (cap.isOpened()== False): 
        raise ValueError("Error opening video file")

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Number of frames in {}: {}".format(video_fn,video_length))


    if video_length < 0:
        raise ValueError("Problem calculating the video length, file likely corrupted.")
        
    if startFrameIndex is None:
        startFrameIndex = 0
    
    if startFrameIndex < 0: 
        raise ValueError("startFrameIndex should be 0 or larger.")
    if startFrameIndex >= video_length:
        raise ValueError("startFrameIndex should be not be larger than the number of frame in the video ({})".format(video_length))
        
        
    if nFrames is None:
        nFrames = video_length-startFrameIndex    
    if nFrames < 1:
        raise ValueError("nFrames should be larger than 0 but was {}".format(nFrames))
    if startFrameIndex + nFrames > video_length:
        raise ValueError("startFrameIndex+nFrames should be smaller than the video length of {}".format(video_length))
                    
    print("Processing {} frames from index {}".format(nFrames,startFrameIndex))
    
    # set the correct location in the file
    cap.set(cv2.CAP_PROP_POS_FRAMES, startFrameIndex)
    
    all_coords = np.empty((nFrames,len(project.object_list)*3))
    for i in tqdm(range(nFrames)):
        ret, image = cap.read()

        if ret == False:
            raise ValueError("Error reading video frame")
            
        input = image.astype(np.float32)
        if BGR2RGBTransformation:
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            
        normInput = transform(image=input)["image"] # normalize 
        
        if plotData:
            fig, ax = plt.subplots(1,2,figsize=(6,3))
            ax[0].imshow(image)
            ax[1].imshow(normInput)
            plt.show()
        
        
        
        # transform to torch tensor, send to gpu, permute the dimensions and unsqueeze to make a batch
        input = torch.tensor(normInput).to(device).permute(2,0,1).unsqueeze(0).float()

        # model prediction
        output = torch.sigmoid(model(input))
        # batch to image, move to cpu memory, transform to numpy array
        output = output.to("cpu").detach().numpy() 
        
    
        if plotData:
            fig, ax = plt.subplots(1,output.shape[1]+1,figsize=((output.shape[1]+1)*3,3))
            ax[0].imshow(normInput)
            ax[0].set_title("Image")
            for j in range(output.shape[1]):
                ax[j+1].imshow(output[0,j])
                ax[j+1].set_title(project.object_list[j])

        coord = detector.detect(output)
        all_coords[i] = coord.squeeze().reshape(1,-1)
      
    cap.release()

    df = pd.DataFrame()
    for i, ob in enumerate(project.object_list):
        df[f"{ob}_x"] = all_coords[:,i*3+0]
        df[f"{ob}_y"] = all_coords[:,i*3+1]
        df[f"{ob}_p"] = all_coords[:,i*3+2]

    return df



def label_video(project,video_fn,tracking_fn, label_fn,nFrames=None):
    """
    Function to label a video (add a marker at the coordinate of the detected objects)
    
    Arguments:
    video_fn: file name of the video to label
    tracking_fn: tracking data for the video to label (Pandas.DataFrame with x,y,p for each object)
    label_fn: name of the labelled video file that will be created
    nFrames: number of frames to process
    """
    df = pd.read_csv(tracking_fn)

    if os.path.exists(label_fn):
        raise IOError(f"{label_fn} already exists, please remove it")

    if not os.path.exists(video_fn):  
        raise IOError("Video file does not exist:",video_fn)

    cap = cv2.VideoCapture(video_fn)

    if (cap.isOpened()== False): 
        raise ValueError("Error opening video file")

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampling_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    print("Number of frames:",video_length)
    
    if video_length < 0:
        raise ValueError("Problem calculating the video length, file likely corrupted.")

    if nFrames is not None:
        if nFrames < 1:
            raise ValueError("nFrames should be larger than 0 but was {}".format(nFrames))
        if nFrames > video_length:
            raise ValueError("nFrames should be smaller than the video length of {}".format(video_length))
        video_length=nFrames    
        
        
        
        
    size=project.image_size[1],project.image_size[0]
    writer = cv2.VideoWriter(label_fn, cv2.VideoWriter_fourcc(*'MJPG'),sampling_rate, size)

    d = df.to_numpy() # to facilitate indexing with numbers

    for i in tqdm(range(video_length)):
        ret, image = cap.read()
        if ret == False:
            raise ValueError("Error reading video frame")

        for j,obj in enumerate(project.object_list):
            if ~np.isnan(d[i,j*3+0]):
                cv2.circle(image,(int(d[i,j*3+0]),int(d[i,j*3+1])), 5, project.object_colors[j], -1)
        writer.write(image)

    cap.release()
    writer.release()