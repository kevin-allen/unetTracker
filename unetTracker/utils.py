import os
from unetTracker.coordinatesFromSegmentationMask import CoordinatesFromSegmentationMask
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

def extract_object_position_from_video(project,transform,model,device,video_fn,blobMinArea=30):
    """
    Function to extract the position of objects in a video
    
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
    #video_length = 10

    print("Number of frames:",video_length)


    if video_length < 0:
        raise ValueError("Problem calculating the video length, file likely corrupted.")


    all_coords = np.empty((video_length,len(project.object_list)*3))
    for i in tqdm(range(video_length)):
        ret, image = cap.read()

        if ret == False:
            raise ValueError("Error reading video frame")

        input = image.astype(np.float32)
        input = transform(image=input) # normalize 
        input = input["image"]
        # transform to torch tensor, send to gpu, permute the dimensions and unsqueeze to make a batch
        input = torch.tensor(input).to(device).permute(2,0,1).unsqueeze(0).float()

        # model prediction
        output = torch.sigmoid(model(input))
        # batch to image, move to cpu memory, transform to numpy array
        output = output.to("cpu").detach().numpy() 
        coord = detector.detect(output)
        all_coords[i] = coord.reshape(1,-1).squeeze() # one row of x,y,prob,x,y,prob,...

    cap.release()


    df = pd.DataFrame()
    for i, ob in enumerate(project.object_list):
        df[f"{ob}_x"] = all_coords[:,i*3+0]
        df[f"{ob}_y"] = all_coords[:,i*3+1]
        df[f"{ob}_p"] = all_coords[:,i*3+2]

    return df
