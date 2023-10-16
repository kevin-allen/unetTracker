import os
from unetTracker.coordinatesFromSegmentationMask import CoordinatesFromSegmentationMask
from unetTracker.cropper import Cropper
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from datetime import datetime


def check_accuracy(model,loader,device):
    """
    Function to test the performance of a unet tracker model
    
    It iterates through the data in the data loader, make predictions and compare the masks to the predictions made by the model
    
    """
    cDetector = CoordinatesFromSegmentationMask()
    
    num_correct = 0
    num_positive = 0
    num_pixels = 0
    num_positive_pixels_masks = 0
    
    dice_score = 0
    num_mask = 0
    num_mask_detected = 0
    num_detected = 0
    sum_distance = 0

    model.eval()
    with torch.no_grad():
        for x,y,c in loader:
            x = x.to(device)
            y = y.to(device)
            output = torch.sigmoid(model(x))
            preds = (output > 0.5).float()
            num_positive += preds.sum()
            num_positive_pixels_masks += y.sum()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds * y).sum() / ((preds+y).sum() + 1e-8)) # work only for binary

            # proportion of the mask detected
            num_mask += y.sum()
            num_mask_detected += preds[y==1.0].sum()
            num_detected += preds.sum()

            # distance between predicted coordinates and labelled coordinates
            output = output.detach().cpu().numpy()
            pred_coords = cDetector.detect(output)

            sum_distance+= np.nanmean(np.sqrt(((pred_coords[:,:,0:2] - c.numpy())**2).sum(axis=2))) # calculate the distance between predicted coordinates and the coordinates from the dataset.
            # we acutally do a mean of the error for the different objects in a batch

    print(f"Number of positive pixels predicted: {num_positive}")
    print(f"Number of positive pixels in masks: {num_positive_pixels_masks}")
    print(f"Percentage of positive pixels predicted: {(num_positive/num_pixels)*100:.3f}")
    print(f"Percentage of positive pixels in masks: {(num_positive_pixels_masks/num_pixels)*100:.3f}")
    
    print(f"Accuracy: {num_correct/num_pixels*100:.3f}")
    print(f"Dice score: {dice_score/len(loader):.3f}")
    print(f"Mask pixels detected (True positives): {num_mask_detected/num_mask*100:.3f}%")
    print(f"False positives: {(num_detected-num_mask_detected)/num_detected*100:.3f}%")
    print(f"Mean distance: {sum_distance/len(loader)}")
    a = model.train()




def extract_object_position_from_video(project,transform,model,device,video_fn,
                                       blobMinArea=30,nFrames=None,startFrameIndex=0,
                                       BGR2RGBTransformation=False,plotData=True,
                                      mask=None):
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
    mask: numpy array containing a mask. The array should have the save width and height as the video image
    
    
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
        
        if mask is not None:
            if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
                raise ValueError("mask shape is not the same as image shape")
                
            normInput = cv2.bitwise_or(normInput,normInput,mask = mask)
            
        
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

def extract_object_position_from_video_cropped(project_large,project_cropped,
                                               transform,model_large,model_cropped,
                                               device,video_fn,
                                               blobMinArea=30,nFrames=None,startFrameIndex=0,
                                               BGR2RGBTransformation=False,plotData=True,
                                               mask=None):
    """
    Function to extract the position of objects in a video
    
    We are combining a model working on full size image and one model working on cropped images. The aim is to speed up inference by analyzing 
    the image when the object was last seen.
    
    If we know where the mouse is, we use a cropped image to speed up inference.
    If we don't know where the mouse is, we used the full size image.
    
    
    Arguments
    project_large: your unet project object for large images (full video size)
    project_cropped: your unet project object for cropped images
    transform: tranformation to apply, usually the same as for the validation set. You don't want to apply rotation, flip, etc. Only normalization
    model_large: model for the inference on full-size images
    model_cropped: model for the inference on cropped images
    device: on which device the models are
    video_fn: path of the video file 
    blobMinArea: minimal size of the blob that will be considered, goes to the openCV function.
    nFrames: number of frames to process. Will start at frame 0 and go up to nFrame. If not given the whole file is processed
    BGR2RGBTransformation: whether to apply a BGR to RGB transformation. This should be the same as when you load images in your dataset class
    plotData: plot the images used as model input and the output of the model. Only use this for debugging as it will be slow.
    mask: numpy array containing a mask. The array should have the save width and height as the video image
    
    Return
    Pandas DataFrame with the object position within each video frame
    For each object, there is 3 columns in the data frame (x,y,probability).
    """
    
    detector = CoordinatesFromSegmentationMask(minArea=blobMinArea) # get coordinates from the output of our models
    
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
    
    
    
    # class to deal with cropping operations
    cropper = Cropper(project_cropped.image_size)
    
    targetLocked=False # flag to decide whether to work on cropped images or full-size images
    
    # set the correct location in the file
    cap.set(cv2.CAP_PROP_POS_FRAMES, startFrameIndex)
    
    all_coords = np.empty((nFrames,len(project_large.object_list)*3))
    for i in tqdm(range(nFrames)):

        ################
        ## get image ###
        ret, image = cap.read()
        if ret == False:
            raise ValueError("Error reading video frame")
    
        ####################
        ## preprocessing ###
        start_time_preprocessing = datetime.now()
        input = image.astype(np.float32)
        if BGR2RGBTransformation:
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        normInput = transform(image=input)["image"] # normalize 
        if mask is not None:
            if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
                raise ValueError("mask shape is not the same as image shape")
            normInput = cv2.bitwise_or(normInput,normInput,mask = mask) 
        end_time_preprocessing = datetime.now()
       
    
        ####################
        ## crop operation ##
        if targetLocked: # we know where the target is, crop the image
            normInput = cropper.crop_image(normInput,crop_coordinate_center)
    
        ############
        ### plot ###
        if plotData:
            fig, ax = plt.subplots(1,2,figsize=(6,3))
            ax[0].imshow(image)
            ax[1].imshow(normInput)
            plt.show()
       
    
    
    
        # get model run time
        start_time_model = datetime.now()
        # transform to torch tensor, send to gpu, permute the dimensions and unsqueeze to make a batch
        input = torch.tensor(normInput).to(device).permute(2,0,1).unsqueeze(0).float()
        # model prediction
        if targetLocked:
            output = torch.sigmoid(model_cropped(input))
        else:
            output = torch.sigmoid(model_large(input))
            
        # batch to image, move to cpu memory, transform to numpy array
        output = output.to("cpu").detach().numpy() 
        end_time_model = datetime.now()
        
        # image detector time
        start_time_detect = datetime.now()
        coord = detector.detect(output)
        coord = coord.squeeze()
        
        
        if targetLocked:
            # translate to the full-size image coordinates system
            for j in range(output.shape[1]):
                x,y = cropper.translate_to_full_size_coordinate(coord[j,0],coord[j,1])
                coord[j,0]=x
                coord[j,1]=y
                                                                              
        if np.any(~np.isnan(coord[:,0:2])):
            crop_coordinate_center = np.nanmean(coord[:,0:2],axis=0)
        else:
            crop_coordinate_center = np.array([np.nan,np.nan])
        #print("crop_coordinate_center:",crop_coordinate_center)
        
        ##############################################
        ## determined if target is locked or not  ####
        ##############################################
        # all object found and we have valid crop_coordinate_center
                  
        if np.all(coord[:,2]>0.90) and np.all(~np.isnan(crop_coordinate_center)):
            targetLocked=True
        
        # all objects are missing or we have invalid crop_coordinate_center
        if np.all(coord[:,2]<0.3) or np.any(np.isnan(crop_coordinate_center)):
            targetLocked=False
        
        all_coords[i] = coord.reshape(1,-1)
        end_time_detect = datetime.now()
        
        
        if plotData:
            fig, ax = plt.subplots(1,output.shape[1]+1,figsize=((output.shape[1]+1)*3,3))
            ax[0].imshow(normInput)
            ax[0].set_title("Image")
            for j in range(output.shape[1]):
                ax[j+1].imshow(output[0,j])
                ax[j+1].set_title(project_large.object_list[j])

      
    cap.release()

    df = pd.DataFrame()
    for i, ob in enumerate(project_large.object_list):
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