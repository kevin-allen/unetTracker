import torch
import torch.utils.data
import glob
import PIL.Image as pilImage
import subprocess
import cv2
import os
import glob
import uuid
import subprocess
from  torchvision.transforms import ToTensor,Resize, Normalize
import ntpath
from sklearn.model_selection import train_test_split
import shutil
import numpy as np
import albumentations as A

class MultiClassUNetDataset(torch.utils.data.Dataset):
    """
    Represent our data for image segmentation.
    
    Can be used to 
    1) get data
    2) save new images and masks
    3) create train and validation datasets
    """
    
    
    def __init__(self, image_dir, mask_dir,coordinates_dir,transform=None):
        super(MultiClassUNetDataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.coordinates_dir = coordinates_dir
        
        self.images = [ntpath.basename(path) for path in glob.glob(os.path.join(image_dir,'*.jpg'))]
        self.masks = [ fn.replace(".jpg",'_mask.npy') for fn in self.images]
        
        # variable to set data augmentation transformation
        self.augmentation_RandomSizedCropProb = 1.0
        self.augmentation_HorizontalFlipProb = 0.0 # If you do right-left discrimination, don't flip the image
        self.augmentation_RotateProb = 0.3
        self.augmentation_RandomBrightnessContrastProb = 0.2
        
        if transform == "default":
            print("Using default augmentation")
            original_height = 480
            original_width = 640
            self.transform =  A.Compose([   
                    A.RandomSizedCrop(min_max_height=(original_height-50, original_height),w2h_ratio=original_width/original_height,height=original_height, width=original_width, p=self.augmentation_RandomSizedCropProb),
                    A.HorizontalFlip(p=self.augmentation_HorizontalFlipProb),
                    A.Rotate (limit=30,border_mode=cv2.BORDER_CONSTANT,p=self.augmentation_RotateProb),
                    A.RandomBrightnessContrast(p=self.augmentation_RandomBrightnessContrastProb)])
        else:
            self.transform = transform
        
        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):

        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg",'_mask.npy'))
        coordinates_path = os.path.join(self.coordinates_dir, self.images[index].replace(".jpg",'_coordinates.csv'))
        
    
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = np.load(mask_path).astype(np.float32)
        
        coordinates = np.loadtxt(coordinates_path)
        
        
        # apply data augmentation 
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # to tensor and permute
        image = torch.from_numpy(image).permute(2,0,1)
        mask = torch.from_numpy(mask).permute(2,0,1)
        
        # normalize
        image = image/255
        mask = mask
        
        
        return image, mask, coordinates # we only need one channel for the mask
    
    def get_image_file_name(self,index):
        img_path = os.path.join(self.image_dir, self.images[index])
        return img_path
    
    
    def save_entry(self, image, mask, coordinates):
        """
        
        Saves an image in BGR8 format to dataset, as jpg
        Save coordinates as csv file
        Save mask as a numpy array (npy file)
        
        """  
        filename_img = str(uuid.uuid1()) + '.jpg'
        image_path = os.path.join(self.image_dir, filename_img)
        cv2.imwrite(image_path, image)
        
        filename = filename_img.replace(".jpg",'_mask.npy')
        mask_path = os.path.join(self.mask_dir, filename)
        np.save(mask_path, mask)
        
        filename = filename_img.replace(".jpg",'_coordinates.csv')
        coordinates_path = os.path.join(self.coordinates_dir, filename)
        np.savetxt(coordinates_path, coordinates)
        
        self.images = [ntpath.basename(path) for path in glob.glob(os.path.join(self.image_dir,'*.jpg'))]
        self.masks = [ fn.replace(".jpg",'_mask.npy') for fn in self.images]
        self.coordinates = [ fn.replace(".jpg",'_coordinates.csv') for fn in self.images]
        
        return image_path,mask_path,coordinates_path
    
    def delete_entry(self, index):
        """
        Delete an entry in the dataset.
        
        Argument:
        index: index of the entry to delete
        """
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg",'_mask.npy'))
        coordinates_path = os.path.join(self.coordinates_dir, self.images[index].replace(".jpg",'_coordinates.csv'))
    
        for filename in [img_path, mask_path, coordinates_path]:
            os.remove(filename)
            
        self.images = [ntpath.basename(path) for path in glob.glob(os.path.join(self.image_dir,'*.jpg'))]
        self.masks = [ fn.replace(".jpg",'_mask.npy') for fn in self.images]
        self.coordinates = [ fn.replace(".jpg",'_coordinates.csv') for fn in self.images]
        
    
    
    def create_mask(self,image, coordinates, radius):
        """
        Create a mask for an image
        
        Arguments
        image: 3D np.array
        coordinates: 2D np.array, 2 columns for x and y coordinate of each object, one row per object
        radius: scalar radius of the target in the mask
        
        Returns
        mask: 3D array of np.bool_ (1 byte per datapoint), first 2 dimensions are the size of image, 3rd dimension is one per object (number of rows in coordinates)
        """
        mask = np.zeros((image.shape[0],image.shape[1],coordinates.shape[0]),dtype=np.bool_)
        rPoints = np.arange(image.shape[0])
        cPoints = np.arange(image.shape[1])
        xs,ys = np.meshgrid(cPoints, rPoints)

        for i in range(coordinates.shape[0]):
             if np.sum(np.isnan(coordinates[i,:])) ==0:
                distanceMap=np.sqrt((xs-coordinates[i][0])**2 +  (ys-coordinates[i][1])**2)
                mask[distanceMap<radius,i]=1
        
        return mask
    
    def create_training_validation_dataset(self,
                                            train_images_dir = "data/noseDataset/train_images",
                                            train_masks_dir = "data/noseDataset/train_masks",
                                            train_coordinates_dir = "data/noseDataset/train_coordinates",
                                            val_images_dir = "data/noseDataset/val_images",
                                            val_masks_dir = "data/noseDataset/val_masks",
                                            val_coordinates_dir = "data/noseDataset/val_coordinates",
                                           
                                           test_size=0.15):
        """
        Function to create a training and validation dataset out of the current dataset.
        
        The images and masks will be splitted into 2 dataset using sklearn.model_selection.train_test_split()
        
        Arguments
        train_images_dir
        train_masks_dir
        train_coordinates_dir
        val_images_dir
        val_masks_dir
        val_coordinates_dir
        test_size
        """
        
        if len(self.images) == 0:
            print("There is no data to create a training and validation dataset")
            return
        
        # create directories if needed
        for folder in [train_images_dir,train_masks_dir, train_coordinates_dir , val_images_dir,val_masks_dir,val_coordinates_dir]:
            if not os.path.exists(folder):
                print("Create",folder)
                os.makedirs(folder)
                
        # Delete any previous image files in those directories
        for folder in [train_images_dir,train_masks_dir, train_coordinates_dir , val_images_dir,val_masks_dir,val_coordinates_dir]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)

                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
                    
        X = self.images
        y = self.masks
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        print("Length of training set: {}".format(len(X_train)))
        print("Length of validation set: {}".format(len(X_test)))
        
        # Copy the files to the 4 directories
        print("Copying files to training and validation directories")
        for fn in X_train:
            src = os.path.join(self.image_dir,fn)
            dst = os.path.join(train_images_dir,fn)
            shutil.copyfile(src, dst)
            
            src = os.path.join(self.coordinates_dir,fn.replace(".jpg",'_coordinates.csv'))
            dst = os.path.join(train_coordinates_dir,fn.replace(".jpg",'_coordinates.csv'))
            shutil.copyfile(src, dst)
            
            
        for fn in X_test:
            src = os.path.join(self.image_dir,fn)
            dst = os.path.join(val_images_dir,fn)
            shutil.copyfile(src, dst)
            
            src = os.path.join(self.coordinates_dir,fn.replace(".jpg",'_coordinates.csv'))
            dst = os.path.join(val_coordinates_dir,fn.replace(".jpg",'_coordinates.csv'))
            shutil.copyfile(src, dst)
            
            
            

        for fn in y_train:
            src = os.path.join(self.mask_dir,fn)
            dst = os.path.join(train_masks_dir,fn)
            shutil.copyfile(src, dst)

        for fn in y_test:
            src = os.path.join(self.mask_dir,fn)
            dst = os.path.join(val_masks_dir,fn)
            shutil.copyfile(src, dst)

       
        
    def extract_frames_from_video(self,video_fn, number_frames, frame_directory):
        """
        Function to extract frames from a video. The frames are chosen randomly.
        
        Arguments
        video_fn: File name of the video
        number_frames: Number of frames to extract
        frame_directory: Directory in which to save the images
        """

        if not os.path.exists(frame_directory):
            print("Create",frame_directory)
            os.makedirs(frame_directory)

        cap = cv2.VideoCapture(video_fn)

        if (cap.isOpened()== False): 
            print("Error opening video file")
            return

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sel_frames = np.random.choice(np.arange(length),size=number_frames, replace=False)
        sel_frames.sort()

        print("Extracting frames:", sel_frames, "to",frame_directory)

        for i in sel_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES,i)
            ret, frame = cap.read()

            if ret == False:
                print 


            filename_img = str(uuid.uuid1()) + '.jpg'
            image_path = os.path.join(frame_directory, filename_img)
            cv2.imwrite(image_path, frame)

        cap.release() 