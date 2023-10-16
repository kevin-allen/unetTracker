import numpy as np
class Cropper():
    """
    Class to crop images and translate cropped coordinates to full-size image coordinate system.
    """
    
    def __init__(self,cropped_image_size=[100,100]):
        """
        Argument 
        cropped_image_size: list with the row,column size of the cropped images
        """
        self.cropped_image_size=cropped_image_size
        

    def crop_image(self, image,crop_coordinate_center):
        """
        Crop an image
        
        Arguments:
        image: numpy array with x,y,color axes
        crop_coordinate_center: numpy array with x,y coordinate of the center of the cropped 
        """
        image_size = image.shape[0:2]
        
        if np.any(np.isnan(crop_coordinate_center)):
            raise ValueError("crop_coordinate_center: {}".format(crop_coordinate_center))
        
        
         # adjust the crop coordinate center to ensure we have enough pixels in the 4 directions from the center point
        for i in range(2):
            if (crop_coordinate_center[i]+self.cropped_image_size[i]/2 > image_size[i]):
                crop_coordinate_center[i] = image_size[i]-self.cropped_image_size[i]/2
            if (crop_coordinate_center[i]-self.cropped_image_size[i]/2 < 0):
                crop_coordinate_center[i] = self.cropped_image_size[i]/2

        # indices for the crop operation
        x,X = int(crop_coordinate_center[0]-self.cropped_image_size[0]/2),int(crop_coordinate_center[0]+self.cropped_image_size[0]/2)
        y,Y = int(crop_coordinate_center[1]-self.cropped_image_size[1]/2),int(crop_coordinate_center[1]+self.cropped_image_size[1]/2)

        # keep a memory of the last crop indices
        self.x = x
        self.y = y 
        
        # crop and return
        return image[y:Y,x:X]
        

        
    def translate_to_full_size_coordinate(self,x,y):
        """
        Translate coordinates from cropped image space to full-size image space
        
        Use the crop indices of the last cropping operation
        
        """
        return (self.x + x,self.y+y)
        