import matplotlib
import numpy as np
import cv2

class CoordinatesFromSegmentationMask():
    """
    Class extracing coordinates out of the output of a segmentation model.
    
    This assumes that the objects are represented by a circle centered on the object.
    
    The of the detect function is the output of a segmentation model that was transformed to a numpy array.
    
    Dimensions are : batch x object x height x width
    
    The cv2.SimpleBlobDetector is used to detect the circle center on each object.
    
    """
    
    def __init__(self,minArea=30):
        """
        Argument 
        minArea: minArea of the cv2.SimpleBlobDetector parameter.
        """
        self.params = cv2.SimpleBlobDetector_Params()
        self.params.filterByArea = True
        self.params.minArea = minArea # between 400 and 500 is the decision point for face, 30 for small mice
        self.detector = cv2.SimpleBlobDetector_create(self.params)

    def detect(self, modelOutput):
        """
        Detect the most likely position of each object tracked by a segmentation model
        
        Arguments:
        modelOutput should be the output of a segmentation model. The range of values should be 0 to 1 (torch.sigmoid()).
        """
        
        if type(modelOutput) is not np.ndarray:
            raise ValueError("modelOutput should be a numpy array")
        if modelOutput.ndim != 4:
            raise ValueError("modelOutput should have 4 dimensions: batch object, height, width")
        if modelOutput.max() > 1 or modelOutput.min() < 0 :
            raise ValueError("modelOutput range should be between 0 and 1")
        
        batchSize = modelOutput.shape[0]
        nObjects = modelOutput.shape[1]
        
        coord = np.empty((batchSize,nObjects,3)) # x, y, probability
        coord[:,:,:] = np.nan
        coord[:,:,2] = modelOutput.reshape(batchSize,nObjects,-1).max(axis=2) # maximal value in each object mask
        
        # we need integers ranging from 0 to 255 for cv2 functions, we also need to reverse the values to look for dark blobs
        outputT = 255- (modelOutput*255).astype(np.uint8)
        
        for item in range(batchSize): 
            for objectIndex in range(nObjects):

                out = np.expand_dims(outputT[item,objectIndex,:,:],axis=2)

                keypoints = self.detector.detect(out)
                if keypoints:
                    # find the largest blob
                    largest_size=0
                    pt = None
                    for k in keypoints:
                        if k.size>largest_size:
                            pt = k.pt
                    coord[item,objectIndex,0] = pt[0]
                    coord[item,objectIndex,1] = pt[1]

        return coord
