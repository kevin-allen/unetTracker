import ipywidgets as widgets
import threading
from ipywidgets import Label, HTML, HBox, Image, VBox, Box, HBox
from ipyevents import Event 
from IPython.display import display
from unetTracker.camera import bgr8_to_jpeg
import cv2
import numpy as np
import glob
import os
import torch
import ntpath
import albumentations as A

class LabelFromCameraGUI():
    """
    Class to show images from camera and add selected frames to the dataset.
    If you give it a model it will show the image with labels. 
    This makes it easier to identify when the model struggle to find the objects.
    
    """
    def __init__(self,camera,project,dataset,model=None,device=None):

        """
        Arguments:
        camera: a camera object from the camera module
        project: a TrackingProject object from the trackingProject module
        dataset: a MultiClassUNetDataset object from the multiClassUNetDataset module
        model: Unet object of the unet module
        """
            
        self.camera = camera
        self.project = project
        self.dataset = dataset
        self.model = model
        self.device = device
        
        
        # image normalization
        means = project.normalization_values["means"]
        stds = project.normalization_values["stds"]
        self.transform = A.Compose([A.Normalize(mean=means, std=stds)])
        
        
        self.currentFrame = None
        self.image_scaling_factor = 1
        
         # image widgets to display different images
        self.imgVideoWidget = Image(format='jpeg',height=project.image_size[0]/2, width=project.image_size[1]/2)  
        self.imgVideoTrackedWidget = Image(format='jpeg',height=project.image_size[0]/2, width=project.image_size[1]/2)
        self.imgSnapshotWidget = Image(format='jpeg',height=project.image_size[0], width=project.image_size[1])
        self.imgLabelWidget =  Image(format='jpeg',height=240, width=320)
        
        # to debug and report information to user
        self.htmlWidget = HTML('Event info')
        
        # widgests for object selection and coordinates
        self.objectLabel = widgets.Label(value='Objects:')
        self.objectRadioButtons = widgets.RadioButtons(
            options=self.project.object_list,
            value=self.project.object_list[0],
            layout={'width': 'max-content'})
        self.coordBounded=[] # a list of list (one per object) of 2 IntText (x and y coordinates)
        for i in enumerate(self.project.object_list):
            self.coordBounded.append([widgets.IntText(value=None, description='X:',disabled=False),widgets.IntText(value=None, description='Y:',disabled=False)])
        manyHBoxes = [ HBox(coords) for coords in self.coordBounded]
        self.coordinatesBox = HBox([self.objectLabel,self.objectRadioButtons,VBox(manyHBoxes)])
        
        # save and play-stop buttons
        self.saveButton = widgets.Button(description='Save labelled frame',
                            disabled=False,
                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                            tooltip='Click me',
                            icon='check') # (FontAwesome names without the `fa-` prefix)
        
        self.playStopButtons = widgets.ToggleButtons(
                            options={'Play':0, 'Stop':1},
                            description='Camera:',
                            disabled=False,
                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                            tooltips=['Show live images' 'Stop live images'])
        
        """
        Events handling
        """
        self.playStopEvent = Event(source=self.playStopButtons, watched_events=['click'])
        self.playStopEvent.on_dom_event(self.play_stop_handle_event)
        
        self.captureEvent = Event(source=self.imgVideoWidget, watched_events=['click'])
        self.captureEvent.on_dom_event(self.capture_handle_event)
        
        self.addCoordinatesEvent = Event(source=self.imgSnapshotWidget, watched_events=['click'])
        self.addCoordinatesEvent.on_dom_event(self.add_coordinates_handle_event)
        
        self.saveEvent =  Event(source=self.saveButton, watched_events=['click'])
        self.saveEvent.on_dom_event(self.save_handle_event)
        
        
        ## start camera
        self.start_video()
        # display all widgets
        display(VBox([HBox([self.imgVideoWidget,self.imgVideoTrackedWidget,self.playStopButtons]),self.htmlWidget,self.coordinatesBox, self.imgSnapshotWidget,HBox([self.imgLabelWidget,self.saveButton])]))

        
    """
    Callback function to handle user inputs
    """    
    def save_handle_event(self,event):
        """
        Callback to save image, mask and coordinates to dataset
        """
        # get coordinates for each object
        # arrays with 2 columns for x and y

        coordinates = np.empty((len(self.project.object_list),2))
        # get the coordinates from widgets
        # if coordinates are 0,0, the object was not label
        for i in range(len(self.project.object_list)):
            if self.coordBounded[i][0].value == 0 and self.coordBounded[i][0].value == 0:
                coordinates[:,0]=np.nan
            else:
                coordinates[i,0] = self.coordBounded[i][0].value
                coordinates[i,1] = self.coordBounded[i][1].value
                
        # create the mask get mask for each object
        frame = self.imgSnapshotWidget.value
        frame = cv2.imdecode(np.frombuffer(frame, np.uint8),-1)
        mask = self.dataset.create_mask(frame,coordinates,self.project.target_radius)
        
        # save the data to the dataset
        img_path, mask_path, coordinates_path = self.dataset.save_entry(frame,mask,coordinates)

    
    def add_coordinates_handle_event(self,event):
        """
        Callback function
        Clicking to add a coordinate to an object
        """
        # get coordinate for the click, object that was selected and its index
        target=(round(event["relativeX"]/self.image_scaling_factor),round(event["relativeY"]/self.image_scaling_factor))
        selectedObject = self.objectRadioButtons.value
        objectIndex = self.project.object_list.index(selectedObject)

        # get the coordinates in the intText widgets
        self.coordBounded[objectIndex][0].value=target[0]
        self.coordBounded[objectIndex][1].value=target[1]

        # move to the next index
        objectIndex=objectIndex+1
        if objectIndex == len(self.project.object_list):
            objectIndex=0

        lines = "select:{}, index:{}".format(selectedObject,objectIndex)
        content = "  ".join(lines)
        self.htmlWidget.value = content
                                      
        self.objectRadioButtons.value=self.project.object_list[objectIndex]

        # label the frame with current coordinates
        frame = self.imgSnapshotWidget.value
        frameNp = cv2.imdecode(np.frombuffer(frame, np.uint8),-1)

        for i,myObject in enumerate(self.project.object_list):
            x=self.coordBounded[i][0].value
            y=self.coordBounded[i][1].value
            if(x!=0 and y!=0):
                cv2.circle(frameNp,(x,y), self.project.target_radius, self.project.object_colors[i], -1)
        self.imgLabelWidget.value = bgr8_to_jpeg(frameNp)    
    
 
    def capture_handle_event(self, event):

        frame = self.imgVideoWidget.value
        target=(event["relativeX"],event["relativeY"])

        self.imgSnapshotWidget.value = frame
        self.imgLabelWidget.value = frame

        # set coordinate to 0,0
        for i in range(len(self.project.object_list)):
            self.coordBounded[i][0].value=0
            self.coordBounded[i][1].value=0
            
        lines = "{},{}, dataset size:{}".format(target[0],target[1],len(self.dataset))
        content = "  ".join(lines)
        self.htmlWidget.value = content

        
    def label_current_frame_with_model(self,image):
        """
        preprocess the frame,
        pass it to newtork
        use the output of the network to label the image
        
        Argument:
        image as a numpy array 
        """
        # preprocess
        input = image.copy().astype(np.float32)
        input = self.transform(image=input)  
        input = input["image"]
        input = torch.tensor(input).to(self.device).permute(2,0,1).unsqueeze(0).float()
        
        # model prediction
        output = torch.sigmoid(self.model(input))
        output = (output>0.5).float()
        output = output[0,:].to("cpu").detach().numpy()

        for i in range(output.shape[0]):
            idx=(output[i]==1.0)
            for c in range(3):
                image[idx,c]=self.project.object_colors[i][c]

        return image, output
        
        
    """
    Functions to deal with camera
    """
    def play_stop_handle_event(self,event):
        if self.camera.running == False:
            self.start_video()
        else:
            self.stop_video()     
        
    def start_video(self):
        self.camera.running = True
        self.camera.observe(self.update_image, names='value')   
    
    def stop_video(self):
        if self.camera.running == True:
            self.camera.unobserve(self.update_image, names='value')
            self.camera.running = False
    
    def update_image(self,change):
        image = change['new']
        
        image_copy = image.copy()
        
        if self.model is not None:
            processed_image, output = self.label_current_frame_with_model(image_copy)
        else:
            processed_image = image_copy
        
        self.imgVideoTrackedWidget.value = bgr8_to_jpeg(processed_image)    
        self.imgVideoWidget.value = bgr8_to_jpeg(image)   
           
        
        
        
        
        
        
        
        
        
################################
####### work with images #######
################################
class LabelFromImagesGUI():
    """
    Class to label frames from a camera feed.
    """
    def __init__(self,image_dir,project,dataset,model=None):

        self.image_dir = image_dir
        self.project = project
        self.dataset = dataset
        self.model = model
        self.image_scaling_factor = self.project.labeling_ImageEnlargeFactor
        
        
        
        self.imgSnapshotWidget = Image(format='jpeg',height=project.image_size[0]*self.image_scaling_factor, width=project.image_size[1]*self.image_scaling_factor)
        self.imgLabelWidget =  Image(format='jpeg',height=project.image_size[0], width=project.image_size[1])
        self.htmlWidget = HTML('Event info')
        
        
        self.objectLabel = widgets.Label(value='Objects:')
        self.objectRadioButtons = widgets.RadioButtons(
            options=self.project.object_list,
            value=self.project.object_list[0],
            layout={'width': 'max-content'})

        self.coordBounded=[] # a list of list (one per object) of 2 IntText (x and y coordinates)
        for i in enumerate(self.project.object_list):
            self.coordBounded.append([widgets.IntText(value=None, description='X:',disabled=False),widgets.IntText(value=None, description='Y:',disabled=False)])

            
        manyHBoxes = [ HBox(coords) for coords in self.coordBounded]
        
        self.coordinatesBox = HBox([self.objectLabel,self.objectRadioButtons,VBox(manyHBoxes)])
        
        
        self.nextButton = widgets.Button(description='Next frame',
                            disabled=False,
                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                            tooltip='Click me',
                            icon='check') # (FontAwesome names without the `fa-` prefix)
        
        
        self.saveButton = widgets.Button(description='Save labelled frame',
                            disabled=False,
                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                            tooltip='Click me',
                            icon='check') # (FontAwesome names without the `fa-` prefix)
        
        self.images =  glob.glob(os.path.join(image_dir,'*.jpg'))
        self.imageIndex = 0
        frame = cv2.imread(self.images[self.imageIndex],0)
        self.imgSnapshotWidget.value = bgr8_to_jpeg(frame)
        self.imgLabelWidget.value = bgr8_to_jpeg(frame)
        
        """
        Events handling
        """
        
        self.nextEvent =  Event(source=self.nextButton, watched_events=['click'])
        self.nextEvent.on_dom_event(self.next_handle_event)
        
        
        
        
        self.addCoordinatesEvent = Event(source=self.imgSnapshotWidget, watched_events=['click'])
        self.addCoordinatesEvent.on_dom_event(self.add_coordinates_handle_event)
        
        self.saveEvent =  Event(source=self.saveButton, watched_events=['click'])
        self.saveEvent.on_dom_event(self.save_handle_event)
        
        """
        Fill images with a default image
        """
        
       
       # self.imgVideoWidget.value = bgr8_to_jpeg(image)
        #self.imgSnapshotWidget.value = bgr8_to_jpeg(image)
        #self.imgLabelWidget.value = bgr8_to_jpeg(image)
        
        display(VBox([self.htmlWidget, self.coordinatesBox, HBox([self.imgSnapshotWidget,self.nextButton]),HBox([self.imgLabelWidget,self.saveButton])]))
        
        self.get_next_image()


    def get_next_image(self):
        self.imageIndex = self.imageIndex+1
        self.images = glob.glob(os.path.join(self.image_dir,'*.jpg'))
        
        if self.imageIndex > (len(self.images)-1):
            self.imageIndex = 0
                
        frame = cv2.imread(self.images[self.imageIndex])
        
        self.imgSnapshotWidget.value = bgr8_to_jpeg(frame)
        self.imgLabelWidget.value = bgr8_to_jpeg(frame)
        
        # set coordinate to 0,0
        for i in range(len(self.project.object_list)):
            self.coordBounded[i][0].value=0
            self.coordBounded[i][1].value=0

        lines = "image {} of {}".format(self.imageIndex,len(self.images))
        content = "  ".join(lines)
        self.htmlWidget.value = content
      
       
        
    """
    Callback function to handle user inputs
    """
    def next_handle_event(self, event):
        self.get_next_image()
        
        
    def save_handle_event(self,event):
        # get coordinates for each object
        # arrays with 2 columns for x and y
       
        coordinates = np.empty((len(self.project.object_list),2))
      
        for i in range(len(self.project.object_list)):
            if self.coordBounded[i][0].value == 0 and self.coordBounded[i][0].value == 0:
                coordinates[:,0]=np.nan
            else:
               
                coordinates[i,0] = self.coordBounded[i][0].value
                coordinates[i,1] = self.coordBounded[i][1].value
       
        # create the mask get mask for each object
        frame = self.imgSnapshotWidget.value
        frame = cv2.imdecode(np.frombuffer(frame, np.uint8),-1)
        mask = self.dataset.create_mask(frame,coordinates,self.project.target_radius)
        img_path, mask_path, coordinates_path = self.dataset.save_entry(frame,mask,coordinates)
        
       
        # remove this image from the extracted folder
        fn = self.images[self.imageIndex]
        os.remove(fn)
        
     
      
        self.get_next_image()
        
        
    def add_coordinates_handle_event(self,event):
        """
        Clicking to add a coordinate to an object
        """

        target=(round(event["relativeX"]/self.image_scaling_factor),round(event["relativeY"]/self.image_scaling_factor))
        selectedObject = self.objectRadioButtons.value
        objectIndex = self.project.object_list.index(selectedObject)

        # get the coordinates in the intText widgets
        self.coordBounded[objectIndex][0].value=target[0]
        self.coordBounded[objectIndex][1].value=target[1]

        # move to the next index
        objectIndex=objectIndex+1
        if objectIndex == len(self.project.object_list):
            objectIndex=0

        lines = "select:{}, index:{}".format(selectedObject,objectIndex)
        content = "  ".join(lines)
        self.htmlWidget.value = content
                                      
        self.objectRadioButtons.value=self.project.object_list[objectIndex]

        # label the frame with current coordinates
        frame = self.imgSnapshotWidget.value
        frameNp = cv2.imdecode(np.frombuffer(frame, np.uint8),-1)

        for i,myObject in enumerate(self.project.object_list):
            x=self.coordBounded[i][0].value
            y=self.coordBounded[i][1].value
            if(x!=0 and y!=0):
                cv2.circle(frameNp,(x,y), self.project.target_radius, self.project.object_colors[i], -1)
        self.imgLabelWidget.value = bgr8_to_jpeg(frameNp)    
    
       
        

    
################################
####### review a dataset #######
################################
class ReviewDatasetGUI():
    """
    Class to label frames from a camera feed.
    """
    def __init__(self,project,dataset):

        self.project = project
        self.dataset = dataset
        
        self.imgWidget = Image(format='jpeg',height=project.image_size[0], width=project.image_size[1])
        self.htmlWidget = HTML('Event info')
        self.frameNameWidget = HTML('Frame name')
        
        
        self.previousButton = widgets.Button(description='Previous frame',
                            disabled=False,
                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                            tooltip='Click me',
                            icon='check') # (FontAwesome names without the `fa-` prefix)
        
        self.nextButton = widgets.Button(description='Next frame',
                            disabled=False,
                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                            tooltip='Click me',
                            icon='check') # (FontAwesome names without the `fa-` prefix)
        
        
        self.deleteButton = widgets.Button(description='Delete frame',
                            disabled=False,
                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                            tooltip='Click me',
                            icon='check') # (FontAwesome names without the `fa-` prefix)
        
        
        self.previousEvent = Event(source=self.previousButton, watched_events=['click'])
        self.previousEvent.on_dom_event(self.previous_handle_event)
        
        self.nextEvent = Event(source=self.nextButton, watched_events=['click'])
        self.nextEvent.on_dom_event(self.next_handle_event)
        
        
        self.deleteEvent = Event(source=self.deleteButton, watched_events=['click'])
        self.deleteEvent.on_dom_event(self.delete_handle_event)
        
        
        
        self.imageIndex = 0
        frame = self.get_labelled_image(self.imageIndex)
        self.imgWidget.value = bgr8_to_jpeg(frame)
        
        fn = self.dataset.get_image_file_name(self.imageIndex)
        lines = ntpath.basename(fn) 
        content = "  ".join(lines)
        self.frameNameWidget.value = content   
        
        
        
        lines = "{} / {}".format(self.imageIndex,len(dataset))
        content = "  ".join(lines)
        self.htmlWidget.value = content   
            
        display(VBox([HBox([self.htmlWidget, self.previousButton, self.nextButton, self.deleteButton]),
                      self.frameNameWidget,
                      self.imgWidget]))
    
    def delete_handle_event(self,event):
        self.dataset.delete_entry(self.imageIndex)
        
        frame = self.get_labelled_image(self.imageIndex)
        fn = self.dataset.get_image_file_name(self.imageIndex)
        self.imgWidget.value = bgr8_to_jpeg(frame)
        
        fn = self.dataset.get_image_file_name(self.imageIndex)
        lines = ntpath.basename(fn) 
        content = "  ".join(lines)
        self.frameNameWidget.value = content   
        

    def previous_handle_event(self,event):
        self.imageIndex-=1
        if self.imageIndex < 0:
            self.imageIndex =  len(self.dataset)-1
       
        fn = self.dataset.get_image_file_name(self.imageIndex)
        lines = ntpath.basename(fn) 
        content = "  ".join(lines)
        self.frameNameWidget.value = content   
    
        lines = "{} / {}".format(self.imageIndex,len(self.dataset))
        content = "  ".join(lines)
        self.htmlWidget.value = content   
        
        frame = self.get_labelled_image(self.imageIndex)
        self.imgWidget.value = bgr8_to_jpeg(frame)    
       
        
    def next_handle_event(self,event):
        self.imageIndex+=1
        if self.imageIndex >= len(self.dataset):
            self.imageIndex = 0
        
        fn = self.dataset.get_image_file_name(self.imageIndex)
        lines = ntpath.basename(fn) 
        content = "  ".join(lines)
        self.frameNameWidget.value = content   
       
        lines = "{} / {}".format(self.imageIndex,len(self.dataset))
        content = "  ".join(lines)
        self.htmlWidget.value = content   
    
    
        frame = self.get_labelled_image(self.imageIndex)
        
       
        
        self.imgWidget.value = bgr8_to_jpeg(frame)
        
    
    def get_labelled_image(self,index):
        frame,mask,coord = self.dataset[self.imageIndex]
        frame = frame.permute(1,2,0).numpy()
        
        for i,myObject in enumerate(self.project.object_list):
            x=int(coord[i][0])
            y=int(coord[i][1])
            if(x!=0 and y!=0):
                cv2.circle(frame,(x,y), self.project.target_radius, self.project.object_colors[i], -1)
        return frame
        
        