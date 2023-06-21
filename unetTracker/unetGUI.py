import ipywidgets as widgets
import threading
from ipywidgets import Label, HTML, HBox, Image, VBox, Box, HBox
from unetTracker.camera import bgr8_to_jpeg
import cv2
import numpy as np
import pandas as pd
import glob
import os
import torch
import ntpath
import albumentations as A
import matplotlib.pyplot as plt

"""
This is the code for the unetTracker GUI

Trick to debug a callback function:
1. Declare this widget globally: debug_view = widgets.Output(layout={'border': '1px solid black'})
2. Put the widget in the list of children so that it is visible
3. Put this line immediately above the function you want to see the output: @debug_view.capture(clear_output=True)   
"""

debug_view = widgets.Output(layout={'border': '1px solid black'})


class LabelFromCameraGUI(VBox):
    """
    Class to show images from camera and add selected frames to the dataset.
    If you give it a model it will show the image with labels. 
    This makes it easier to identify when the model struggle to find the objects.
    
    You will need to call %matplotlib widget in your notebook
    
    https://matplotlib.org/ipympl/examples/full-example.html
    
    """
    def __init__(self,camera,project,dataset,model=None,device=None):

        """
        Arguments:
        camera: a camera object from the camera module
        project: a TrackingProject object from the trackingProject module
        dataset: a MultiClassUNetDataset object from the multiClassUNetDataset module
        model: Unet object of the unet module
        """
            
        super().__init__()
        
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
        self.imgLabelWidget =  Image(format='jpeg',height=project.image_size[0]/2, width=project.image_size[1]/2)
        
        
        self.imgSnapshot = None
        plt.ioff()
        self.fig = plt.figure()
        plt.ion()
        self.fig.canvas.toolbar_visible = False
        # Disable the resizing feature
        self.fig.canvas.resizable = True
        self.fig.canvas.header_visible = False
        self.ax = self.fig.gca()

        
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
        self.saveButton = widgets.Button(description='Save to dataset',
                            disabled=False,
                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                            tooltip='Click me',
                            icon='Click me') # (FontAwesome names without the `fa-` prefix)
        
        self.playStopButtons = widgets.ToggleButtons(
                            options={'Play':0, 'Stop':1},
                            description='Camera:',
                            disabled=False,
                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                            tooltips=['Show live images' 'Stop live images'])
        
        self.captureButton = widgets.Button(description ="Capture a frame",
                            disabled=False,
                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                            tooltip='Click me')
        
        """
        Events handling
        """
        self.playStopButtons.observe(self.play_stop_handle_event, names='value')
        self.saveButton.on_click(self.save_handle_event)
        self.captureButton.on_click(self.capture_handle_event)
       
        # deal with click on the mpl canvas
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.add_coordinates_handle_event)
       
        ## start camera
        self.start_video()
        
        
        # display all widgets
        self.children = [HBox([self.imgVideoWidget,self.imgVideoTrackedWidget,self.imgLabelWidget]),
                         VBox([self.playStopButtons,self.captureButton,self.saveButton]),
                      self.htmlWidget,
                      self.coordinatesBox,  
                      self.fig.canvas]

        
    """
    Callback function to handle user inputs
    """    
    
    def add_coordinates_handle_event(self,event):
        """
        Callback function
        Clicking to add a coordinate to an object
        """
        # get coordinate for the click, object that was selected and its index
        target=(event.xdata,event.ydata)
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
        frame = self.imgSnapshot
        frameNp = cv2.imdecode(np.frombuffer(frame, np.uint8),-1)

        for i,myObject in enumerate(self.project.object_list):
            x=self.coordBounded[i][0].value
            y=self.coordBounded[i][1].value
            if(x!=0 and y!=0):
                cv2.circle(frameNp,(x,y), self.project.target_radius, self.project.object_colors[i], -1)
        self.imgLabelWidget.value = bgr8_to_jpeg(frameNp)    
    
    
    def capture_handle_event(self, event):

        frame = self.imgVideoWidget.value
          
        self.imgSnapshot = frame
        frame_np = cv2.imdecode(np.frombuffer(frame, np.uint8),-1)
        self.ax.imshow(frame_np)
        self.fig.canvas.draw()  
        self.imgLabelWidget.value = frame

        # set coordinate to 0,0
        for i in range(len(self.project.object_list)):
            self.coordBounded[i][0].value=0
            self.coordBounded[i][1].value=0
        
        self.objectRadioButtons.value=self.project.object_list[0]
        
        lines = "dataset size:{}".format(len(self.dataset))
        content = "  ".join(lines)
        self.htmlWidget.value = content

   
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
                coordinates[i,:]=np.nan
            else:
                coordinates[i,0] = self.coordBounded[i][0].value
                coordinates[i,1] = self.coordBounded[i][1].value
                
        # create the mask get mask for each object
        frame = self.imgSnapshot
        frame = cv2.imdecode(np.frombuffer(frame, np.uint8),-1)
        mask = self.dataset.create_mask(frame,coordinates,self.project.target_radius)
        
        # save the data to the dataset
        img_path, mask_path, coordinates_path = self.dataset.save_entry(frame,mask,coordinates)

         # set coordinate to 0,0
        for i in range(len(self.project.object_list)):
            self.coordBounded[i][0].value=0
            self.coordBounded[i][1].value=0
        
        self.objectRadioButtons.value=self.project.object_list[0]
        
        lines = "dataset size:{}".format(len(self.dataset))
        content = "  ".join(lines)
        self.htmlWidget.value = content
        
        self.imgLabelWidget.value = self.imgSnapshot
    
  
 
   
        
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
        output = (output>0.3).float()
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
           
        
        

#############################################
####### work with images from a video #######
####### with or without model         #######
#############################################
class RepeatTimer(threading.Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)   
            
class LabelFromVideoGUI(VBox):
    """
    Class to label frames from a camera feed.
    
    You will need to call %matplotlib widget in your notebook 
    
    """
    def __init__(self,video_fn,project,dataset,model=None,device=None):

        super().__init__()
        
        
        self.video_fn = video_fn
        self.project = project
        self.dataset = dataset
        self.model = model
        self.device = device
        
    
        # image normalization
        means = project.normalization_values["means"]
        stds = project.normalization_values["stds"]
        
        self.transform = A.Compose([A.Normalize(mean=means, std=stds)])
        
        self.image_scaling_factor =  self.project.labeling_ImageEnlargeFactor
        
        self.imgVideoWidget = Image(format='jpeg',height=project.image_size[0]/2, width=project.image_size[1]/2) # image from video 
        self.imgVideoTrackedWidget = Image(format='jpeg',height=project.image_size[0]/2, width=project.image_size[1]/2) # image from video plus label
        #self.imgSnapshotWidget = Image(format='jpeg',height=project.image_size[0]*self.image_scaling_factor, width=project.image_size[1]*self.image_scaling_factor)# captured image
        self.imgLabelWidget =  Image(format='jpeg',height=project.image_size[0], width=project.image_size[1]) # caputred image with labels
        
        self.imgSnapshot = None
        plt.ioff()
        self.fig = plt.figure()
        plt.ion()
        self.fig.canvas.toolbar_visible = False
        # Disable the resizing feature
        self.fig.canvas.resizable = True
        self.fig.canvas.header_visible = False
        self.ax = self.fig.gca()

        
        
        # to debug and display information
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
        
        
        self.previousButton = widgets.Button(description='Previous frame',
                            disabled=False,
                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                            tooltip='Click me') # (FontAwesome names without the `fa-` prefix)
        
        self.nextButton = widgets.Button(description='Next frame',
                            disabled=False,
                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                            tooltip='Click me') # (FontAwesome names without the `fa-` prefix)
        
        self.captureButton = widgets.Button(description ="Capture a frame",
                            disabled=False,
                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                            tooltip='Click me')
        
        self.saveButton = widgets.Button(description='Save labelled frame',
                            disabled=False,
                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                            tooltip='Click me',
                            icon='check') # (FontAwesome names without the `fa-` prefix)
        
        self.slider = widgets.IntSlider(value=0,min=0, max=10, step=1, description='Scroll:', disabled=False, continuous_update=False, orientation='horizontal', readout=True, readout_format='d')
        self.slider.observe(self.slider_handle_event, names='value')
        
        self.timerRunning = False
        self.timerWait=0.01
        
        self.playStopButtons = widgets.ToggleButtons(
                            options={'Play':0, 'Stop':1},
                            description='Camera:',
                            disabled=False,
                            value = 1,
                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                            tooltips=['Show live images' 'Stop live images'])
        
        
        
        """
        Events handling
        """
        
        self.nextButton.on_click(self.next_handle_event)
        self.previousButton.on_click(self.previous_handle_event)
        self.playStopButtons.observe(self.play_stop_handle_event)
        self.saveButton.on_click(self.save_handle_event)
        self.captureButton.on_click(self.capture_handle_event)
        
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.add_coordinates_handle_event)
        
      
        """
        Get the first image from the video
        """
    
        self.htmlWidget.value = "getting image from video"
        if not os.path.exists(self.video_fn):  
            raise IOError("Video file does not exist:",self.video_fn)
            
        self.cap = cv2.VideoCapture(self.video_fn)

        if (self.cap.isOpened()== False): 
            raise ValueError("Error opening video file")

        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.image_index = 0
        if self.video_length < 0:
            print("Problem calculating the video length, file likely corrupted.")
        
        self.slider.max=self.video_length
        
     
    
        
        self.htmlWidget.value = "calling update_image()"
        self.update_image()
        
        
        # put something in the matplotlib image
        if self.image_index > self.video_length:
            self.image_index = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.image_index)
        ret, image = self.cap.read()
        self.image = image
        if ret == False:
            raise ValueError("Error reading video frame")
            
        self.htmlWidget.value = "setting an image in matplotlib"
                
           
        self.children = [
                      HBox([self.imgVideoWidget,self.imgVideoTrackedWidget,
                            VBox([self.playStopButtons,
                                  self.previousButton,
                                  self.nextButton,
                                  self.slider,
                                  self.captureButton])]),
                      self.htmlWidget,
                      self.coordinatesBox, 
                      self.fig.canvas,
                      HBox([self.imgLabelWidget,self.saveButton])]
                      
            
    
        
        
        
        
 
    def add_coordinates_handle_event(self,event):
        """
        Clicking to add a coordinate to an object
        """

        #target=(round(event["relativeX"]/self.image_scaling_factor),round(event["relativeY"]/self.image_scaling_factor))
        target=(event.xdata,event.ydata)
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
        frame = self.imgSnapshot
        frameNp = cv2.imdecode(np.frombuffer(frame, np.uint8),-1)

        for i,myObject in enumerate(self.project.object_list):
            x=self.coordBounded[i][0].value
            y=self.coordBounded[i][1].value
            if(x!=0 and y!=0):
                cv2.circle(frameNp,(x,y), self.project.target_radius, self.project.object_colors[i], -1)
        self.imgLabelWidget.value = bgr8_to_jpeg(frameNp)    
    
       
    def play_stop_handle_event(self,event):
        lines = "play_stop_handle_event"
        content = "  ".join(lines)
        self.htmlWidget.value = content
        
        
        if self.timerRunning == False:
            self.playTimer = RepeatTimer(self.timerWait, self.on_play_timer)
            self.playTimer.start()
            self.timerRunning = True
        else:
            self.playTimer.cancel()
            self.timerRunning=False
            
 
    def capture_handle_event(self, event):
        
        frame = self.imgVideoWidget.value
        self.imgSnapshot = frame
        
        frame_np = self.image #cv2.imdecode(np.frombuffer(frame, np.uint8),-1)
        self.ax.imshow(frame_np)
        self.fig.canvas.draw()  
        self.imgLabelWidget.value = frame

        # set coordinate to 0,0
        for i in range(len(self.project.object_list)):
            self.coordBounded[i][0].value=0
            self.coordBounded[i][1].value=0
        
        self.objectRadioButtons.value=self.project.object_list[0]
        
        lines = "dataset size:{},{}".format(len(self.dataset),datetime.now())
        content = "  ".join(lines)
        self.htmlWidget.value = content

 
    def update_image(self):
        
        if self.image_index > self.video_length:
            self.image_index = 0
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.image_index)
        
        ret, image = self.cap.read()

        # numpy array with the image, use this for model and saving to dataset
        self.image=image.copy()
        
        if ret == False:
            raise ValueError("Error reading video frame")
        
        image_copy = image.copy()
        if self.model is not None:
            processed_image, output = self.label_current_frame_with_model(image_copy)
        else:
            processed_image = image_copy
            
        self.imgVideoTrackedWidget.value = bgr8_to_jpeg(processed_image)    
        self.imgVideoWidget.value = bgr8_to_jpeg(image)  
         
       
        
    """
    Callback function to handle user inputs
    """
   
    def next_handle_event(self, event):
        lines = "next"
        content = "  ".join(lines)
        self.htmlWidget.value = content
        
        if (self.image_index+1) < self.video_length:
            self.image_index+=1
        self.slider.value=self.image_index
        self.update_image()
   
    def on_play_timer(self):
        if (self.image_index+1) < self.video_length:
            self.image_index+=1
        else:
            self.image_index = 0
        self.slider.value=self.image_index

   
    def previous_handle_event(self, event):
        lines = "previous"
        content = "  ".join(lines)
        self.htmlWidget.value = content
        
        if (self.image_index-1) >= 0:
            self.image_index-=1
        self.slider.value=self.image_index
        self.update_image()
   
    def slider_handle_event(self,change):
        self.image_index=change["new"]
        self.update_image()
   
    def label_current_frame_with_model(self,image):
        """
        preprocess the frame,
        pass it to newtork
        use the output of the network to label the image
        
        Argument:
        image as a numpy array 
        """
        # preprocess
        input = image.astype(np.float32)
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
        
    
   
    def save_handle_event(self,event):
        # get coordinates for each object
        # arrays with 2 columns for x and y
       
        coordinates = np.empty((len(self.project.object_list),2))
      
        for i in range(len(self.project.object_list)):
            if self.coordBounded[i][0].value == 0 and self.coordBounded[i][0].value == 0:
                coordinates[i,:]=np.nan
            else:
               
                coordinates[i,0] = self.coordBounded[i][0].value
                coordinates[i,1] = self.coordBounded[i][1].value
       
        # create the mask get mask for each object
        ##frame = self.imgSnapshot
        ##frame = cv2.imdecode(np.frombuffer(frame, np.uint8),-1)
        mask = self.dataset.create_mask(frame,coordinates,self.project.target_radius)
        img_path, mask_path, coordinates_path = self.dataset.save_entry(self.image,mask,coordinates)
         
        lines = "dataset size:{}".format(len(self.dataset))
        content = "  ".join(lines)
        self.htmlWidget.value = content
        
        
        
################################
####### work with images #######
################################
class LabelFromImagesGUI(VBox):
    """
    Class to label frames from single images.
    """
    def __init__(self,image_dir,
                 frame_info_file,
                 project,dataset,
                 model=None):
        super().__init__()
        
        self.image_dir = image_dir
        self.frame_info_file = frame_info_file
        self.project = project
        self.dataset = dataset
        self.model = model
        self.image_scaling_factor = self.project.labeling_ImageEnlargeFactor
        self.image_extension = project.image_extension
        
        
        
        if os.path.exists(self.frame_info_file) == False:
            raise IOError("{} does not exist".format(self.frame_info_file))
        self.frameInfoDf = pd.read_csv(self.frame_info_file)
        
  
        self.imgLabelWidget =  Image(format='jpeg',height=project.image_size[0]/2, width=project.image_size[1]/2)
        self.htmlWidget = HTML('Event info')
        
        self.imgSnapshot = None
        plt.ioff()
        self.fig = plt.figure()
        plt.ion()
        self.fig.canvas.toolbar_visible = False
        # Disable the resizing feature
        self.fig.canvas.resizable = True
        self.fig.canvas.header_visible = False
        self.ax = self.fig.gca()

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
        
        
        
        self.images =  glob.glob(os.path.join(image_dir,f'*{self.image_extension}'))
        self.imageIndex = 0
        
        if len(self.images) == 0:
            raise ValueError(f"No image found in {self.image_dir}")
        
        
        self.imageFileName = self.images[self.imageIndex]
        self.frame = cv2.imread(self.imageFileName)
        print("frame.shape:",self.frame.shape)
        self.imgSnapshot = bgr8_to_jpeg(self.frame)
        self.ax.imshow(self.frame)
        #self.fig.canvas.draw()  
        self.imgLabelWidget.value = bgr8_to_jpeg(self.frame)
        ifn = os.path.basename(self.imageFileName)
        self.imageVideoFileName = self.frameInfoDf[self.frameInfoDf.imageFileName == ifn].videoFileName.item()
        self.imageFrameId = self.frameInfoDf[self.frameInfoDf.imageFileName == ifn].frameId.item()
        
        
        """
        Events handling
        """
        
        self.nextButton.on_click(self.next_handle_event)
        self.saveButton.on_click(self.save_handle_event)
        
        # deal with click on the mpl canvas
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.add_coordinates_handle_event)
       
        
        """
        Fill images with a default image
        """
        #self.get_next_image()

        self.children = [self.htmlWidget, self.coordinatesBox, self.nextButton,self.fig.canvas, 
                         HBox([self.imgLabelWidget,self.saveButton]),
                        debug_view]
        #self.children = [self.fig.canvas,self.coordinatesBox,self.imgLabelWidget]
       

    def get_next_image(self):
        self.imageIndex = self.imageIndex+1
        self.images = glob.glob(os.path.join(self.image_dir,f'*{self.image_extension}'))
        
        if self.imageIndex > (len(self.images)-1):
            self.imageIndex = 0
        
        self.imageFileName = self.images[self.imageIndex]
        self.frame = cv2.imread(self.imageFileName)
        ifn = os.path.basename(self.imageFileName)
        self.imageVideoFileName = self.frameInfoDf[self.frameInfoDf.imageFileName == ifn].videoFileName.item()
        self.imageFrameId = self.frameInfoDf[self.frameInfoDf.imageFileName == ifn].frameId.item()
        
        
        
        self.imgSnapshot = bgr8_to_jpeg(self.frame)
        self.imgLabelWidget.value = bgr8_to_jpeg(self.frame)
        
        
        self.ax.imshow(self.frame)
        self.fig.canvas.draw()
        
        # set coordinate to 0,0
        for i in range(len(self.project.object_list)):
            self.coordBounded[i][0].value=0
            self.coordBounded[i][1].value=0
         
        self.objectRadioButtons.value=self.project.object_list[0]
        
        lines = "image {} of {}, from {}, frameID: {}".format(self.imageIndex,len(self.images),self.imageVideoFileName,self.imageFrameId)
        content = "  ".join(lines)
        self.htmlWidget.value = content
      
       
        
    """
    Callback function to handle user inputs
    """
    def next_handle_event(self, event):
        self.get_next_image()
        
    @debug_view.capture(clear_output=True) 
    def save_handle_event(self,event):
        # get coordinates for each object
        # arrays with 2 columns for x and y
      
        coordinates = np.empty((len(self.project.object_list),2))
        # get the coordinates from widgets
        # if coordinates are 0,0, the object was not label
        for i in range(len(self.project.object_list)):
            if self.coordBounded[i][0].value == 0 and self.coordBounded[i][0].value == 0:
                coordinates[i,:]=np.nan
            else:
                coordinates[i,0] = self.coordBounded[i][0].value
                coordinates[i,1] = self.coordBounded[i][1].value
                
        # create the mask get mask for each object
        #frame = self.imgSnapshot
        #frame = cv2.imdecode(np.frombuffer(frame, np.uint8),-1)
        mask = self.dataset.create_mask(self.frame,coordinates,self.project.target_radius)
        
        # save the data to the dataset
        img_path, mask_path, coordinates_path = self.dataset.save_entry(self.frame,mask,coordinates,videoFileName=self.imageVideoFileName,
                                                                        frameId=self.imageFrameId)

         # set coordinate to 0,0
        for i in range(len(self.project.object_list)):
            self.coordBounded[i][0].value=0
            self.coordBounded[i][1].value=0
        
        self.objectRadioButtons.value=self.project.object_list[0]
        
        lines = "dataset size:{}".format(len(self.dataset))
        content = "  ".join(lines)
        self.htmlWidget.value = content
            
        # remove this image from the extracted folder
        fn = self.images[self.imageIndex]
        os.remove(fn)
        
     
      
        self.get_next_image()
        
        
    def add_coordinates_handle_event(self,event):
        """
        Clicking to add a coordinate to an object
        """
        target=(event.xdata,event.ydata)
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
        frame = self.imgSnapshot
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
        
        
        self.previousButton.on_click(self.previous_handle_event)
        self.nextButton.on_click(self.next_handle_event)
        self.deleteButton.on_click(self.delete_handle_event)
        
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
                      self.imgWidget]),
               debug_view)
    
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
            x = coord[i][0]
            y = coord[i][1]
            if not np.any(np.isnan(coord[i,:])):
                x=int(x)
                y=int(y)
                cv2.circle(frame,(x,y), self.project.target_radius, self.project.object_colors[i], -1)
        return frame
        
        
