import yaml
import os
import matplotlib
from datetime import datetime
import torch
import warnings

class TrackingProject():
    """
    Represent our tracking project
    This class is mainly used to define project specific variables,
    save and load the configuration file and create project directories.
    
    There is a config.yalm file that can be edited manually to change settings.
    
    """
    
    def __init__(self, name, root_folder, object_list=None, target_radius=None, transform=None):
        
        self.name = name
        self.project_dir = os.path.join(root_folder,self.name)
        print("Project directory:",self.project_dir)
        self.dataset_dir = os.path.join(self.project_dir,"dataset")
        self.image_dir = os.path.join(self.dataset_dir,"images")
        self.mask_dir = os.path.join(self.dataset_dir,"masks")
        self.coordinate_dir = os.path.join(self.dataset_dir,"coordinates")
        self.models_dir = os.path.join(self.project_dir,"models")
        self.augmentation_dir = os.path.join(self.project_dir,"augmentation")
        
        self.config_fn = os.path.join(self.project_dir,"config.yalm")
        self.model_fn = os.path.join(self.models_dir,"UNet.pt")
        self.image_size = [480,640]  # height,width
        
        
        # variables for data augmentation
        self.augmentation_RandomSizedCropProb = 1.0
        self.augmentation_HorizontalFlipProb = 0.5
        self.augmentation_RotateProb = 0.3
        self.augmentation_RandomBrightnessContrastProb = 0.2
        
        # variables for normalization
        self.normalization_values = None
               
        
        # for labeling object, so that it is easier to be presice with the mouse click
        self.labeling_ImageEnlargeFactor = 2.0
        
        
        if object_list is None: # assumes we are supposed to get the details from a config file.
            self.load_configuration()
        else: # assumes the user is setting up a new project
            
            if os.path.exists(self.project_dir):
                warnings.warn("The directory {} already exist.\n If you run save_configuration() you will overwrite the previous configuration.".format(self.project_dir))
              
            
            self.object_list = object_list
            self.set_object_colors()
            self.target_radius=target_radius
    
    def set_object_colors(self):
        rgb_colors = {}
        for name, hex in matplotlib.colors.cnames.items():
            tp = matplotlib.colors.to_rgb(hex)
            rgb_colors[name]=(tp[0]*255,tp[1]*255,tp[2]*255)
        
        # preferred default colors
        self.object_colors = [rgb_colors["blue"], rgb_colors["red"],rgb_colors["yellow"],
                              rgb_colors["azure"],rgb_colors["green"],rgb_colors["darkgrey"],
                              rgb_colors["deeppink"],rgb_colors["firebrick"]]
        
        # if we have enough preffered default colors
        if len(self.object_list) < len(self.object_colors):
            self.object_colors = self.object_colors[:len(self.object_list)]
        else: # else use colors from the color list, there are 141
            clist = [rgb_colors[key] for key in rgb_colors]
            self.object_colors = clist[:len(self.object_list)]
    
    
    def create_project_directories(self):
        """
        Create all the directory for the project and dataset
        """
        # create directories if needed
        for dir in [self.project_dir, self.dataset_dir, self.image_dir, self.coordinate_dir, self.mask_dir,self.models_dir,self.augmentation_dir]:
            if not os.path.exists(dir):
                print("Create",dir)
                os.makedirs(dir)
    
    def save_model(self,model):
        """
        Save our model for later
        
        """
        print("saving model state dict to",self.model_fn)
        print(datetime.now())
        torch.save(model.state_dict(),self.model_fn)
        
    
    def load_model(self,model):
        """
        Load the previous model state
        """
        model.load_state_dict(torch.load(self.model_fn))
        model.eval()
    
    def set_normalization_values(self,means,stds):
        """
        Set the normalization values to use for the model inputs
        
        Will be stored in the configuration files to avoid recaculating all the time
        
        Arguments:
        means: 1D numpy array with the mean of the 3 image channels
        stds: 1D numpy array with the standard deviation of the 3 image channels
        """
        self.normalization_values = {"means":means.tolist(),
                                    "stds":stds.tolist()}
        
    
    def save_configuration(self):
        """
        Save the configuration to a config.yaml file
        """
        print("Saving",self.config_fn)
        doc_file = {"name":self.name, 
                    "objects":self.object_list, 
                    "object_colors":self.object_colors,
                    "target_radius":self.target_radius,
                    "image_size":self.image_size,
                    "augmentation_RandomSizedCropProb": self.augmentation_RandomSizedCropProb,
                    "augmentation_HorizontalFlipProb": self.augmentation_HorizontalFlipProb,
                    "augmentation_RotateProb": self.augmentation_RotateProb,
                    "augmentation_RandomBrightnessContrastProb": self.augmentation_RandomBrightnessContrastProb,
                    "labeling_ImageEnlargeFactor": self.labeling_ImageEnlargeFactor,
                    "normalization_values": self.normalization_values
                   }
        
        with open(self.config_fn, 'w') as file:
            yaml.dump(doc_file, file,default_flow_style=False)
    
    def load_configuration(self):
        """
        Load the configuration from a config.yaml file
        """
        if os.path.exists(self.config_fn):
            print("Loading",self.config_fn)
            with open(self.config_fn) as file:
                self.configDict = yaml.full_load(file)
            self.object_list=self.configDict["objects"]
            self.object_colors = self.configDict["object_colors"]
            self.target_radius = self.configDict["target_radius"]
            self.image_size = self.configDict["image_size"]
            self.augmentation_RandomSizedCropProb = self.configDict["augmentation_RandomSizedCropProb"]
            self.augmentation_HorizontalFlipProb = self.configDict["augmentation_HorizontalFlipProb"]
            self.augmentation_RotateProb = self.configDict["augmentation_RotateProb"]
            self.augmentation_RandomBrightnessContrastProb = self.configDict["augmentation_RandomBrightnessContrastProb"]
            self.labeling_ImageEnlargeFactor = self.configDict["labeling_ImageEnlargeFactor"]
            self.normalization_values = self.configDict["normalization_values"]
            print(self.configDict)
            
        else:
            raise IOError("No configuration file present,",self.config_fn)
            