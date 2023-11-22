import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import ipywidgets as widgets
from tqdm import tqdm
from datetime import datetime
import os
import pickle

from unetTracker.trackingProject import TrackingProject

model_name = "finger_tracker"
root_path = "/home/kevin/Documents/trackingProjects"
project = TrackingProject(name=model_name,root_folder = root_path)