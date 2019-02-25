# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
 
# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--dataset", required = True,
	help="The path to the dataset (faces + images)")

parser.add_argument("-e", "--embeddings", required = True,
	help="The path to the output for serialized db - of facial embeddings (rois)")

parser.add_argument("-d", "--detector", required = True,
	help="Path to the OpenCV deep learning face detector.")

parser.add_argument("-m", "--embedding-model", required = True,
	help="Path to the OpenCV face embedding model.")

parser.add_argument("-c", "--confidence", type=float, default=0.5,
	help="The minimum probability - filters weak detections.")

args = vars(parser.parse_args())
