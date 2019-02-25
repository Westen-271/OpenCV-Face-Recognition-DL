## USAGE: python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector detection_model --embedding-model openface_nn4.small2.v1.t7

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

# Load face detector and embedder.

print("Loading face detector...")

protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])

detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load serialized face embedding model.
print(args["embedding_model"])
embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

# Grab the paths to the input images.
print("Loading faces...")

imagePaths = list(paths.list_images(args["dataset"]))

knownEmbeddings = []
knownNames = []

total = 0

# Loop over image paths.
for (i, imagePath) in enumerate(imagePaths):
    print("Processing image {} of {}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # Load the image, resize it to 600x600 and maintain aspect ratio.
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width = 600)
    (h, w) = image.shape[:2]

    # Next, construct a blob from the image.
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB = False, crop = False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    # Ensure at least one face was found.
    if len(detections) > 0:
        # Find the bounding box with largest probability.
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fH < 20 or fW < 20: # We don't want the face to be too small...can't read it then!
                continue
    
            # Otherwise, construct a blob for the face's region of interest. Pass the blob through the face embedding model to get the face quantification in 128-D.
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB = True, crop = False)

            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Add the name and the face.
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

# Dump to disk.
print("Serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
file = open(args["embeddings"], "wb")
file.write(pickle.dumps(data))
file.close()
