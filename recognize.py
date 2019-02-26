# Usage

import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# Arguments!
parser = argparse.ArgumentParser()

parser.add_argument("-d", "--detector", required = True,
                    help = "The path to the OpenCV deep learning face detector.")

parser.add_argument("-m", "--embedding-model", required = True,
                    help = "The path to the OpenCV deep learning face embedding model.")

parser.add_argument("-r", "--recognizer", required = True,
                    help = "The path to the model trained to recognize faces.")

parser.add_argument("-l", "--le", required = True,
                    help = "The path to the label encoder.")

parser.add_argument("-c", "--confidence", type = float, default = 0.8,
                    help = "The minimum probability (filters weak detections)")

args = vars(parser.parse_args())

# Load serialized face detector from disk.
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])

detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load face embedding model from disk.
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# Load actual face recognition model & label encoder.
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# Load the image and resize it to width of 600 pixels.
configPath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'OpenCV-Face-Recognition-DL\\config'))
videoConfigFile = open(configPath + '\\video_capture_device.cfg', 'r')

videoCaptureDevice = videoConfigFile.read()

if not videoCaptureDevice.isnumeric():
    videoCaptureDevice = 0
else:
    videoCaptureDevice = int(videoCaptureDevice)
    
video = cv2.VideoCapture(videoCaptureDevice)

while True:
    ret, frame = video.read()
    
    image = frame
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # Img -> blob
    imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Apply face detector
    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
     
        # filter out weak detections
        if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for the
                # face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
     
                # extract the face ROI
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
     
                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                        continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
     
                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
 
    # show the output image
    cv2.imshow("Image", image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
