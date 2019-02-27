import cv2
import numpy as np
import argparse
import os.path
import time

# Construct arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascade", required = True,
                    help = "Name of the face cascade to use.")

parser.add_argument("-o", "--output", required = True,
                    help = "The name of the output file to use.")

args = vars(parser.parse_args())

##configPath = configPath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", '\\config'))
##videoConfigFile = open(configPath + '\\video_capture_device.cfg', 'r')
##
##print(configPath)
##
##videoCaptureDevice = videoConfigFile.read()
##
##if not videoCaptureDevice.isnumeric():
##    videoCaptureDevice = 0
##else:
##    videoCaptureDevice = int(videoCaptureDevice)
    
video = cv2.VideoCapture(1)

# Get the path of the cascade file that has been requested.
cascade_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'cascades/'))
cascade_path = cascade_path + '\\' + args["cascade"]
face_cascade = cv2.CascadeClassifier(cascade_path)

time.sleep(2) # Wait 2 seconds before capturing!

total = 0

while True:
    ret, frame = video.read()

    original = frame.copy()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Recogniser', frame)
    
    

    if cv2.waitKey(1) & 0xFF == ord('k'):
        
        fPath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'dataset'))
        fPath += '\\' + args["output"]
        
        if not os.path.exists(fPath):
            os.mkdir(fPath)
        
        fPath += '\\' + '{}.png'.format(str(total).zfill(5))
        cv2.imwrite(fPath, original)
        print("Image saved to " + fPath)
        total += 1

    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
