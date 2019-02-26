# OpenCV-Face-Recognition-DL

### A facial recognition software project using Python and OpenCV(-python)

For the purposes of this project I will be using the following tutorial:
https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/

### Process:
1. Change the value in 'video_capture_device.txt' to the number that corresponds to the capture device you wish to use.<br>
<i>For example, main webcam or first connected device is 0, front-facing camera is 1...</i>
2. CD to dataset.
3. Create the dataset for a person:<br>
    python create_dataset --cascade [CASCADE NAME HERE] (e.g. haarcascade_frontalface_default.xml) --output [name of folder, e.g. Jack]
4. Exit dataset - CD to root directory of the project.
5. Extract the embeddings from the new dataset:<br>
    python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector detection_model --embedding-model openface_nn4.small2.v1.t7
6. Train the model with the new embeddings:<br>
    python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle
7. Run the recognizer: <br>
    python recognize.py --detector detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle
