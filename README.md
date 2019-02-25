# OpenCV-Face-Recognition-DL

### A facial recognition software project using Python and OpenCV(-python)

For the purposes of this project I will be using the following tutorial:
https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/

### The following python modules are required:
opencv-python
imutils
scikit-learn
numpy


### Process:
1. CD to dataset.
2. Create the dataset for a person
    python create_dataset --cascade [CASCADE NAME HERE] (e.g. haarcascade_frontalface_default.xml) --output [name of folder, e.g. Jack]
3. Extract the embeddings from the new dataset:
    python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector detection_model --embedding-model openface_nn4.small2.v1.t7
4. Train the model with the new embeddings:
    python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle
5. Run the recognizer
    python recognize.py --detector detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle