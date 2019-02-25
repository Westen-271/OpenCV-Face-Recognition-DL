from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# Parse arguments as per.
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--embeddings", required = True,
                    help = "Path to the serialized DB of face embeddings (.pickle)")

parser.add_argument("-r", "--recognizer", required = True,
                    help = "Path to the output model trained to recognize faces.")

parser.add_argument("-l", "--le", required = True,
                    help = "Path to the output label encoder.")

args = vars(parser.parse_args())

# Load face embeddings from the pickle file... this means the extract_embeddings needs to be run first!
data = pickle.loads(open(args["embeddings"], "rb").read())

# Encode!
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# Time to train the SVM model...oh no
recognizer = SVC(C = 1.0, kernel = "linear", probability = True)
recognizer.fit(data["embeddings"], labels)

# Write the face recognition data to disk.
file = open(args["recognizer"], "wb")
file.write(pickle.dumps(recognizer))
file.close()

# Write label encoder to disk.
file = open(args["le"], "wb")
file.write(pickle.dumps(le))
file.close()

print("Model trained!")
