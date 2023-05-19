import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import tensorflow as tf

# Set the path to the CSV file containing the names and image paths
database_file = "./database.csv"

# Set the path to the CSV file to save the embeddings to
embeddings_file = "./embeddings.csv"

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create a DataFrame to store the names and embeddings of the faces
df = pd.DataFrame(columns=["name", "embedding_facenet", "embedding_vgg"])

# Load the CSV file containing the names and image paths
database = pd.read_csv(database_file)

# Loop through each image in the database
for i in range(len(database)):
    # Load the image
    image_path = database.iloc[i]["image_path"]
    image = cv2.imread(image_path)

    # Detect faces in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through each face detected
    for (x, y, w, h) in faces:
        # Extract the face from the image
        face = image[y:y+h, x:x+w]

        # Preprocess the face for face recognition
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))

        # Use deepface to extract the face embedding
        embedding_facenet = DeepFace.represent(face, model_name="Facenet", enforce_detection=False)
        embedding_vgg = DeepFace.represent(face, model_name="VGG-Face", enforce_detection=False)

        # Get the name of the person in the image
        name = database.iloc[i]["name"]

        # Add the name and embedding to the DataFrame
        df = df.append({"name": name, "embedding_facenet": embedding_facenet, "embedding_vgg": embedding_vgg}, ignore_index=True)

# Save the DataFrame to the CSV file
df.to_csv(embeddings_file, index=False)