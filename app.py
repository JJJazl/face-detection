import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace

# Set the path to the CSV file containing the embeddings
embeddings_file = "./embeddings.csv"

# Load the embeddings CSV file into a pandas DataFrame
embeddings_df = pd.read_csv(embeddings_file)

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

# Set capture resolution to 480p
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set font
font = cv2.FONT_HERSHEY_SIMPLEX

def recognize(face, model_name, column):
    # Use deepface to extract the face embedding
    embedding = DeepFace.represent(face, model_name=model_name, enforce_detection=False)

    # Compute the Euclidean distances between the embedding and the embeddings in the CSV file
    distances = embeddings_df[column].apply(lambda x: np.linalg.norm(np.squeeze(embedding[0]["embedding"]) - np.squeeze(np.array(eval(x))[0]["embedding"])))

    # Find the index of the embedding with the smallest distance
    min_distance_index = np.argmin(distances)

    # Get the name and similarity score of the person corresponding to the embedding with the smallest distance
    name = embeddings_df.iloc[min_distance_index]["name"]
    score = 1 / (1 + distances[min_distance_index])
    
    return name, score

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through each face detected
    for (x, y, w, h) in faces:
        # Extract the face from the frame
        face = frame[y:y+h, x:x+w]

        # Preprocess the face for face recognition
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))

        # result = recognize(face, "Facenet", "embedding_facenet")
        result = recognize(face, "VGG-Face", "embedding_vgg")
        
        # Draw a rectangle around the detected face and write the name and match percentage above the rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{result[0]} ({result[1]*100:.2f}%)", (x, y-10), font, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera capture and close the window
cap.release()
cv2.destroyAllWindows()