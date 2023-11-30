import cv2
import dlib
import numpy as np
from keras.models import load_model

# Load the pre-trained facial emotion recognition model
model = load_model('best_model.h5')

# Define the emotions labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral', 'Surprise']

# Load the image
image_path = 'Demo/test.JPG'  # Replace with your image path
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize Dlib's face detector
detector = dlib.get_frontal_face_detector()

# Detect faces in the image
faces = detector(gray_image)

# Process each detected face
for face in faces:
    # Get the coordinates of the face rectangle
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())

    # Check if the detected region is a valid face
    if w > 30 and h > 30:
        # Draw a rectangle around the detected face
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) for emotion prediction
        roi = gray_image[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = np.expand_dims(roi, axis=0)
        roi = roi / 255.0  # Normalize the image

        # Predict the emotion
        predicted_emotion = model.predict(roi)
        emotion_index = np.argmax(predicted_emotion)
        emotion = emotion_labels[emotion_index]

        # Display the predicted emotion on the image
        cv2.putText(image, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Display the image with faces and predicted emotions
cv2.imshow('Facial Emotion Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
