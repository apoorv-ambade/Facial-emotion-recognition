import cv2
import numpy as np
import dlib
from keras.models import load_model

# Load the pre-trained model
model = load_model('best_model.h5')

# Define the emotions (assuming your model predicts emotions)
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Neutral', 6: 'Surprise'}

# Initialize Dlib's face detector
detector = dlib.get_frontal_face_detector()

# Load and preprocess the input image
def preprocess_input(img):
    # Check if the image has 3 channels (BGR)
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Convert BGR to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        # Image is already grayscale
        pass
    else:
        raise ValueError("Invalid image format. Expected BGR or grayscale image.")

    img = cv2.resize(img, (48, 48))
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    img = img.astype('float32')  # Convert to float32
    img /= 255.0  # Normalize
    return img

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Dlib
    faces = detector(gray)

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        # Extract the face region
        face = gray[y:y + h, x:x + w]

        # Preprocess the face for emotion prediction
        processed_face = preprocess_input(face)

        # Predict the emotion
        prediction = model.predict(processed_face)
        emotion_label = emotions[np.argmax(prediction)]

        # Draw a rectangle around the face and display the predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the live feed with emotion predictions
    cv2.imshow('Emotion Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
