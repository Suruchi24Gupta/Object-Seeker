import cv2

# Define the preprocess_image function
def preprocess_image(image):
    # Resize the image to the desired dimensions (e.g., 224x224)
    resized_image = cv2.resize(image, (224, 224))
    
    # Normalize pixel values to be in the range [0, 1]
    normalized_image = resized_image / 255.0
    
    return normalized_image

# Initialize the video capture
cap = cv2.VideoCapture(0)  # Use the default camera (you can specify a different camera)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    processed_frame = preprocess_image(frame)
    
    # Apply the model for object shape detection
    predictions = model.predict(np.expand_dims(processed_frame, axis=0))
    shape_class = np.argmax(predictions)
    
    # Draw bounding boxes or labels based on the detected shape
    if shape_class == 0:  # Example: 0 for circles
        cv2.circle(frame, (x, y), radius, color, thickness)
    elif shape_class == 1:  # Example: 1 for squares
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, thickness)
    # Add similar code for other shapes
    
    # Display the frame with detected shapes
    cv2.imshow('Object Shape Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
