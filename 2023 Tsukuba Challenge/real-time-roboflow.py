import cv2
from roboflow import Roboflow

# Initialize Roboflow client
rf = Roboflow(api_key="rJbzS6OOSy0lHVUofBsc")
project = rf.workspace().project("letter-classification")
model = project.version(10).model

# Open the laptop camera (you can change the argument to 0 or 1 depending on your camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define the zoom factor
    zoom_factor = 1 # Adjust the zoom factor as needed

    # Get the original image size
    height, width = gray.shape

    # Calculate the new size after zooming
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Crop the region of interest (ROI) to maintain the original image size
    start_row = int((new_height - height) / 2)
    start_col = int((new_width - width) / 2)
    end_row = start_row + height
    end_col = start_col + width

    # Crop the ROI from the zoomed grayscale image
    zoomed_gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    zoomed_gray_cropped = zoomed_gray[start_row:end_row, start_col:end_col]

    # Save the frame to a temporary file
    cv2.imwrite("temp_frame.jpg", zoomed_gray_cropped)

    # Send the frame to Roboflow model for inference
    predictions_data = model.predict("temp_frame.jpg").json()

    # Extract the predicted classes and their confidence values
    prediction_item = predictions_data['predictions'][0]
    predicted_classes = prediction_item['predictions']

    # Find the letter with the highest confidence
    max_confidence_class = max(predicted_classes, key=lambda k: predicted_classes[k]['confidence'])

    # Extract the letter and its confidence value
    max_confidence_letter = max_confidence_class
    max_confidence_value = predicted_classes[max_confidence_class]['confidence']

    # Display the result
    result_text = f"The letter is {max_confidence_letter}, confidence: {max_confidence_value:.2f}"
    cv2.putText(zoomed_gray_cropped, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Real-time Classification', zoomed_gray_cropped)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
