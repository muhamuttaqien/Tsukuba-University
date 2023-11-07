import cv2
import torch
import torchvision.transforms as transforms

from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('./yolov8x.pt')

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
    predictions_data = model.predict(source="temp_frame.jpg", conf=0.25)

    print(predictions_data)
    
    # Extract the predicted classes and their confidence values
    for prediction_item in predictions_data['labels']:
        class_name = prediction_item['label']
        confidence = prediction_item['confidence']

        # Display the result
        result_text = f"The letter is {class_name}, confidence: {confidence:.2f}"
        cv2.putText(zoomed_gray_cropped, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Real-time Classification', zoomed_gray_cropped)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()