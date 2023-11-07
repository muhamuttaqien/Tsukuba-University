import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

# Load your PyTorch model
# Replace 'your_model.pth' with the actual path to your saved PyTorch model
model = resnet18(pretrained=True)  # assuming you have a resnet18 architecture
# loaded_model.load_state_dict(torch.load('your_model.pth'))
model.eval()

# Open the laptop camera (you can change the argument to 0 or 1 depending on your camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the image to grayscale

    # Define the zoom factor
    zoom_factor = 1 # Adjust the zoom factor as needed

    # Get the original image size
    height, width, _ = frame.shape

    # Calculate the new size after zooming
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Crop the region of interest (ROI) to maintain the original image size
    start_row = int((new_height - height) / 2)
    start_col = int((new_width - width) / 2)
    end_row = start_row + height
    end_col = start_col + width

    # Crop the ROI from the zoomed grayscale image
    zoomed_rgb = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    zoomed_rgb_cropped = zoomed_rgb[start_row:end_row, start_col:end_col]
    
    # Save the frame to a temporary file
    cv2.imwrite("temp_frame.jpg", zoomed_rgb_cropped)

    # Load the saved image for inference
    pil_image = Image.open("temp_frame.jpg")

    # Apply transformations to prepare the image for the model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Assuming ResNet18 input size
        transforms.ToTensor(),
    ])
    input_tensor = transform(pil_image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted class index
    _, predicted_class = torch.max(output, 1)

    # Display the result
    result_text = f"The predicted class is {predicted_class.item()}"
    cv2.putText(zoomed_rgb_cropped, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Real-time Classification', zoomed_rgb_cropped)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()