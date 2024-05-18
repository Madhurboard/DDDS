import cv2
import torch
import torchvision.transforms as transforms

# Load your pretrained PyTorch model
# Load the checkpoint
checkpoint = torch.load('pretrainedModel.pth')

# Extract the model from the checkpoint
model = checkpoint['model']

# Set the model to evaluation mode
model.eval()

# Define a function for drowsiness detection
def detect_drowsiness(frame):
    # Preprocess the frame (resize, normalize, etc.)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frame = preprocess(frame)
    frame = frame.unsqueeze(0)  # Add batch dimension
    
    # Perform inference
    with torch.no_grad():
        output = model(frame)
    
    # Analyze model output to detect drowsiness
    # Assuming output is a probability distribution over classes
    # You might need to adjust this based on how your model was trained
    is_drowsy = output[0][1] > 0.5  # Assuming index 1 corresponds to drowsiness class
    confidence = output[0][1] if is_drowsy else 1.0 - output[0][1]
    
    return is_drowsy, confidence

# Access the phone's camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Detect drowsiness in the frame
    is_drowsy, confidence = detect_drowsiness(frame)
    
    # Display the frame with drowsiness status
    if is_drowsy:
        cv2.putText(frame, "Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
