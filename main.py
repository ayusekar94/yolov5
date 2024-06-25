import torch
import cv2
import numpy as np
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath


# Load custom-trained YOLOv5 model
model = torch.hub.load('.','custom',path='best.pt',source='local')
model.conf = 0.5  # Set confidence threshold to 0.5 (adjust as needed)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Process detection results (e.g., draw bounding boxes)
    for det in results.pred[0]:
        if det[-1] == 0:  # Class index for helmet (adjust if needed)
            x1, y1, x2, y2, conf = det[:5]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Display frame
    cv2.imshow('Helmet Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# def detect_helmet(image_path):
#     # Load image
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Perform inference
#     results = model(img)

#     # Process detection results
#     for obj in results.xyxy[0]:
#         class_index = int(obj[-1])
#         class_name = model.names[class_index]
#         confidence = obj[2]

#         if class_name == 'helmet' and confidence > 0.5:
#             print("Detected helmet with confidence:", confidence)
#         elif class_name == 'no_helmet' and confidence > 0.5:
#             print("Warning: No helmet detected!")
#             # Trigger the buzzer (Windows only)
#             winsound.Beep(1000, 1000)  # Adjust frequency and duration as needed

# # Example usage
# image_path = 'data\images\helm2.jpg'
# detect_helmet(image_path)
