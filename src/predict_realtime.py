import cv2
import torch
import numpy as np
from lstm import ASLLSTM
from feature_extraction import MediaPipe

index_to_label = [chr(i) for i in range(ord('A'), ord('Z') + 1)]+ ['del','nothing','space']

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = ASLLSTM()
model.load_state_dict(torch.load(r'C:\Users\ypanw\PycharmProjects\PythonProject\checkpoints\asl_lstm.pth'))
model.eval()
model.to(device)

extractor = MediaPipe()

cap = cv2.VideoCapture(0)
print('starting webcam')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    coords, frame_with_landmarks = extractor.extract_from_frame(frame)

    # Only predict if hand landmarks found
    if coords:
        coords_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(device)  # shape: (1, 21, 3)

        with torch.no_grad():
            logits = model(coords_tensor)
            pred = torch.argmax(logits, dim=1).item()

        # Map class index to letter (e.g., 0 = 'A', 1 = 'B', ..., 25 = 'Z', 26 = 'del', 27 = 'space', etc.)
        label = chr(ord('A') + pred) if pred < 26 else f"Class {pred}"
        cv2.putText(frame_with_landmarks, f"Prediction: {label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        cv2.putText(frame_with_landmarks, "No hand detected", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ASL Real-Time", frame_with_landmarks)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()