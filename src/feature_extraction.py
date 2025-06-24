import mediapipe as mp
import cv2


class MediaPipe():
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=1)
        self.mp_drawing = mp.solutions.drawing_utils

    def feature_extraction(self, img_path):

        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            return None
        # else:
        #     print(f"extracting features from image: {img_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks is None:
            print(f"[INFO:] No hand landmarks found in {img_path}")
            return None  # Skip image

            # Try/except block to be extra safe
        try:
            hand_landmarks = results.multi_hand_landmarks[0]  # First hand
            coords = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            return coords
        except Exception as e:
            print(f"[ERROR] Failed to extract landmarks: {e}")
            return None



    def extract_from_frame(self,frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        coords = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            coords = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return coords, frame
#saving the landmark images:
  # with self.mp_hands.Hands(static_image_mode=True) as hands:
        #     results = hands.process(image_rgb)
        #     if results.multi_hand_landmarks:
        #         for hand_landmarks in results.multi_hand_landmarks:
        #             # Draw landmarks on the image
        #             self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        # cv2.imwrite('output_with_landmarks.jpg', image)