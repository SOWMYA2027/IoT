import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize camera
cam = cv2.VideoCapture(0)

# Initialize face mesh detector
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Initialize smoothing and blink parameters
smooth_x, smooth_y = 0, 0
smoothing_factor = 0.15
blink_threshold = 0.21  # EAR threshold for blink detection
blink_consec_frames = 5  # Consecutive frames required to register a blink
blink_counter = 0

# Indices for the left eye (from MediaPipe face mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(landmarks, eye_indices):
    A = np.linalg.norm(np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) -
                       np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y]))
    B = np.linalg.norm(np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y]) -
                       np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]))
    C = np.linalg.norm(np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y]) -
                       np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y]))
    ear = (A + B) / (2.0 * C)
    return ear

# Main loop
while True:
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    frame_h, frame_w, _ = frame.shape

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Use iris landmark (468) to track eye gaze
        iris = landmarks[468]
        x, y = int(iris.x * frame_w), int(iris.y * frame_h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Map iris position to screen coordinates
        target_x = screen_w * iris.x
        target_y = screen_h * iris.y

        # Smooth the mouse movement
        smooth_x = smooth_x + (target_x - smooth_x) * smoothing_factor
        smooth_y = smooth_y + (target_y - smooth_y) * smoothing_factor
        pyautogui.moveTo(smooth_x, smooth_y)

        # Blink detection using EAR
        ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        cv2.putText(frame, f'EAR: {ear:.2f}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)

        if ear < blink_threshold:
            blink_counter += 1
        else:
            if blink_counter >= blink_consec_frames:
                pyautogui.click()
                time.sleep(1)  # Prevent rapid clicks
            blink_counter = 0

    # Show the frame
    cv2.imshow("Eye Controlled Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
