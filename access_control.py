import cv2
import os
import csv
from datetime import datetime

DATASETS_DIR = 'datasets'
MODEL_PATH = 'trainer.yml'
LOG_FILE = 'entry_log.csv'
CONFIDENCE_THRESHOLD = 100
DISPLAY_DURATION = 5


def load_recognizer(model_path=MODEL_PATH):
    """Load trained face recognizer model."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    return recognizer


def load_face_detector():
    """Load Haar Cascade face detector."""
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def get_user_id_map():
    """Build deterministic ID mapping from sorted usernames."""
    user_map = {}
    user_id_to_name = {}
    
    if os.path.exists(DATASETS_DIR):
        user_folders = sorted([d for d in os.listdir(DATASETS_DIR) if os.path.isdir(os.path.join(DATASETS_DIR, d))])
        for idx, user_name in enumerate(user_folders, 1):
            user_map[user_name] = idx
            user_id_to_name[idx] = user_name
    
    return user_id_to_name


def init_log_file():
    """Initialize CSV log file if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Date', 'Time'])


def log_access(user_name):
    """Log user access to CSV file."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([user_name, date_str, time_str])
    
    print(f">>> Access Granted: {user_name} at {time_str}")


def recognize_face(face_roi, recognizer, user_map):
    """Recognize face and return username and confidence."""
    user_id, confidence = recognizer.predict(face_roi)
    user_name = user_map.get(user_id, "Unknown")
    return user_name, confidence


def draw_face_box(frame, x, y, w, h, user_name, confidence):
    """Draw bounding box around detected face."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    text = f"{user_name} ({confidence:.0f})"
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def draw_status(frame, status_text, is_granted=False):
    """Draw status text on frame."""
    color = (0, 255, 0) if is_granted else (0, 0, 255)
    cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)


def access_control():
    """Main access control loop."""
    init_log_file()
    recognizer = load_recognizer()
    detector = load_face_detector()
    user_map = get_user_id_map()
    
    if not user_map:
        print("[ERROR] No trained users found! Train the model first.")
        return
    
    cap = cv2.VideoCapture(0)
    last_recognized = {"name": "", "time": 0}
    
    print("[INFO] Access Control System Active")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        
        current_time = datetime.now().timestamp()
        is_authorized = False
        current_user = None
        locked_due_unknown = False

        for (x, y, w, h) in faces:
            user_name, confidence = recognize_face(gray[y:y + h, x:x + w], recognizer, user_map)

            if confidence < CONFIDENCE_THRESHOLD and user_name != "Unknown":
                if current_user is None:
                    current_user = user_name
                elif current_user != user_name:
                    # If multiple different recognized users appear, keep system locked until stable
                    locked_due_unknown = True

                draw_face_box(frame, x, y, w, h, user_name, confidence)
            else:
                # any unknown face should lock the system
                locked_due_unknown = True
                break

        if len(faces) > 0 and not locked_due_unknown and current_user:
            is_authorized = True
            if current_time - last_recognized["time"] > 30 or last_recognized["name"] != current_user:
                log_access(current_user)
                last_recognized = {"name": current_user, "time": current_time}

        # Display status
        if is_authorized and current_user:
            draw_status(frame, f"ACCESS GRANTED: {current_user}", True)
        else:
            draw_status(frame, "SYSTEM LOCKED", False)

        cv2.imshow('Access Control System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] System closed.")


if __name__ == '__main__':
    access_control()

