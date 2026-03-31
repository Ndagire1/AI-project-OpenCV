import cv2
import os

DATASETS_DIR = 'datasets'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'


def setup_dataset_folder(folder=DATASETS_DIR):
    """Create dataset folder if it doesn't exist."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def load_face_detector():
    """Load Haar Cascade face detector."""
    return cv2.CascadeClassifier(CASCADE_PATH)


def detect_faces(gray_frame, detector):
    """Detect faces in grayscale frame."""
    return detector.detectMultiScale(gray_frame, 1.3, 5)


def draw_rectangles(frame, faces):
    """Draw bounding boxes around detected faces."""
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


def capture_faces(user_name, num_images=20):
    """Auto capture faces and save to user folder."""
    dataset_dir = setup_dataset_folder()
    user_dir = os.path.join(dataset_dir, user_name)
    
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    
    detector = load_face_detector()
    cap = cv2.VideoCapture(0)
    count = 0

    print(f"Capturing {num_images} images for {user_name}...")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, detector)

        # Guard against trigger in case face detector returns None or empty list
        if faces is None or len(faces) == 0:
            cv2.imshow('Face Capture', frame)
            cv2.waitKey(1)
            continue

        frame = draw_rectangles(frame, faces)

        cv2.imshow('Face Capture', frame)
        cv2.waitKey(1)

        x, y, w, h = faces[0]
        count += 1
        file_path = os.path.join(user_dir, f"{count}.jpg")
        cv2.imwrite(file_path, gray[y:y + h, x:x + w])
        print(f"Captured: {count}/{num_images}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done! {num_images} images saved to {user_dir}")


if __name__ == '__main__':
    user_name = input("Enter username: ")
    capture_faces(user_name)
