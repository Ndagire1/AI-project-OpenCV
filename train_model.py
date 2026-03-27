import cv2
import numpy as np
import os

DATASETS_DIR = 'datasets'
MODEL_PATH = 'trainer.yml'


def load_face_detector():
    """Load Haar Cascade face detector."""
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    return detector


def get_user_id_map():
    """Build deterministic ID mapping from sorted usernames."""
    user_map = {}

    if os.path.exists(DATASETS_DIR):
        user_folders = sorted([
            d for d in os.listdir(DATASETS_DIR)
            if os.path.isdir(os.path.join(DATASETS_DIR, d))
        ])
        for idx, user_name in enumerate(user_folders, 1):
            user_map[user_name] = idx

    return user_map


def load_faces_from_user(user_path, detector, user_id):
    """Load all face samples from a single user folder."""
    faces = []
    ids = []

    image_paths = [
        os.path.join(user_path, f)
        for f in os.listdir(user_path)
        if f.endswith('.jpg')
    ]

    for image_path in image_paths:
        img = cv2.imread(image_path)

        if img is None:
            print(f"[WARNING] Could not read {image_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 🔥 Improved detection
        detected_faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
        )

        print(f"[DEBUG] {image_path} -> {len(detected_faces)} face(s) found")

        for (x, y, w, h) in detected_faces:
            face = gray[y:y + h, x:x + w]

            # 🔥 Resize for consistency
            face = cv2.resize(face, (200, 200))

            faces.append(face)
            ids.append(user_id)

    return faces, ids


def load_all_faces(detector):
    """Load all faces from all user folders."""
    all_faces = []
    all_ids = []

    if not os.path.exists(DATASETS_DIR):
        print(f"[ERROR] {DATASETS_DIR} folder not found!")
        return all_faces, all_ids

    user_id_map = get_user_id_map()

    for user_name, user_id in user_id_map.items():
        user_path = os.path.join(DATASETS_DIR, user_name)

        faces, ids = load_faces_from_user(user_path, detector, user_id)

        all_faces.extend(faces)
        all_ids.extend(ids)

        print(f"[INFO] Loaded {len(faces)} faces from {user_name} (ID: {user_id})")

    return all_faces, all_ids


def train_model(faces, ids):
    """Train LBPH face recognizer."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    return recognizer


def save_model(recognizer, model_path=MODEL_PATH):
    """Save trained model."""
    recognizer.write(model_path)
    print(f"[INFO] Model saved to {model_path}")


if __name__ == '__main__':
    print("[INFO] Loading face detector...")
    detector = load_face_detector()

    print("[INFO] Building user ID map...")
    user_id_map = get_user_id_map()
    print(f"[INFO] User mapping: {user_id_map}")

    print("[INFO] Loading faces from datasets...")
    faces, ids = load_all_faces(detector)

    if len(ids) > 0:
        print(f"[INFO] Training on {len(faces)} faces...")
        recognizer = train_model(faces, ids)
        save_model(recognizer)
        print(f"[INFO] {len(np.unique(ids))} user(s) trained successfully!")
    else:
        print("[ERROR] No faces found in datasets folder!")