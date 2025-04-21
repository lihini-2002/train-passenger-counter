# mediapipe_counter.py

import cv2
import mediapipe as mp

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def detect_people(image_path):
    """
    Detects whether there is at least one person in the given image.

    Args:
        image_path (str): Path to the image to check.

    Returns:
        bool: True if a person is detected, False otherwise.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or unreadable.")
        return False

    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Return True if pose landmarks (body points) are detected
    return results.pose_landmarks is not None

if __name__ == "__main__":
    image_path = "test.jpg"  # Put a test image in backend/ to try this
    found = detect_people(image_path)
    print("Person Detected ✅" if found else "No Person Detected")
