import cv2
import os
import time


def capture_video(path):
    save_dir = "output_frames"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture(path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        if frame_count % 50 == 0:
            timestamp = int(time.time())
            frame_filename = f"{save_dir}/frame_{timestamp}.jpg"
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def find_file_paths(directory):
    file_paths = []

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    return file_paths


directory_path = ""
all_file_paths = find_file_paths(directory_path)

for path in all_file_paths:
    capture_video(path)
