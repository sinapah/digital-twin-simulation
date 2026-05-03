# sender_detrac.py
import socket
import os
import cv2
import time

TARGET_IP = "10.181.160.88"
PORT = 5005

BASE_PATH = "../DETRAC-Images/DETRAC-Images"
NUM_FOLDERS = 10
FPS = 25
FRAME_INTERVAL = 1.0 / FPS

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Get first 10 folders (videos)
folders = sorted([
    os.path.join(BASE_PATH, d)
    for d in os.listdir(BASE_PATH)
    if os.path.isdir(os.path.join(BASE_PATH, d))
])[:NUM_FOLDERS]

print(f"Using {len(folders)} DETRAC sequences")

SEQ = 0

for folder_id, folder in enumerate(folders):
    print(f"\nStreaming folder {folder_id}: {folder}")

    # Get images sorted (important for frame order)
    images = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith((".jpg", ".png"))
    ])

    for frame_id, img_path in enumerate(images):
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        # Resize to reduce packet size
        frame = cv2.resize(frame, (320, 240))

        # Encode as JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        data = buffer.tobytes()

        # Split into UDP-safe chunks
        CHUNK_SIZE = 1400
        chunks = [data[i:i+CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)]

        for chunk_id, chunk in enumerate(chunks):
            header = f"{SEQ}|{folder_id}|{frame_id}|{chunk_id}|{len(chunks)}".encode()
            packet = header + b"||" + chunk

            sock.sendto(packet, (TARGET_IP, PORT))

        SEQ += 1

        # simulate camera frame rate
        time.sleep(FRAME_INTERVAL)

print("Done streaming all folders.")