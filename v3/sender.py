import argparse
import os
import socket
import time
import xml.etree.ElementTree as ET
from typing import List, Tuple

import cv2

VEHICLE_CLASSES = {"car": 0, "van": 1, "bus": 2, "others": 3}


def parse_windows(value: str) -> List[Tuple[float, float]]:
    if not value:
        return []
    windows = []
    for part in value.split(","):
        start, end = part.split(":")
        start_f = float(start)
        end_f = float(end)
        if end_f <= start_f:
            raise ValueError(f"Invalid outage window '{part}': end must be after start")
        windows.append((start_f, end_f))
    return windows


def active_window(elapsed: float, windows: List[Tuple[float, float]]):
    for start, end in windows:
        if start <= elapsed < end:
            return start, end
    return None


def select_sequences(base_path: str, sequence_start: int, sequence_count: int) -> List[str]:
    sequences = sorted(
        os.path.join(base_path, name)
        for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name))
    )
    selected = sequences[sequence_start : sequence_start + sequence_count]
    if not selected:
        raise RuntimeError(
            f"No sequences selected from {base_path} with start={sequence_start}, "
            f"count={sequence_count}"
        )
    return selected


def iter_images(sequences: List[str]):
    for folder_id, sequence in enumerate(sequences):
        images = sorted(
            os.path.join(sequence, name)
            for name in os.listdir(sequence)
            if name.endswith((".jpg", ".png"))
        )
        for frame_id, image_path in enumerate(images):
            yield folder_id, frame_id, image_path


def iter_crops(sequences: List[str], annotation_path: str):
    for folder_id, sequence in enumerate(sequences):
        sequence_name = os.path.basename(sequence)
        xml_path = os.path.join(annotation_path, f"{sequence_name}.xml")
        if not os.path.exists(xml_path):
            print(f"Missing annotation: {xml_path}", flush=True)
            continue

        root = ET.parse(xml_path).getroot()
        for frame_elem in root.findall("frame"):
            frame_id = int(frame_elem.get("num"))
            image_path = os.path.join(sequence, f"img{frame_id:05d}.jpg")
            if not os.path.exists(image_path):
                continue

            for sample_id, target in enumerate(frame_elem.findall(".//target")):
                box_elem = target.find("box")
                attr_elem = target.find("attribute")
                if box_elem is None or attr_elem is None:
                    continue

                label = VEHICLE_CLASSES.get(attr_elem.get("vehicle_type", "others"), 3)
                box = (
                    float(box_elem.get("left")),
                    float(box_elem.get("top")),
                    float(box_elem.get("width")),
                    float(box_elem.get("height")),
                )
                yield folder_id, frame_id, sample_id, image_path, box, label


def wait_outage_if_needed(start_time: float, windows: List[Tuple[float, float]]) -> None:
    while True:
        elapsed = time.monotonic() - start_time
        window = active_window(elapsed, windows)
        if window is None:
            return
        outage_start, outage_end = window
        remaining = max(outage_end - elapsed, 0.0)
        print(
            f"[outage_start] elapsed={elapsed:.3f}s window={outage_start:.3f}:{outage_end:.3f}",
            flush=True,
        )
        time.sleep(remaining)
        print(
            f"[outage_end] elapsed={time.monotonic() - start_time:.3f}s "
            f"window={outage_start:.3f}:{outage_end:.3f}",
            flush=True,
        )


def run(args) -> None:
    windows = parse_windows(args.outage_windows)
    sequences = select_sequences(args.base_path, args.sequence_start, args.sequence_count)
    frame_interval = 1.0 / args.fps
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0
    start_time = time.monotonic()

    print(
        f"Sender {args.sender_id} streaming {len(sequences)} sequences to "
        f"{args.target_host}:{args.port}",
        flush=True,
    )

    if args.mode == "frames":
        for folder_id, frame_id, image_path in iter_images(sequences):
            wait_outage_if_needed(start_time, windows)
            frame = cv2.imread(image_path)
            if frame is None:
                continue
            frame = cv2.resize(frame, (args.width, args.height))
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            send_packet(
                sock,
                args,
                seq,
                folder_id,
                frame_id,
                0,
                -1,
                buffer.tobytes(),
            )
            seq += 1
            time.sleep(frame_interval)
    else:
        for folder_id, frame_id, sample_id, image_path, box, label in iter_crops(
            sequences, args.annotation_path
        ):
            wait_outage_if_needed(start_time, windows)
            frame = cv2.imread(image_path)
            if frame is None:
                continue

            left, top, width, height = box
            img_h, img_w = frame.shape[:2]
            x1 = max(0, int(left))
            y1 = max(0, int(top))
            x2 = min(img_w, int(left + width))
            y2 = min(img_h, int(top + height))
            crop = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else frame
            crop = cv2.resize(crop, (args.width, args.height))
            ok, buffer = cv2.imencode(".jpg", crop)
            if not ok:
                continue
            send_packet(
                sock,
                args,
                seq,
                folder_id,
                frame_id,
                sample_id,
                label,
                buffer.tobytes(),
            )
            seq += 1
            time.sleep(frame_interval)

    print(f"Sender {args.sender_id} finished.", flush=True)


def send_packet(sock, args, seq, folder_id, frame_id, sample_id, label, data):
    chunks = [
        data[i : i + args.chunk_size]
        for i in range(0, len(data), args.chunk_size)
    ]
    send_ts = time.time()
    for chunk_id, chunk in enumerate(chunks):
        header = (
            f"{args.sender_id}|{seq}|{folder_id}|{frame_id}|{sample_id}|"
            f"{label}|{chunk_id}|{len(chunks)}|{send_ts:.9f}"
        ).encode()
        packet = header + b"||" + chunk
        sock.sendto(packet, (args.target_host, args.port))


def parse_args():
    parser = argparse.ArgumentParser(description="v3 UA-DETRAC UDP sender")
    parser.add_argument("--sender-id", required=True)
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument(
        "--base-path",
        default=os.path.join("..", "DETRAC-Images", "DETRAC-Images"),
    )
    parser.add_argument(
        "--annotation-path",
        default=os.path.join(
            "..",
            "DETRAC-Train-Annotations-XML",
            "DETRAC-Train-Annotations-XML",
        ),
    )
    parser.add_argument("--mode", choices=("crops", "frames"), default="crops")
    parser.add_argument("--sequence-start", type=int, default=0)
    parser.add_argument("--sequence-count", type=int, default=10)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--outage-windows", default="")
    parser.add_argument("--chunk-size", type=int, default=1400)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
