# receiver.py
import socket
import time
import csv

PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))

print("Receiver started...")

prev_time = None

with open("interarrival_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "interarrival", "sender_ip"])

    while True:
        data, addr = sock.recvfrom(2048)

        t = time.perf_counter()

        if prev_time is not None:
            delay = t - prev_time
            writer.writerow([t, delay, addr[0]])
            print(f"Delay: {delay:.6f}s")

        prev_time = t