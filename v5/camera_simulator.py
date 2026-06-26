import os
import csv
import random
import time
from typing import List, Tuple, Dict, Optional

from models.simple_cnn import SimpleCNN
from utils.detrac_loader import DETRACLoader, DETRACDataset


INTERARRIVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'v2')

KDE_PATH = os.path.join(INTERARRIVAL_DIR, 'synthetic_interarrival_kde.csv')
WGAN_PATH = os.path.join(INTERARRIVAL_DIR, 'synthetic_interarrival_wgan.csv')

IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'DETRAC-Images', 'DETRAC-Images')
ANNOTATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'DETRAC-Train-Annotations-XML', 'DETRAC-Train-Annotations-XML')

NUM_EDGES = 3
VIDEOS_PER_EDGE = 7
HISTORICAL_FOLDERS = 3
TOTAL_VIDEOS = NUM_EDGES * VIDEOS_PER_EDGE + HISTORICAL_FOLDERS


def load_delays(filepath: str) -> List[float]:
    delays = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                try:
                    delays.append(float(row[0]))
                except ValueError:
                    pass
    return delays


class CameraSimulator:
    def __init__(self, delay_model: str = 'kde',
                 max_frames_per_video: int = 50,
                 image_dir: str = IMAGE_DIR,
                 annotation_dir: str = ANNOTATION_DIR):
        self.delay_model = delay_model.lower()
        self.max_frames_per_video = max_frames_per_video
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir

        if self.delay_model == 'kde':
            self.delays = load_delays(KDE_PATH)
        elif self.delay_model == 'wgan':
            self.delays = load_delays(WGAN_PATH)
        else:
            raise ValueError(f"Unknown delay model: {delay_model}")

        self.detrac_loader = DETRACLoader(self.image_dir, self.annotation_dir)
        all_folders = self.detrac_loader.get_video_folders()

        annotated = [
            f for f in all_folders
            if os.path.exists(os.path.join(self.annotation_dir, f"{f}.xml"))
        ]

        # Sort by class diversity so diverse folders are spread across edges.
        # Interleave: pick one from each diversity tier per edge rather than
        # assigning the top-7 to edge 0, next-7 to edge 1, etc.
        import xml.etree.ElementTree as ET

        def diversity_score(folder):
            xml_path = os.path.join(self.annotation_dir, f"{folder}.xml")
            try:
                root = ET.parse(xml_path).getroot()
                classes = set()
                for frame in root.findall('.//frame'):
                    if int(frame.get('num')) > self.max_frames_per_video:
                        break
                    for t in frame.findall('.//target'):
                        classes.add(t.find('attribute').get('vehicle_type'))
                return len(classes)
            except Exception:
                return 0

        annotated.sort(key=diversity_score, reverse=True)

        # Interleave: slot 0 → edge 0, slot 1 → edge 1, slot 2 → edge 2,
        # slot 3 → edge 0, slot 4 → edge 1, ... etc.
        interleaved = [[] for _ in range(NUM_EDGES + 1)]  # +1 for historical pool
        for idx, folder in enumerate(annotated):
            bucket = idx % (NUM_EDGES + 1)
            interleaved[bucket].append(folder)

        # Each edge gets VIDEOS_PER_EDGE from its interleaved bucket
        # topped up from the historical pool if needed
        pool = interleaved[NUM_EDGES]  # leftover bucket used for historical + top-up

        if len(annotated) < TOTAL_VIDEOS:
            print(f"WARNING: Only {len(annotated)} annotated UA-DETRAC folders found, "
                  f"need {TOTAL_VIDEOS}")
            self.edge_folders = []
            self.historical_folders = []
            return

        self.edge_folders = []
        for i in range(NUM_EDGES):
            bucket = interleaved[i][:VIDEOS_PER_EDGE]
            # top up from pool if bucket is short
            while len(bucket) < VIDEOS_PER_EDGE and pool:
                bucket.append(pool.pop(0))
            self.edge_folders.append(bucket)

        self.historical_folders = pool[:HISTORICAL_FOLDERS]

        print(f"CameraSimulator: {delay_model.upper()} delay model")
        for i in range(NUM_EDGES):
            print(f"  Edge {i} videos: {self.edge_folders[i]}")
            print(f"  Edge {i} historical: {self.historical_folders[i]}")

    def schedule_outages(self, num_rounds: int = 100,
                          baseline: bool = False) -> List[Tuple[int, int]]:
        if baseline:
            print("  Baseline mode: no outages scheduled")
            return [(-1, -1)] * NUM_EDGES
        if num_rounds < 30:
            return [(0, 0), (0, 0), (0, 0)]
        gap = (num_rounds - 20) // NUM_EDGES
        outages = []
        for edge_id in range(NUM_EDGES):
            center = 10 + edge_id * gap
            duration = random.randint(5, min(15, gap - 5))
            start = center - duration // 2
            end = start + duration
            outages.append((start, end))
        for i, (s, e) in enumerate(outages):
            print(f"  Edge {i} outage: rounds {s}-{e} ({e - s} rounds)")
        return outages

    def sample_delay(self) -> float:
        return random.choice(self.delays)

    def get_data_for_edge(self, edge_id: int, round_num: int,
                          outage_periods: List[Tuple[int, int]],
                          is_outage: Optional[bool] = None) -> Tuple[List, bool]:
        if is_outage is None:
            is_outage = False
            if edge_id < len(outage_periods):
                start, end = outage_periods[edge_id]
                is_outage = start <= round_num <= end

        if is_outage:
            folders = [self.historical_folders[edge_id]]
            source_type = "historical"
        else:
            folders = self.edge_folders[edge_id]
            source_type = "normal"

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset = DETRACDataset(
            loader=self.detrac_loader,
            video_folders=folders,
            max_frames=self.max_frames_per_video,
            transform=transform
        )

        samples = []
        for i in range(len(dataset)):
            img, label = dataset[i]
            samples.append((img, label))

        print(f"  [CamSim] Edge {edge_id} round {round_num}: {source_type}, "
              f"{len(samples)} crops from {folders}")

        return samples, is_outage