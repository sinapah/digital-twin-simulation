# =========================================================
# V4 Utilities - DETRAC Data Loading
# =========================================================
# Provides utilities for loading UA-DETRAC dataset
# =========================================================

import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VehicleDetection:
    """Represents a vehicle detection in an image"""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    vehicle_type: str
    track_id: int


class DETRACLoader:
    """Loader for UA-DETRAC dataset"""
    
    # Vehicle class mapping
    CLASS_MAP = {
        'car': 0,
        'van': 1,
        'bus': 2,
        'others': 3
    }
    
    def __init__(self, image_dir: str, annotation_dir: str):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.video_folders = []
        self._find_videos()
    
    def _find_videos(self):
        """Find all video folders"""
        if os.path.exists(self.image_dir):
            self.video_folders = sorted([
                d for d in os.listdir(self.image_dir)
                if os.path.isdir(os.path.join(self.image_dir, d))
            ])
    
    def get_video_folders(self) -> List[str]:
        """Get list of video folder names"""
        return self.video_folders
    
    def get_video_path(self, folder: str) -> str:
        """Get full path to video folder"""
        return os.path.join(self.image_dir, folder)
    
    def get_annotation_path(self, folder: str) -> str:
        """Get full path to annotation file"""
        xml_file = f"gt_{folder}.xml"
        return os.path.join(self.annotation_dir, folder, xml_file)
    
    def load_annotations(self, folder: str) -> Dict[int, List[VehicleDetection]]:
        """Load annotations for a video folder"""
        ann_path = self.get_annotation_path(folder)
        
        if not os.path.exists(ann_path):
            return {}
        
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        detections = {}
        
        for frame in root.findall('.//frame'):
            frame_num = int(frame.get('num'))
            detections[frame_num] = []
            
            for target in frame.findall('.//target'):
                bbox_elem = target.find('box')
                if bbox_elem is not None:
                    x = int(float(bbox_elem.get('left')))
                    y = int(float(bbox_elem.get('top')))
                    w = int(float(bbox_elem.get('width')))
                    h = int(float(bbox_elem.get('height')))
                    
                    vehicle_type = target.find('attribute').get('vehicle_type')
                    track_id = int(target.get('id'))
                    
                    det = VehicleDetection(
                        bbox=(x, y, w, h),
                        vehicle_type=vehicle_type,
                        track_id=track_id
                    )
                    detections[frame_num].append(det)
        
        return detections
    
    def load_frame(self, folder: str, frame_num: int) -> Optional[np.ndarray]:
        """Load a specific frame"""
        img_path = os.path.join(
            self.get_video_path(folder),
            f"img_{frame_num:05d}.jpg"
        )
        
        if not os.path.exists(img_path):
            return None
        
        img = Image.open(img_path).convert('RGB')
        return np.array(img)
    
    def extract_crops(self, frame: np.ndarray, detections: List[VehicleDetection],
                     target_size: int = 64) -> List[np.ndarray]:
        """Extract vehicle crops from frame"""
        crops = []
        
        for det in detections:
            x, y, w, h = det.bbox
            
            # Add margin
            margin = int(max(w, h) * 0.2)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            crop = frame[y1:y2, x1:x2]
            
            # Resize to target size
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                crop = np.array(Image.fromarray(crop).resize((target_size, target_size)))
                crops.append(crop)
        
        return crops


class DETRACDataset:
    """PyTorch Dataset for UA-DETRAC"""
    
    def __init__(self, loader: DETRACLoader, video_folders: List[str],
                 max_frames: int = 50, transform=None):
        self.loader = loader
        self.video_folders = video_folders
        self.max_frames = max_frames
        self.transform = transform
        self.samples = []
        
        self._build_samples()
    
    def _build_samples(self):
        """Build list of samples from all videos"""
        for folder in self.video_folders:
            annotations = self.loader.load_annotations(folder)
            
            for frame_num, detections in annotations.items():
                if frame_num > self.max_frames:
                    break
                
                frame = self.loader.load_frame(folder, frame_num)
                if frame is None:
                    continue
                
                crops = self.loader.extract_crops(frame, detections)
                
                for crop in crops:
                    # Get class label
                    if detections:
                        class_label = DETRACLoader.CLASS_MAP.get(
                            detections[0].vehicle_type, 3
                        )
                    else:
                        class_label = 3  # others
                    
                    self.samples.append({
                        'image': crop,
                        'label': class_label,
                        'folder': folder,
                        'frame': frame_num
                    })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = sample['image']
        label = sample['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_dataloader(image_dir: str, annotation_dir: str,
                     video_folders: List[str], batch_size: int = 64,
                     shuffle: bool = True, max_frames: int = 50):
    """Create a PyTorch DataLoader for UA-DETRAC"""
    from torch.utils.data import DataLoader
    
    loader = DETRACLoader(image_dir, annotation_dir)
    dataset = DETRACDataset(loader, video_folders, max_frames)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
