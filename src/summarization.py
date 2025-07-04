from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import numpy as np
from ultralytics import YOLO

# Time-based summarization
def summarize_by_time(captions: dict, window: int = 5) -> dict:
    frames = sorted(captions.keys())
    out = {}
    for i in range(0, len(frames), window):
        group = frames[i:i+window]
        combined = " ".join(captions[f] for f in group)
        out[group[0]] = combined
    return out

# Object-based summarization
def summarize_by_object(captions: dict) -> dict:
    model = YOLO('yolov8n.pt')
    obj_groups = defaultdict(list)
    for frame, cap in captions.items():
        results = model(frame)
        names = results[0].names
        for cls_id in set(results[0].boxes.cls.tolist()):
            name = names[int(cls_id)]
            obj_groups[name].append(cap)
    return {obj: " ".join(set(caps)) for obj, caps in obj_groups.items()}