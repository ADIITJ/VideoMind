from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import numpy as np
from ultralytics import YOLO

# Time-based summarization
def summarize_by_time(captions: dict, window: int = 5) -> dict:
    print(f"Summarizing captions by time window: {window}")
    frames = sorted(captions.keys())
    out = {}
    for i in range(0, len(frames), window):
        group = frames[i:i+window]
        combined = " ".join(captions[f] for f in group)
        out[group[0]] = combined
    print(f"Time-based summary keys: {list(out.keys())}")
    return out

# Object-based summarization
def summarize_by_object(captions: dict) -> dict:
    print("Summarizing captions by detected objects...")
    model = YOLO('yolov8n.pt')
    obj_groups = defaultdict(list)

    for frame, cap in captions.items():
        print(f"Detecting objects in frame: {frame}")
        try:
            results = model.predict(
                source=frame,
                device='cpu',    # use M1 CPU
                stream=False,    # avoid streaming
                workers=0        # no multiprocessing
            )
            names = results[0].names
            for cls_id in set(results[0].boxes.cls.tolist()):
                name = names[int(cls_id)]
                obj_groups[name].append(cap)
        except Exception as e:
            print(f"Error during YOLO inference on {frame}: {e}")
            continue  # Skip this frame if YOLO fails

    print(f"Object-based summary keys: {list(obj_groups.keys())}")
    return {obj: " ".join(set(caps)) for obj, caps in obj_groups.items()}

def summarize_group_with_llm(captions: list, context: str = "") -> str:
    from src.llm_reasoning import ask_with_chunks
    joined = "\n".join(captions)
    prompt = (
        "You are given a list of image captions from a video segment. "
        "Summarize the main events or objects described in these captions. "
        "Base your summary strictly on the provided captions—do not add any new information or speculate. "
        "Keep the summary concise, factual, and under 60 words.\n\n"
        f"Context: {context}\n"
        "Captions:\n"
        f"{joined}\n\n"
        "Summary:"
    )
    # Use all captions as a single chunk for summarization
    return ask_with_chunks([joined], prompt)

def summarize_by_time_llm(captions: dict, window: int = 5) -> dict:
    """
    Summarize captions in time windows using LLM.
    """
    from src.llm_reasoning import summarize_time_llm
    frames = sorted(captions.keys())
    out = {}
    for i in range(0, len(frames), window):
        group = frames[i:i+window]
        group_captions = [captions[f] for f in group]
        summary = summarize_time_llm(group_captions)
        out[group[0]] = summary
    return out

def summarize_by_object_llm(captions: dict) -> dict:
    """
    Summarize captions by detected object using LLM.
    """
    from src.llm_reasoning import summarize_object_llm
    from ultralytics import YOLO
    from collections import defaultdict

    model = YOLO('yolov8n.pt')
    obj_groups = defaultdict(list)
    for frame, cap in captions.items():
        try:
            results = model.predict(
                source=frame,
                device='cpu',
                stream=False,
                workers=0
            )
            names = results[0].names
            for cls_id in set(results[0].boxes.cls.tolist()):
                name = names[int(cls_id)]
                obj_groups[name].append(cap)
        except Exception as e:
            print(f"Error during YOLO inference on {frame}: {e}")
            continue

    out = {}
    for obj, caps in obj_groups.items():
        summary = summarize_object_llm(obj, list(set(caps)))
        out[obj] = summary
    return out
