import cv2
import numpy as np
from ultralytics import YOLO
import requests
import json
import os
import torch
import torch.serialization

# Add DetectionModel to safe globals
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

class VideoProcessor:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)  # Load YOLOv8 Nano
        self.ollama_url = "http://localhost:11434/api/generate"

    def process_video(self, video_path, frame_interval=10):
        """Process video and extract scene descriptions."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        descriptions = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                # Object detection
                results = self.model(frame)
                objects = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        label = self.model.names[cls]
                        conf = float(box.conf)
                        if conf > 0.5:
                            objects.append(label)

                # Generate scene description with Mistral 7B
                prompt = f"Describe this scene: Objects detected include {', '.join(objects)}."
                description = self._call_ollama(prompt)
                timestamp = frame_idx / fps
                descriptions.append({"timestamp": timestamp, "objects": objects, "description": description})

            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return descriptions

    def _call_ollama(self, prompt):
        """Call Mistral 7B via Ollama API."""
        try:
            payload = {
                "model": "mistral:7b-instruct-q4_0",
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            return json.loads(response.text).get("response", "")
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"