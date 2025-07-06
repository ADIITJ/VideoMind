import sys
from ultralytics import YOLO

def main(frame_path):
    print(f"Testing YOLOv8 on frame: {frame_path}")
    try:
        model = YOLO('yolov8n.pt')
        results = model.predict(
            source=frame_path,
            device='cpu',
            stream=False,
            workers=0
        )
        print("YOLO inference successful.")
        print(f"Detected classes: {results[0].names}")
        print(f"Boxes: {results[0].boxes}")
    except Exception as e:
        print(f"Error during YOLO inference: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_yolo.py data/frames/f0000000.jpg")
        sys.exit(1)
    main(sys.argv[1])
