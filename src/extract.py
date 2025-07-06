import cv2, os

def extract_frames(video_path, out_dir, fps=0.5):
    print(f"Extracting frames from {video_path} to {out_dir} at {fps} FPS...")
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Video FPS: {vfps}")
    step = max(1, round(vfps / fps))
    count, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read.")
            break
        if count % step == 0:
            fp = os.path.join(out_dir, f"f{count:07d}.jpg")
            cv2.imwrite(fp, frame)
            saved += 1
            print(f"Saved frame {count} to {fp}")
        count += 1
    cap.release()
    print(f"Total frames saved: {saved}")
    return saved