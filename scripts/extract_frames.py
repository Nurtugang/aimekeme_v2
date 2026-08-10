import base64
import sys
from pathlib import Path
import cv2

if len(sys.argv) < 2:
    sys.exit("Использование: python extract_frames.py <путь_к_видео>")

video_path = Path(sys.argv[1]).resolve()

if not video_path.exists():
    sys.exit(f"Файл не найден: {video_path}")

out_dir = video_path.parent / video_path.stem
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

frame_idx = 0
saved_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx % int(fps) == 0:
        name = f"frame_{saved_idx:04d}"
        frame_dir = out_dir / name
        frame_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(frame_dir / f"{name}.jpg"), frame)
        
        _, buffer = cv2.imencode(".jpg", frame)
        b64_str = base64.b64encode(buffer).decode("utf-8")
        
        with open(frame_dir / f"{name}.txt", "w", encoding="utf-8") as f:
            f.write(b64_str)
            
        saved_idx += 1
    
    frame_idx += 1

cap.release()
print(f"Готово! Результаты сохранены в: {out_dir}")