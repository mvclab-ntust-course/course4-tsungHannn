import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# 讀取影片
video_path = 'argoverse.mp4'
cap = cv2.VideoCapture(video_path)

codec = cv2.VideoWriter_fourcc(*'mp4v')
frameRate = int(cap.get(cv2.CAP_PROP_FPS))
outputStream = cv2.VideoWriter("output.mp4", codec, frameRate, (int(cap.get(3)),int(cap.get(4))))

if not cap.isOpened():
    print(f"Error opening video file {video_path}")
    exit()

# 載入YOLOv8模型
model = YOLO('yolov8n.pt')  
track_history = defaultdict(lambda: [])
# 讀取影片
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    

    # car的類別是2
    results = model.track(frame, tracker="bytetrack.yaml", persist=True, classes=2)


    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    annotated_frame = results[0].plot()


    # Plot the tracks
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 30:  # retain 90 tracks for 90 frames
            track.pop(0)

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)


    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    outputStream.write(annotated_frame)


cap.release()
cv2.destroyAllWindows()
outputStream.release()
