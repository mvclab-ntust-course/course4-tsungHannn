# Homework 1
讀取影片後，使用pre-train的yolov8n.pt進行物件追蹤
```python
# 因為只偵測car類別，所以classes設定為2
results = model.track(frame, tracker="bytetrack.yaml", persist=True, classes=2)
```

# Homework 2
使用 Chess Pieces Dataset 偵測西洋棋\
https://public.roboflow.com/object-detection/chess-full

這個Dataset包含西洋棋盤跟棋子，所有照片均從一個固定角度拍攝，使用三腳架置於棋盤左側。\
![image](https://github.com/mvclab-ntust-course/course4-tsungHannn/assets/85086644/0dce0da4-36a5-4cb0-b660-15addc7ae887)

## 訓練：
20個epoch
```python
# Train the model
results = model.train(data='data.yaml', epochs=20, imgsz=640, device=[0,1])
```
![image](https://github.com/mvclab-ntust-course/course4-tsungHannn/assets/85086644/66f7296a-c6cb-411f-8970-6b538ea966a2)
