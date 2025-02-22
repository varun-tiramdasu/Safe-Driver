from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n-seg.pt")

# Perform inference on the image
results = model(r"C:\Users\peket\Documents\drowsiness_2\test1.jpg", save=True)

names = model.names

for r in results:
    for c in r.boxes.cls:
        if(names[int(c)] == "cell phone"):
            print("cell phone detected!")
            