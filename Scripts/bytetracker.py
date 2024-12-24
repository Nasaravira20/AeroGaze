from ultralytics import YOLO

model = YOLO("/home/cyberkid/Desktop/best.pt")
result = model.track(
    source=r"/home/cyberkid/Desktop/jaya.mp4",
    show=True,
    tracker="/home/cyberkid/Desktop/bytetracker1.yaml",
)
# source = 0

# cli
# yolo track model=yolo11m.pt source=0;
