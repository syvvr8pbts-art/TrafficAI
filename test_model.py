from ultralytics import YOLO

# Load your trained model
model = YOLO("models/best.pt")
while True:
    # Run detection
    results = model.predict(
        source="test_videos/sample.mp4",
        show=True,
        save=True,
        conf=0.25
    )

print("Detection complete!")