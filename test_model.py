from pathlib import Path
from ultralytics import YOLO

# Load your trained model
model = YOLO("models/best.pt")

# Folder containing test videos
VIDEO_FOLDER = Path("test_videos")

# Supported video formats
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg", ".m4v"}

# Find all video files
video_files = sorted(
    [
        video
        for video in VIDEO_FOLDER.iterdir()
        if video.is_file() and video.suffix.lower() in VIDEO_EXTENSIONS
    ]
)

if not video_files:
    print("❌ No video files found in the 'test_videos' folder.")
    exit()

# Run detection on every video
for video in video_files:
    print(f"\n{'=' * 70}")
    print(f"Processing: {video.name}")
    print(f"{'=' * 70}")

    model.predict(
        source=str(video),
        show=True,
        save=True,
        conf=0.25
    )

print("\n✅ Detection completed for all videos!")