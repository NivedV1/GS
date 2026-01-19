import cv2
import os

# -----------------------------
# SETTINGS
# -----------------------------
image_folder = "Images"        # folder with 5 images
output_video = "output.mp4"
fps = 30                       # frames per second
seconds_per_image = 2          # each image shown for 2 seconds
image_exts = (".png", ".jpg", ".jpeg")

# -----------------------------
# LOAD IMAGES
# -----------------------------
images = sorted([
    img for img in os.listdir(image_folder)
    if img.lower().endswith(image_exts)
])

if len(images) != 5:
    print(f"⚠️ Found {len(images)} images (expected 5)")

# Read first image to get size
first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = first_frame.shape

# -----------------------------
# VIDEO WRITER
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

frames_per_image = fps * seconds_per_image

# -----------------------------
# WRITE VIDEO
# -----------------------------
for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    frame = cv2.imread(img_path)

    # Resize just in case
    frame = cv2.resize(frame, (width, height))

    # Repeat the same frame
    for _ in range(frames_per_image):
        video.write(frame)

video.release()

print("✅ 10-second video created successfully!")
