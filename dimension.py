from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ExifTags
import json

# -----------------------------
# STEP 1: Initialize Roboflow Client
# -----------------------------
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="tVEUHfHYIYYbM9tc8ZNd"  
)

# -----------------------------
# STEP 2: Get Detection Result
# -----------------------------
result = client.run_workflow(
    workspace_name="billing-board-detection",
    workflow_id="detect-count-and-visualize",
    images={"image": "images/sample20.jpg"},
    use_cache=True
)

# Ensure result is dict
if isinstance(result, str):
    result = json.loads(result)
if isinstance(result, list):
    result = result[0]

# -----------------------------
# STEP 3: Draw Bounding Boxes
# -----------------------------
img = Image.open("images/sample20.jpg")
draw = ImageDraw.Draw(img)

preds = result["predictions"]["predictions"]
img_width = result["predictions"]["image"]["width"]
img_height = result["predictions"]["image"]["height"]

for p in preds:
    x, y, w, h = p["x"], p["y"], p["width"], p["height"]
    x0, y0 = x - w / 2, y - h / 2
    x1, y1 = x + w / 2, y + h / 2
    draw.rectangle([x0, y0, x1, y1], outline="red", width=4)
    label = f"{p['class']} ({p['confidence']:.2f})"
    draw.text((x0, y0 - 10), label, fill="red")

img.save("detected_bounding_box.jpg")
print("✅ Bounding box saved as detected_bounding_box.jpg")

# -----------------------------
# STEP 4: Extract Focal Length from Metadata
# -----------------------------
def get_focal_length(image_path):
    img = Image.open(image_path)
    exif_data = img._getexif()
    focal_length = None
    if exif_data:
        for tag, value in exif_data.items():
            if ExifTags.TAGS.get(tag, tag) == "FocalLength":
                if isinstance(value, tuple):
                    focal_length = value[0] / value[1]
                else:
                    focal_length = float(value)
    return focal_length

# -----------------------------
# STEP 5: Billboard Size Estimation
# -----------------------------
def estimate_billboard_size(pred, img_width, img_height, image_path, distance_m, 
                            sensor_width_mm=6.4, sensor_height_mm=4.8):
    bbox_width_px = pred["width"]
    bbox_height_px = pred["height"]

    # Focal length from EXIF
    focal_length_mm = get_focal_length(image_path)
    if focal_length_mm is None:
        raise RuntimeError("⚠️ Focal length not found in EXIF metadata!")

    # Apply pinhole camera model
    real_width_m = (bbox_width_px * distance_m * sensor_width_mm) / (img_width * focal_length_mm)
    real_height_m = (bbox_height_px * distance_m * sensor_height_mm) / (img_height * focal_length_mm)

    return real_width_m, real_height_m

# -----------------------------
# STEP 6: Run Size Estimation
# -----------------------------
distance_to_billboard_m = 20 # you can change this
for p in preds:
    width, height = estimate_billboard_size(
        p, img_width, img_height, "images/sample20.jpg", distance_to_billboard_m
    )
    print(f"📏 Estimated Billboard Size: {width:.2f} m (W) x {height:.2f} m (H)")
