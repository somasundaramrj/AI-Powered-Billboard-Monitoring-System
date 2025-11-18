import os
import re
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pytesseract
import easyocr
import json

# ---- CONFIG ----
API_URL = "https://serverless.roboflow.com"
API_KEY = "tVEUHfHYIYYbM9tc8ZNd"
WORKSPACE = "billing-board-detection"
WORKFLOW_ID = "detect-count-and-visualize"
INPUT_IMAGE = "images/billi2.jpg"
OUTPUT_DIR = "output_crops"
OCR_ENGINE = "tesseract"  # or "easyocr"
CONFIDENCE_THRESH = 0.5
TARGET_CLASS = "billboard"  # Focus extraction only on this class

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set Tesseract OCR executable path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_for_ocr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
    den = cv2.bilateralFilter(clahe, d=7, sigmaColor=50, sigmaSpace=50)
    thr = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 31, 5)
    kernel = np.ones((2,2), np.uint8)
    morph = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    return morph

def deskew(image_gray):
    inv = cv2.bitwise_not(image_gray)
    coords = cv2.findNonZero(inv)
    if coords is None:
        return image_gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    (h, w) = image_gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(image_gray, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated

def run_easyocr(prep):
    reader = easyocr.Reader(['en'], gpu=False)
    prep_rgb = cv2.cvtColor(prep, cv2.COLOR_GRAY2RGB)
    results = reader.readtext(prep_rgb)
    lines = [txt.strip() for _, txt, conf in results if txt.strip()]
    return "\n".join(lines)

def run_tesseract(prep):
    config = "--oem 3 --psm 6"
    return pytesseract.image_to_string(prep, config=config)

def clean_ocr_text(raw_text):
    # Remove non-ASCII characters and excessive whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', raw_text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---- Run YOLO workflow to detect billboards ----
client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)
result = client.run_workflow(
    workspace_name=WORKSPACE,
    workflow_id=WORKFLOW_ID,
    images={"image": INPUT_IMAGE},
    use_cache=True
)
if isinstance(result, str):
    result = json.loads(result)
if isinstance(result, list):
    result = result[0]
preds = result.get("predictions", {}).get("predictions", [])
if not preds:
    raise RuntimeError("No detections found.")

# ---- Load original image ----
img_pil = Image.open(INPUT_IMAGE).convert("RGB")
draw = ImageDraw.Draw(img_pil)
img_np = np.array(img_pil)[:, :, ::-1]  # Convert RGB to BGR for OpenCV
H, W = img_np.shape[:2]
ocr_results = []

# ---- Process each detected object, focusing only on 'billboard' ----
for p in preds:
    if p.get("confidence", 0) < CONFIDENCE_THRESH:
        continue
    
    if p.get("class").lower() != TARGET_CLASS:
        # Skip non-billboard detections
        continue
    
    x, y, w, h = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"])
    x0, y0 = max(0, int(round(x - w/2))), max(0, int(round(y - h/2)))
    x1, y1 = min(W, int(round(x + w/2))), min(H, int(round(y + h/2)))

    # Draw bounding box for billboard only
    draw.rectangle([x0, y0, x1, y1], outline="red", width=4)
    label = f"{p['class']} ({p['confidence']:.2f})"
    draw.text((x0, y0 - 15), label, fill="red")

    # Crop billboard region for OCR
    crop_bgr = img_np[y0:y1, x0:x1]
    if crop_bgr.size == 0:
        continue

    # Preprocess and deskew
    prep = preprocess_for_ocr(crop_bgr)
    prep = deskew(prep)

    # OCR on the billboard area
    if OCR_ENGINE.lower() == "easyocr":
        raw_text = run_easyocr(prep)
    else:
        raw_text = run_tesseract(prep)
    
    clean_text = clean_ocr_text(raw_text)

    # Save crop images for inspection
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"crop_{x0}_{y0}.jpg"), crop_bgr)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"prep_{x0}_{y0}.png"), prep)

    # Store results
    ocr_results.append({
        "bbox": [x0, y0, x1, y1],
        "confidence": p.get("confidence", 0),
        "class": p.get("class"),
        "raw_text": raw_text.strip(),
        "clean_text": clean_text,
    })

# ---- Save and display image with bounding boxes ----
img_pil.save(os.path.join(OUTPUT_DIR, "detected_bounding_box.jpg"))
img_pil.show()

# ---- Print OCR results only for billboards ----
for item in ocr_results:
    print(f"BBox: {item['bbox']} (conf {item['confidence']:.2f}) class: {item['class']}")
    print("Raw OCR Text:")
    print(item['raw_text'])
    print("Cleaned OCR Text:")
    print(item['clean_text'])
    print("="*40)
