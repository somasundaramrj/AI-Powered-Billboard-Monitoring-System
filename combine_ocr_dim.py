from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ExifTags
import json
import os
import math
import requests

# ---------------- CONFIG ----------------
API_URL = "https://serverless.roboflow.com"
API_KEY_RF = "tVEUHfHYIYYbM9tc8ZNd"   # Roboflow API key
OCR_API_KEY = "K89310060288957"        # OCR.Space API key
GEMINI_API_KEY = "AIzaSyC3w1snSmdeaWeYibnUa6Lduh2x3qBd6T4"  # Gemini API key
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
imagepath = "images/sample21.jpg"       # Default input image path

# ---------------- INIT ROBFLOW ----------------
client = InferenceHTTPClient(
    api_url=API_URL,
    api_key=API_KEY_RF
)

result = client.run_workflow(
    workspace_name="billing-board-detection",
    workflow_id="detect-count-and-visualize",
    images={"image": imagepath},
    use_cache=True
)

# Ensure result is dict
if isinstance(result, str):
    result = json.loads(result)
if isinstance(result, list):
    result = result[0]

# ---------------- IMAGE PREP ----------------
img = Image.open(imagepath)
draw = ImageDraw.Draw(img)

# Extract predictions
preds = result["predictions"]["predictions"]
img_width = result["predictions"]["image"]["width"]
img_height = result["predictions"]["image"]["height"]

# Create output directory
os.makedirs("cropped_boards", exist_ok=True)


# ---------------- UTILITIES ----------------
def resize_to_1mp_if_needed(image_path, output_path):
    """Resize image to max 1 megapixel (for OCR API)."""
    img = Image.open(image_path)
    width, height = img.size
    total_pixels = width * height
    target_pixels = 1_000_000  # 1MP

    if total_pixels <= target_pixels:
        img.save(output_path)
        return output_path

    scale_factor = math.sqrt(target_pixels / total_pixels)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.ANTIALIAS

    resized_img = img.resize((new_width, new_height), resample_filter)
    resized_img.save(output_path)
    return output_path


def ocr_space_file(filename, api_key=OCR_API_KEY):
    """Extract text from image using OCR.Space API."""
    with open(filename, 'rb') as f:
        r = requests.post(
            'https://api.ocr.space/parse/image',
            files={filename: f},
            data={'apikey': api_key, 'language': 'eng'}
        )
    result = r.json()
    if result.get("IsErroredOnProcessing"):
        print(f"⚠️ OCR Error: {result.get('ErrorMessage')}")
        return None
    return result.get("ParsedResults")[0].get("ParsedText")


def get_focal_length(image_path):
    """Extract focal length from EXIF metadata."""
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


def estimate_billboard_size(pred, img_width, img_height, image_path, distance_m,
                            sensor_width_mm=6.4, sensor_height_mm=4.8):
    """Estimate billboard real-world dimensions using pinhole camera model."""
    bbox_width_px = pred["width"]
    bbox_height_px = pred["height"]

    focal_length_mm = get_focal_length(image_path)
    if focal_length_mm is None:
        raise RuntimeError("⚠️ Focal length not found in EXIF metadata!")

    real_width_m = (bbox_width_px * distance_m * sensor_width_mm) / (img_width * focal_length_mm)
    real_height_m = (bbox_height_px * distance_m * sensor_height_mm) / (img_height * focal_length_mm)

    return real_width_m, real_height_m


def call_gemini_api(text, width_m, height_m, api_key=GEMINI_API_KEY):
    """Evaluate billboard legality using Gemini API."""
    default_city = "Andhra Pradesh"
    default_location = "Tirupati Highway"

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key
    }

    prompt_text = (
"You are a regulatory compliance evaluator for outdoor advertisements. "

"Your task is to review the billboard details carefully and respond ONLY in valid JSON format.\n\n"

"⚠️ RULES:\n"
"- Respond strictly with a JSON object, no extra text, no markdown.\n"
"- JSON must have exactly two keys:\n"
"   • \"result\": boolean OR null\n"
"   • \"reason\": multi-line string (at least 5 lines) explaining WHY.\n\n"

"INTERPRETATION RULES:\n"
"1. If the billboard text contains offensive, abusive, or bad words → result = false.\n"
"2. If the billboard content is unclear or not understandable → result = null.\n"
"3. If the billboard makes a claim that is factually impossible (probability of happening = 0) → result = false.\n"
"4. If the billboard makes an exaggerated but still possible claim (e.g., advertising promises, slogans, future hopes) → result = true.\n"
"5. Otherwise, accept the billboard as valid.\n\n"

f"- City: {default_city}\n"
f"- Location: {default_location}\n"
f"- Size: {width_m:.2f} meters (W) x {height_m:.2f} meters (H)\n"
f"- Text/Content: {text}\n\n"

"👉 OUTPUT FORMAT (STRICT):\n"
"{\n"
" \"result\": true/false/null,\n"
" \"reason\": \"First line.\\nSecond line.\\nThird line.\\nFourth line.\\nFifth line.\"\n"
"}"
)


    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    response = requests.post(GEMINI_API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        try:
            data = response.json()
            answer_text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            if answer_text.startswith("```"):
                answer_text = answer_text.strip("`").replace("json", "").strip()

            parsed = json.loads(answer_text)
            if "result" in parsed and "reason" in parsed and isinstance(parsed["result"], bool):
                return parsed
            else:
                return {"result": None, "reason": f"Unexpected format. Raw: {answer_text}"}
        except Exception as e:
            return {"result": None, "reason": f"Parsing error: {str(e)}. Raw: {response.text}"}
    else:
        return {"result": None, "reason": f"Gemini API failed {response.status_code}: {response.text}"}


# ---------------- MAIN LOOP ----------------
if len(preds) == 0:
    print("⚠️ No billboards detected in the image.")
    exit(0)

distance_to_billboard_m = 10  # Default viewing distance (can adjust dynamically)

for idx, p in enumerate(preds):
    # Bounding box
    x, y, w, h = p["x"], p["y"], p["width"], p["height"]
    x0, y0, x1, y1 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
    draw.rectangle([x0, y0, x1, y1], outline="red", width=4)
    label = f"{p['class']} ({p['confidence']:.2f})"
    draw.text((x0, max(0, y0 - 15)), label, fill="red")

    # Crop
    crop_filename = f"cropped_boards/board_{idx+1}.jpg"
    cropped = img.crop((x0, y0, x1, y1))
    cropped.save(crop_filename)

    # Resize + OCR
    resized_path = f"cropped_boards/board_{idx+1}_resized.jpg"
    processed_path = resize_to_1mp_if_needed(crop_filename, resized_path)
    extracted_text = ocr_space_file(processed_path)

    # Estimate billboard size
    try:
        width_m, height_m = estimate_billboard_size(p, img_width, img_height, imagepath, distance_to_billboard_m)
    except RuntimeError as e:
        print(f"⚠️ Size estimation failed: {e}")
        width_m, height_m = 5, 3  # fallback default

    print(f"\n📌 Board {idx+1}:")
    print(f"📏 Estimated Size: {width_m:.2f} m (W) x {height_m:.2f} m (H)")

    if extracted_text and extracted_text.strip():
        print("📄 Extracted Text:\n", extracted_text.strip())

        # Billboard legality check with real dimensions
        gemini_response = call_gemini_api(
            text=extracted_text.strip(),
            width_m=width_m,
            height_m=height_m
        )
        print("\n🧠 Billboard Legality Check:\n", json.dumps(gemini_response, indent=2))
    else:
        print("⚠️ No text detected; skipping Gemini API call.")

# Save annotated image
img.save("detected_bounding_box.jpg")
print("\n✅ Process completed: Cropped boards saved, OCR done, size estimation + Gemini verification complete, annotated image saved as detected_bounding_box.jpg")
