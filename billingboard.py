from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
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
imagepath = "images/sample10.jpg"       # Input image path

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

# Create output directory
os.makedirs("cropped_boards", exist_ok=True)

# ---------------- UTILITIES ----------------
def resize_to_1mp_if_needed(image_path, output_path):
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


def call_gemini_api(text, width_m, height_m, api_key=GEMINI_API_KEY):
    # Default city & location
    default_city = "Andhra Pradesh"
    default_location = "Tirupati Highway"

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key
    }

    # Force Gemini to ONLY output JSON, with a longer reason (5 lines minimum)
    prompt_text = (
        "You are a regulatory compliance evaluator for outdoor advertisements. "
        "Your task is to review the billboard details carefully and respond ONLY in valid JSON format.\n\n"
        "⚠️ RULES:\n"
        "- Respond strictly with a JSON object, no extra text, no markdown, no explanations outside JSON.\n"
        "- JSON must have exactly two keys:\n"
        "   • \"result\": a boolean (true if acceptable, false if illegal/unacceptable)\n"
        "   • \"reason\": a multi-line string (at least 5 lines) explaining WHY the billboard is or is not allowed. "
        "Reasons must be balanced, realistic, and regulatory-focused.\n\n"
        "Billboard details:\n"
        f"- City: {default_city}\n"
        f"- Location: {default_location}\n"
        f"- Size: {width_m} meters (width) x {height_m} meters (height)\n"
        f"- Text/Content: {text}\n\n"
        "👉 OUTPUT FORMAT (STRICT):\n"
        "{\n"
        "  \"result\": true/false,\n"
        "  \"reason\": \"First line of reasoning.\\nSecond line of reasoning.\\nThird line.\\nFourth line.\\nFifth line.\"\n"
        "}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ]
    }

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

            # Ensure JSON only (strip markdown wrappers if any)
            if answer_text.startswith("```"):
                answer_text = answer_text.strip("`").replace("json", "").strip()

            parsed = json.loads(answer_text)

            # ✅ Ensure correct format
            if "result" in parsed and "reason" in parsed and isinstance(parsed["result"], bool):
                return parsed
            else:
                return {
                    "result": None,
                    "reason": f"Unexpected response format. Raw output: {answer_text}"
                }

        except Exception as e:
            return {
                "result": None,
                "reason": f"Parsing error: {str(e)}. Raw response: {response.text}"
            }
    else:
        return {
            "result": None,
            "reason": f"Gemini API request failed with status {response.status_code}: {response.text}"
        }




# ---------------- MAIN LOOP ----------------
if(len(preds)==0):
    print("⚠️ No billboards detected in the image.")
    exit(0)
for idx, p in enumerate(preds):
    x, y, w, h = p["x"], p["y"], p["width"], p["height"]

    # Draw bounding box
    x0, y0, x1, y1 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
    draw.rectangle([x0, y0, x1, y1], outline="red", width=4)
    label = f"{p['class']} ({p['confidence']:.2f})"
    draw.text((x0, max(0, y0 - 15)), label, fill="red")

    # Crop
    cropped = img.crop((x0, y0, x1, y1))
    crop_filename = f"cropped_boards/board_{idx+1}.jpg"
    cropped.save(crop_filename)

    # Resize + OCR
    resized_path = f"cropped_boards/board_{idx+1}_resized.jpg"
    processed_path = resize_to_1mp_if_needed(crop_filename, resized_path)
    extracted_text = ocr_space_file(processed_path)

    print(f"\n📌 Board {idx+1}:")
    if extracted_text and extracted_text.strip():
        print("📄 Extracted Text:\n", extracted_text.strip())

        # Billboard legality check (using default city & location)
        gemini_response = call_gemini_api(
            text=extracted_text.strip(),
            width_m=5,  # default example size
            height_m=3
        )

        print("\n🧠 Billboard Legality Check:\n", json.dumps(gemini_response, indent=2))
    else:
        print("⚠️ No text detected; skipping Gemini API call.")

# ---------------- SAVE FINAL IMAGE ----------------
img.save("detected_bounding_box.jpg")
print("\n✅ Process completed: Cropped boards saved, OCR done, Gemini verification complete, and annotated image saved as detected_bounding_box.jpg")
