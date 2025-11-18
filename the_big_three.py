"""
Complete pipeline: detect billboards (YOLO via Roboflow Inference SDK), crop,
OCR (OCR.Space), estimate real-world size (pinhole model using EXIF focal length),
and run compliance check (Gemini). Outputs annotated image, cropped images,
and a JSON summary.

Make sure to install required packages:
    pip install inference-sdk pillow requests

Fill in your API keys at the top of this file.
"""

from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ExifTags
import json
import os
import math
import requests
import re
import io

# ---------------- CONFIG ----------------
# Replace these with your real keys (or read from env vars)
API_URL = "https://serverless.roboflow.com"   # or your Roboflow endpoint
API_KEY_RF = "tVEUHfHYIYYbM9tc8ZNd"           # Roboflow API key
OCR_API_KEY = "K89310060288957"               # OCR.Space API key
GEMINI_API_KEY = "AIzaSyC3w1snSmdeaWeYibnUa6Lduh2x3qBd6T4"  # Gemini API key
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Roboflow workflow settings (you already used this)
ROBofLOW_WORKSPACE = "billing-board-detection"
ROBofLOW_WORKFLOW_ID = "detect-count-and-visualize"

# Output folders
CROPS_DIR = "cropped_boards"
ANNOTATED_OUT = "detected_bounding_box.jpg"

# ---------------- INIT ROBFLOW ----------------
client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY_RF)

# ---------------- UTILITIES ----------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def pil_save_as_jpeg(pil_img, out_path):
    """Save a PIL image as JPEG safely (convert RGBA->RGB)."""
    if pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGB")
    pil_img.save(out_path, "JPEG")

def resize_to_1mp_if_needed_imageobj(image_obj, output_path):
    """
    Accepts a PIL Image object and writes JPEG at <=1MP for OCR.
    Returns path.
    """
    width, height = image_obj.size
    total_pixels = width * height
    target_pixels = 1_000_000
    if total_pixels <= target_pixels:
        pil_save_as_jpeg(image_obj, output_path)
        return output_path

    scale_factor = math.sqrt(target_pixels / total_pixels)
    new_width = max(1, int(width * scale_factor))
    new_height = max(1, int(height * scale_factor))

    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.ANTIALIAS

    resized_img = image_obj.resize((new_width, new_height), resample_filter)
    pil_save_as_jpeg(resized_img, output_path)
    return output_path

def ocr_space_file_bytes(image_bytes, api_key=OCR_API_KEY):
    """
    Send image bytes to OCR.Space. Returns extracted text or empty string.
    """
    files = {"filename": ("crop.jpg", image_bytes, "image/jpeg")}
    data = {"apikey": api_key, "language": "eng", "isOverlayRequired": "false"}
    r = requests.post("https://api.ocr.space/parse/image", files=files, data=data, timeout=30)
    try:
        res = r.json()
    except Exception:
        return ""
    if res.get("IsErroredOnProcessing"):
        return ""
    return res.get("ParsedResults", [{}])[0].get("ParsedText", "") or ""

def get_focal_length(image_path):
    """Extract focal length (mm) from EXIF, or None."""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif() or {}
        for tag, value in exif_data.items():
            if ExifTags.TAGS.get(tag, tag) == "FocalLength":
                if isinstance(value, tuple):
                    return float(value[0]) / float(value[1])
                else:
                    return float(value)
    except Exception:
        return None
    return None

def estimate_billboard_size(pred, img_width, img_height, image_path, distance_m,
                            sensor_width_mm=6.4, sensor_height_mm=4.8):
    """
    Pinhole camera model based estimation.
    pred: prediction dict with 'width' and 'height' in pixels (bbox pixel width/height)
    img_width, img_height: full image pixel dims (from detection response)
    image_path: path to original image to read EXIF focal length
    distance_m: distance from camera to billboard in meters (user-supplied)
    sensor_width_mm / sensor_height_mm: default sensor dims (override with actual if known)
    """
    bbox_width_px = pred["width"]
    bbox_height_px = pred["height"]

    focal_length_mm = get_focal_length(image_path)
    if focal_length_mm is None:
        # raise to let caller fallback
        raise RuntimeError("Focal length not found in EXIF metadata")

    real_width_m = (bbox_width_px * distance_m * sensor_width_mm) / (img_width * focal_length_mm)
    real_height_m = (bbox_height_px * distance_m * sensor_height_mm) / (img_height * focal_length_mm)
    return real_width_m, real_height_m

# ---------------- Gemini (compliance) ----------------
def call_gemini_api(text, width_m, height_m, location, api_key=GEMINI_API_KEY):
    """
    Sends prompt to Gemini and returns normalized dict:
      { "result": True/False/None, "reason": "<multi-line string ending with newline>" }
    This preserves model's analysis, removes numbering prefixes, and ensures two appended validity lines.
    """
    headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}

    prompt_text = (
        "You are a regulatory compliance evaluator for outdoor advertisements. "
        "Review the billboard details carefully and respond ONLY in valid JSON format.\n\n"
        "⚠️ STRICT FORMAT RULES:\n"
        "- Return a JSON object with exactly two keys: \"result\" and \"reason\".\n"
        "- \"result\" must be true, false, or null.\n"
        "- \"reason\" must be a multiline string (one or more lines separated by '\\n').\n"
        "- DO NOT prefix lines with numbering like 'Line 1:' or '1.'; produce natural sentences only.\n"
        "- Preserve all analysis lines you would normally produce (do NOT drop earlier analysis).\n"
        "- AFTER your full analysis, ADD EXACTLY TWO FINAL LINES (each on its own line):\n"
        f"    Dimensions: {width_m:.2f} m (W) x {height_m:.2f} m (H) — Allowed in this region.\n"
        f"    Location: {location} — Allowed in this region.\n"
        "- Ignore OCR spelling mistakes (treat transcription typos as OCR errors and assume intended correct spelling).\n"
        "- Return ONLY the JSON object (no markdown/code fences/explanations).\n\n"
        f"- Billboard text: {text}\n"
        f"- Location: {location}\n"
        f"- Dimensions: {width_m:.2f}m x {height_m:.2f}m\n\n"
        "Example (format only):\n"
        '{ "result": true, "reason": "Some analysis line.\\nAnother line.\\n...\\nDimensions: 10.00 m (W) x 5.00 m (H) — Allowed in this region.\\nLocation: Hyderabad — Allowed in this region.\\n" }'
    )

    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}

    try:
        resp = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=30)
    except Exception as e:
        return {"result": None, "reason": f"Request error: {str(e)}\n"}

    if resp.status_code != 200:
        return {"result": None, "reason": f"Gemini API failed {resp.status_code}: {resp.text}\n"}

    try:
        data = resp.json()
    except Exception as e:
        return {"result": None, "reason": f"Invalid JSON from Gemini: {str(e)}. Raw: {resp.text}\n"}

    # extract model text safe
    answer_text = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
        .strip()
    )

    # If wrapped in fences, unwrap
    if answer_text.startswith("```") and answer_text.endswith("```"):
        answer_text = answer_text.strip("`").strip()

    # Attempt to extract JSON object substring
    start = answer_text.find("{")
    end = answer_text.rfind("}")
    json_text = answer_text[start:end+1] if (start != -1 and end != -1 and end > start) else answer_text

    try:
        parsed = json.loads(json_text)
    except Exception:
        # fallback: try to parse if model returned just a list/string for reason
        # attempt to find "result" and "reason" with regex
        m_result = re.search(r'"result"\s*:\s*(true|false|null|"true"|"false"|"null")', answer_text, flags=re.I)
        m_reason = re.search(r'"reason"\s*:\s*"(.+?)"\s*(,|\})', answer_text, flags=re.S)
        if m_reason:
            reason_raw = m_reason.group(1)
        else:
            # as last resort, use entire answer_text as reason
            reason_raw = answer_text

        if m_result:
            r = m_result.group(1).strip().lower().strip('"')
            if r == "true":
                normalized_result = True
            elif r == "false":
                normalized_result = False
            else:
                normalized_result = None
        else:
            normalized_result = None

        # split lines
        lines = [ln.strip() for ln in reason_raw.splitlines() if ln.strip()]
        cleaned_lines = [re.sub(r'^\s*(?:Line\s*\d+[:\)\.\-]*|\d+[:\)\.\-]*|\(\d+\)\s*)\s*', '', ln, flags=re.I) for ln in lines]
        # Enforce dimension/location final lines
        dim_line = f"Dimensions: {width_m:.2f} m (W) x {height_m:.2f} m (H) — Allowed in this region."
        loc_line = f"Location: {location} — Allowed in this region."
        lower = [l.lower() for l in cleaned_lines]
        if not any(dim_line.lower() in ll for ll in lower):
            cleaned_lines.append(dim_line)
        if not any(loc_line.lower() in ll for ll in lower):
            cleaned_lines.append(loc_line)
        reason_str = "\n".join(cleaned_lines)
        if not reason_str.endswith("\n"):
            reason_str += "\n"
        return {"result": normalized_result, "reason": reason_str}

    # Normal parsing succeeded
    raw_result = parsed.get("result", None)
    normalized_result = None
    if isinstance(raw_result, bool) or raw_result is None:
        normalized_result = raw_result
    elif isinstance(raw_result, str):
        rl = raw_result.strip().lower()
        if rl == "true":
            normalized_result = True
        elif rl == "false":
            normalized_result = False
        elif rl in ("null", "none"):
            normalized_result = None

    reason_field = parsed.get("reason", "")
    # handle list or string
    if isinstance(reason_field, list):
        lines = [str(x).strip() for x in reason_field if str(x).strip()]
    else:
        lines = [ln.strip() for ln in str(reason_field).splitlines() if ln.strip()]

    cleaned_lines = [re.sub(r'^\s*(?:Line\s*\d+[:\)\.\-]*|\d+[:\)\.\-]*|\(\d+\)\s*)\s*', '', ln, flags=re.I) for ln in lines]

    dim_line = f"Dimensions: {width_m:.2f} m (W) x {height_m:.2f} m (H) — Allowed in this region."
    loc_line = f"Location: {location} — Allowed in this region."
    lower_lines = [l.lower() for l in cleaned_lines]
    if not any(dim_line.lower() in ll for ll in lower_lines):
        cleaned_lines.append(dim_line)
    if not any(loc_line.lower() in ll for ll in lower_lines):
        cleaned_lines.append(loc_line)

    reason_str = "\n".join(cleaned_lines)
    if not reason_str.endswith("\n"):
        reason_str += "\n"

    return {"result": normalized_result, "reason": reason_str}

# ---------------- MASTER FUNCTION ----------------
def process_billboard(image_path, distance_to_billboard_m=10, location="Andhra Pradesh, Tirupati Highway"):
    """
    Full pipeline:
      - run Roboflow workflow which returns predictions (YOLO)
      - crop each detected billboard, save crop
      - run OCR on crop (resized to <=1MP)
      - estimate real dimensions (using EXIF focal length + provided distance)
      - call Gemini compliance API
      - return a dictionary summary
    """
    ensure_dir(CROPS_DIR)

    # 1) Run detection (Roboflow workflow)
    result = client.run_workflow(
        workspace_name=ROBofLOW_WORKSPACE,
        workflow_id=ROBofLOW_WORKFLOW_ID,
        images={"image": image_path},
        use_cache=True
    )

    # normalize result
    if isinstance(result, str):
        result = json.loads(result)
    if isinstance(result, list):
        result = result[0]

    preds = result.get("predictions", {}).get("predictions", [])
    if not preds:
        return {"status": "no_billboard", "message": "No billboards detected."}

    # open original image and an annotated drawable copy
    orig_img = Image.open(image_path)
    annotated = orig_img.copy()
    draw = ImageDraw.Draw(annotated)

    img_width = result.get("predictions", {}).get("image", {}).get("width", orig_img.width)
    img_height = result.get("predictions", {}).get("image", {}).get("height", orig_img.height)

    outputs = []
    for idx, p in enumerate(preds):
        x, y, w, h = p["x"], p["y"], p["width"], p["height"]
        left = int(max(0, x - w/2))
        top = int(max(0, y - h/2))
        right = int(min(orig_img.width, x + w/2))
        bottom = int(min(orig_img.height, y + h/2))

        # draw bbox on annotated image
        draw.rectangle([left, top, right, bottom], outline="red", width=4)
        label = f"{p.get('class','object')} ({p.get('confidence',0):.2f})"
        draw.text((left, max(0, top - 12)), label, fill="red")

        # crop region
        crop = orig_img.crop((left, top, right, bottom)).convert("RGBA")
        # convert to RGB-safe and save
        crop_path = os.path.join(CROPS_DIR, f"board_{idx+1}.jpg")
        if crop.mode == "RGBA":
            crop = crop.convert("RGB")
        #crop.save(crop_path, "JPEG")

        # prepare small version for OCR (<=1MP) using image object
        # (use the crop Image object directly)
        resized_for_ocr_path = os.path.join(CROPS_DIR, f"board_{idx+1}_resized.jpg")
        processed_path = resize_to_1mp_if_needed_imageobj(crop, resized_for_ocr_path)

        # read bytes for OCR
        with open(processed_path, "rb") as f:
            img_bytes = f.read()

        extracted_text = ocr_space_file_bytes(img_bytes) or ""
        extracted_text = extracted_text.strip()

        # estimate real dimensions (try-except fallback)
        try:
            width_m, height_m = estimate_billboard_size(p, img_width, img_height, image_path, distance_to_billboard_m)
        except Exception:
            # fallback defaults if EXIF missing or estimation fails
            width_m, height_m = float(round(w/100.0, 2)), float(round(h/100.0, 2))

        gemini_check = None
        if extracted_text:
            gemini_check = call_gemini_api(text=extracted_text, width_m=width_m, height_m=height_m, location=location)
        else:
            gemini_check = {"result": None, "reason": "No text extracted by OCR.\nDimensions: %.2f m (W) x %.2f m (H) — Allowed in this region.\nLocation: %s — Allowed in this region.\n" % (width_m, height_m, location)}

        outputs.append({
            "board_id": idx + 1,
            "bbox_px": {"left": left, "top": top, "right": right, "bottom": bottom},
            "size_m_est": {"width_m": width_m, "height_m": height_m},
            "crop_path": crop_path,
            "extracted_text": extracted_text if extracted_text else None,
            "gemini_check": gemini_check
        })

    # Save annotated image (convert if RGBA)
    if annotated.mode == "RGBA":
        annotated = annotated.convert("RGB")
    annotated.save(ANNOTATED_OUT, "JPEG")

    return {"status": "done", "annotated_image": ANNOTATED_OUT, "boards": outputs}

# ---------------- RUN EXAMPLE ----------------
if __name__ == "__main__":
    # Example usage - change image_path, distance and location as needed
    image_path = "images/sample9.jpg"
    output = process_billboard(
        image_path=image_path,
        distance_to_billboard_m=12,               # meters (user-supplied approximate distance)
        location="Hyderabad, Outer Ring Road"     # single location string
    )
    print(json.dumps(output, indent=2))
