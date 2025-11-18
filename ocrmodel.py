from PIL import Image
import math
import requests

def resize_to_1mp_if_needed(image_path, output_path):
    img = Image.open(image_path)
    width, height = img.size

    total_pixels = width * height
    target_pixels = 1_000_000  # 1 Megapixel

    if total_pixels <= target_pixels:
        print(f"Image is already under or equal to 1MP ({width}x{height}), no resizing done.")
        img.save(output_path)
        return output_path

    # Calculate scale factor to reduce to ~1MP while keeping aspect ratio
    scale_factor = math.sqrt(target_pixels / total_pixels)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Use the appropriate resampling filter based on Pillow version
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.ANTIALIAS

    resized_img = img.resize((new_width, new_height), resample_filter)
    resized_img.save(output_path)
    print(f"Resized image from ({width}x{height}) to ({new_width}x{new_height})")
    return output_path

def ocr_space_file(filename, api_key='helloworld'):
    with open(filename, 'rb') as f:
        r = requests.post(
            'https://api.ocr.space/parse/image',
            files={filename: f},
            data={'apikey': api_key, 'language': 'eng'}
        )
    result = r.json()
    if result.get("IsErroredOnProcessing"):
        print(f"Error: {result.get('ErrorMessage')}")
        return None
    return result.get("ParsedResults")[0].get("ParsedText")

# Usage
API_KEY = 'K89310060288957'  # Your OCR.Space API key
IMAGE_PATH = 'images/sample8.jpg'
TEMP_RESIZED_PATH = 'images/sample8_resized.jpg'

processed_image_path = resize_to_1mp_if_needed(IMAGE_PATH, TEMP_RESIZED_PATH)
extracted_text = ocr_space_file(processed_image_path, API_KEY)
print("Extracted Text:\n", extracted_text)
