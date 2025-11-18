from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

img = Image.open("images/sample20.jpg")
exif_data = img._getexif()

# Decode all EXIF tags
exif = {TAGS.get(k, k): v for k, v in exif_data.items()}

# GPS Info
gps_info = exif.get("GPSInfo")
if gps_info:
    gps_data = {GPSTAGS.get(t, t): gps_info[t] for t in gps_info}
    print(gps_data)
