import gdown
import zipfile

# File ID từ Google Drive
file_id = "1L3m6pImvbd7pxnikDaiM5D6bWpQC-ZYv"
output = "data.zip"

print("⬇️ Downloading full data.zip...")
gdown.download(id=file_id, output=output, quiet=False)

print("📦 Extracting...")
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall()

print("✅ Done. Check the 'data/' folder.")
