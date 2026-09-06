import gdown
import zipfile

# File ID từ Google Drive
file_id = "19X5UE1s8OxbdvQsG80WgQkNqVY2Y3jFC"
output = "data.zip"

print("⬇️ Downloading full data.zip...")
gdown.download(id=file_id, output=output, quiet=False)

print("📦 Extracting...")
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall()

print("✅ Done. Check the 'data/' folder.")
