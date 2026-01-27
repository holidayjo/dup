import json
import os

# 1. Load the JSON
json_path = "TrainVal.json"  # Ensure this matches your file name
print(f"🔍 Inspecting: {json_path}")

try:
    with open(json_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("❌ Error: File not found!")
    exit()

# 2. Check Basic Counts
num_images = len(data.get('images', []))
num_anns = len(data.get('annotations', []))
categories = data.get('categories', [])

print(f"✅ Images Found: {num_images}")
print(f"✅ Annotations Found: {num_anns}")

# 3. Check Categories
print("\n📋 Categories in JSON:")
for cat in categories:
    print(f"   - ID: {cat['id']}, Name: '{cat['name']}'")

# 4. Critical Check: Annotations
if num_anns == 0:
    print("\n⚠️ CRITICAL WARNING: 0 Annotations found!")
    print("   This is why MMDetection is crashing.")
    print("   Possible Cause: The conversion script didn't find the .txt files.")
    print("   Solution: Ensure .txt files are in the SAME folder as images.")

# 5. Critical Check: File Paths
if num_images > 0:
    sample_img = data['images'][0]['file_name']
    print(f"\n📂 Sample File Path in JSON: {sample_img}")
    if sample_img.startswith("/"):
        print("   ℹ️ Note: JSON contains Absolute Paths.")
        print("   In your config, 'data_prefix' might be duplicating the path.")