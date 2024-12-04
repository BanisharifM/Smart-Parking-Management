import os
import json

def generate_all_image_files_json(directory="uploaded_images", output_path="src/all_image_files.json"):
    all_image_files = {"test": {}}
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        if os.path.isdir(category_path):
            all_image_files["test"][category] = [
                file for file in os.listdir(category_path) if file.endswith((".jpg", ".png"))
            ]
    with open(output_path, "w") as f:
        json.dump(all_image_files, f, indent=4)
    print(f"JSON file saved at {output_path}")

# Generate the JSON file
generate_all_image_files_json()

