#normalises the coodrinates between 0 and 1 

import os
from PIL import Image

def normalize_coordinates(label_file, image_file):
    with Image.open(image_file) as img:
        width, height = img.size

    with open(label_file, 'r') as f:
        lines = f.readlines()

    normalized_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 9:
            continue  # skip malformed lines

        class_index = parts[0]
        coords = list(map(float, parts[1:]))

        # Normalize each coordinate
        norm_coords = []
        for i, val in enumerate(coords):
            if i % 2 == 0:  # x-coordinate
                norm_coords.append(str(round(val / width, 6)))
            else:           # y-coordinate
                norm_coords.append(str(round(val / height, 6)))

        normalized_lines.append(f"{class_index} " + " ".join(norm_coords))

    with open(label_file, 'w') as f:
        f.write("\n".join(normalized_lines))

def normalize_all_labels(dataset_root):
    for subdir, _, files in os.walk(dataset_root):
        if "labels" in subdir:
            for file in files:
                if file.endswith('.txt'):
                    label_path = os.path.join(subdir, file)
                    image_path = label_path.replace("labels", "images").replace(".txt", ".jpg")
                    if not os.path.exists(image_path):
                        image_path = image_path.replace(".jpg", ".png")  # fallback

                    if os.path.exists(image_path):
                        normalize_coordinates(label_path, image_path)

# Usage
normalize_all_labels("Dataset")  # Replace with your actual dataset folder name
