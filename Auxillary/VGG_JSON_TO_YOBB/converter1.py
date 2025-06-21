#This code is used to convert the dota datset to required obb datset for yolo

import os

def convert_labels_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                label_path = os.path.join(root, file)
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                converted_lines = []
                for line in lines:
                    parts = line.strip().split()

                    # Validate line format
                    if len(parts) < 10:
                        continue  # Skip malformed lines

                    x1_y1_to_x4_y4 = parts[:8]
                    class_index = parts[-1]  # Last element is the class index
                    new_line = f"{class_index} " + " ".join(x1_y1_to_x4_y4)
                    converted_lines.append(new_line)

                # Overwrite the file with converted lines
                with open(label_path, 'w') as f:
                    f.write("\n".join(converted_lines))

# Example usage
dataset_root = "Dataset"  # Replace with full path if needed
convert_labels_in_folder(dataset_root)
