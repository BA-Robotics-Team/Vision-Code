import json
import os

def convert_vgg_to_dota(vgg_json_path, output_dir):
    with open(vgg_json_path, 'r') as f:
        vgg_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for image_key, image_info in vgg_data.items():
        image_filename = image_info.get('filename')
        regions_dict = image_info.get('regions', {})
        
        # Some VIA versions store 'regions' as a list; others as a dict
        if isinstance(regions_dict, dict):
            regions = list(regions_dict.values())
        elif isinstance(regions_dict, list):
            regions = regions_dict
        else:
            continue  # skip if invalid

        base_filename = os.path.splitext(image_filename)[0]
        dota_txt_path = os.path.join(output_dir, base_filename + '.txt')

        with open(dota_txt_path, 'w') as out_file:
            for region in regions:
                shape_attr = region.get('shape_attributes', {})
                region_attr = region.get('region_attributes', {})
                
                # Get class name (update this key if your JSON uses something different)
                class_name = region_attr.get('class', 'object')  

                all_x = shape_attr.get('all_points_x', [])
                all_y = shape_attr.get('all_points_y', [])
                if len(all_x) < 4 or len(all_y) < 4:
                    continue  # skip invalid polygons

                # Take first 4 points (or better: find oriented rectangle if needed)
                polygon = list(zip(all_x[:4], all_y[:4]))
                if len(polygon) != 4:
                    continue  # ensure we have 4 points

                coords_flat = ' '.join(str(coord) for point in polygon for coord in point)
                difficulty = '0'
                out_file.write(f"{coords_flat} {class_name} {difficulty}\n")

    print(f"Conversion completed! DOTA-format annotations are in: {output_dir}")
convert_vgg_to_dota("Dataset\\Valid.json", "Dataset\\Valid\\labels")
