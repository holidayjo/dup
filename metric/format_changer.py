import os

def reformat_yolo_labels(input_folder, output_folder=None):
    # If no output folder is provided, overwrite the input files
    if output_folder is None:
        output_folder = input_folder

    # Make sure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of all files in the input folder
    all_files = os.listdir(input_folder)
    total_files = len([f for f in all_files if f.endswith(".txt")])
    current_file_number = 0

    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        # print(filename)
        if filename.endswith(".txt"):
            current_file_number += 1
            print(f"Processing file {current_file_number}/{total_files}: {filename}")

            input_file_path  = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)

            with open(input_file_path, 'r') as file:
                lines = file.readlines()

            with open(output_file_path, 'w') as file:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 6:
                        class_id = parts[0]
                        x_center = parts[1]
                        y_center = parts[2]
                        box_width = parts[3]
                        box_height = parts[4]
                        confidence = parts[5]

                        # Reformat to <class, conf, x, y, w, h>
                        new_line = f"{class_id} {confidence} {x_center} {y_center} {box_width} {box_height}\n"
                        file.write(new_line)
                    else:
                        print(f"Warning: Skipping malformed line in {input_file_path}: {line}")

# Example usage
input_folder  = r'E:\Downloads\trainset_pred_v7_lite-20240614T224132Z-001\trainset_pred_v7_lite'
output_folder = r'E:\gdrive\My Drive\IEEE_Access\code\metric\results\v7_lite\trainset_pred_v7_lite_formatted'  # Optional, can be the same as input_folder

reformat_yolo_labels(input_folder, output_folder)
