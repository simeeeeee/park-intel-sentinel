import argparse
import os
import random
import json

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Create text files indicating the train/test/val splits.')

    # Input directory argument
    parser.add_argument('--data_dir', type=str, help='Base directory to load images')

    parser.add_argument('--test_split', type=float, help='Percentage of images to split into the test set',
                        default=0.2)
    parser.add_argument('--val_split', type=float, help='Percentage of images to split into the val set',
                        default=0.1)
    parser.add_argument('--num_classes', type=int, help='Specify the number of classes if classes json does not exist',
                        default=1)
    parser.add_argument("--onnx_config",  action='store_true', help="Create a Split and Config for onnx export")

    # Parse arguments
    args = parser.parse_args()

    img_input_root_dir = os.path.join(args.data_dir, "images")

    all_img_paths = []
    for root, _, files in os.walk(img_input_root_dir):
        if len(files) == 0:
            continue

        for file in files:
            file_path = os.path.join(root, file)
            relative_path = file_path.replace(img_input_root_dir, "./images")
            all_img_paths.append(relative_path + '\n')

    random.shuffle(all_img_paths)
    file_suffix = "_onnx" if args.onnx_config else ""

    total_images = len(all_img_paths)
    test_num = int(args.test_split * total_images)

    test_filepath = os.path.join(args.data_dir, f"test{file_suffix}.txt")

    val_num = int(args.val_split * total_images)
    val_filepath = os.path.join(args.data_dir, f"val{file_suffix}.txt")

    train_filepath = os.path.join(args.data_dir, f"train{file_suffix}.txt")

    test_images = [all_img_paths.pop() for _ in range(test_num)]
    val_images = [all_img_paths.pop() for _ in range(val_num)]

    # If it's for ONNX calibration export only use 300 examples
    if args.onnx_config:
        test_images = test_images[:300]
        val_images = val_images[:300]
        all_img_paths = all_img_paths[:300]

    with open(test_filepath, 'w') as f:
        f.writelines(test_images)

    with open(val_filepath, 'w') as f:
        f.writelines(val_images)

    with open(train_filepath, 'w') as f:
        f.writelines(all_img_paths)


    # Create data config yaml
    try:
        class_dict_filepath = os.path.join(args.data_dir, "classes.json")
        with open(class_dict_filepath) as f:
            d = json.load(f)
            num_classes = len(list(d.keys()))
    except Exception as e:
        print(f"No classes.json file found at {args.data_dir}, please define! (see example)")
        print(e)
        return

    base_dir = args.data_dir.split("/")[-1]
    if base_dir == '':
        base_dir = args.data_dir.split("/")[-2]
    yaml_filename = f"configs/{base_dir}{file_suffix}_config.yaml"

    with open(yaml_filename, 'w') as f:
        f.write(f"# Train images\n")
        f.write(f"train: {train_filepath}\n")
        f.write(f"\n")

        f.write(f"# Validation images\n")
        f.write(f"val: {val_filepath}\n")
        f.write(f"\n")

        f.write(f"# Test images\n")
        f.write(f"test: {test_filepath}\n")
        f.write(f"\n")

        f.write(f"# Number of classes\n")
        f.write(f"nc: {str(num_classes)}\n")
        f.write(f"\n")

        f.write(f"# Class names\n")
        f.write(f"names: {list(d.keys())}\n")

if __name__ == "__main__":
    main()