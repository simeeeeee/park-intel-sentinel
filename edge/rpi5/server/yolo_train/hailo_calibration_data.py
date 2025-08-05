from PIL import Image
import os
import argparse
from tqdm import tqdm
import random


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create the calibration data for the Hailo quantization")
    parser.add_argument("--data_dir", type=str, required=True, help="The dataset path")
    parser.add_argument("--target_dir", type=str, default="calib", help="The target directory to save processed images")
    parser.add_argument("--image_size", nargs='+', type=int, default=(640, 640), help="Input image size")
    parser.add_argument("--num_images", type=int, default=1024, help="Number of calibration images")
    parser.add_argument("--image_extensions", nargs='+', type=str, default=['.jpg', '.jpeg', '.png'],
                        help="Image file extensions to look for")

    return parser.parse_args()


def resize_and_crop(img, target_w, target_h):
    """Resize and crop image to target size while maintaining aspect ratio."""
    # Calculate the resize dimensions
    ratio = max(target_w / img.width, target_h / img.height)
    new_width = int(img.width * ratio)
    new_height = int(img.height * ratio)

    # Resize so the smallest dimension is target_size
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Center crop
    left = (img.width - target_w) // 2
    top = (img.height - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    return img.crop((left, top, right, bottom))


def find_image_files(directory, extensions):
    """Find all image files with specified extensions in directory and its subdirectories."""
    image_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in extensions):
                image_files.append(os.path.join(root, file))

    return image_files


def main():
    args = parse_arguments()

    # Ensure target directory exists
    target_dir_path = os.path.join(args.target_dir, "calib")
    os.makedirs(target_dir_path, exist_ok=True)

    # Find all image files in the data directory
    print(f"Searching for images in {args.data_dir}...")
    all_image_paths = find_image_files(args.data_dir, args.image_extensions)

    if not all_image_paths:
        print(f"No images found in {args.data_dir} with extensions {args.image_extensions}")
        return

    print(f"Found {len(all_image_paths)} images")

    # Randomly select images if there are more than needed
    random.shuffle(all_image_paths)
    images_list = all_image_paths[:args.num_images]

    # Adjust num_images if we don't have enough
    actual_num_images = len(images_list)
    if actual_num_images < args.num_images:
        print(f"Warning: Only found {actual_num_images} images, less than the requested {args.num_images}")

    print("Creating calibration data for Hailo export")
    for idx, filepath in tqdm(enumerate(images_list)):
        try:
            # Open the image using PIL
            with Image.open(filepath) as img:
                # Convert image to RGB (in case it's not)
                img = img.convert("RGB")

                # Resize and crop
                processed_img = resize_and_crop(img, args.image_size[0], args.image_size[1])

                # Save the processed image
                output_filepath = os.path.join(target_dir_path, f"processed_{idx}.jpg")
                processed_img.save(output_filepath, format="JPEG")
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue


if __name__ == "__main__":
    main()