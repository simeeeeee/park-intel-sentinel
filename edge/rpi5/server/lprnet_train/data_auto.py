import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from ultralytics import YOLO

# ===================================================================
# TUNING PARAMETERS (Feel free to adjust these values)
# ===================================================================
# 1. YOLOv8 Model & Input Settings
YOLO_INPUT_SIZE = 640
OUTER_PADDING_PERCENT = 0.1
CONF_THRESHOLD = 0.45

# 2. Detection Filtering
MIN_ASPECT_RATIO_FILTER = 1.8 # Detections with aspect ratio less than this will be ignored
# ===================================================================

def setup_korean_font():
    """Sets up a Korean font for Matplotlib visualization."""
    font_path = 'NanumGothic.ttf'
    if not os.path.exists(font_path):
        print(f"[Warning] Font file not found at '{font_path}'.")
        return
    try:
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        matplotlib.rc('font', family=font_name)
        matplotlib.rcParams['axes.unicode_minus'] = False
        print(f"[Info] Font '{font_name}' has been set up successfully.")
    except Exception as e:
        print(f"[Error] An error occurred during font setup: {e}")

def prepare_image_for_yolo(image, target_size):
    """Prepares the image for YOLOv8 input by applying outer padding and letterboxing."""
    h, w, _ = image.shape
    pad_h = int(h * OUTER_PADDING_PERCENT)
    pad_w = int(w * OUTER_PADDING_PERCENT)
    padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, 
                                      cv2.BORDER_CONSTANT, value=[114, 114, 114])
    h_padded, w_padded, _ = padded_image.shape
    scale = min(target_size / h_padded, target_size / w_padded)
    new_w, new_h = int(w_padded * scale), int(h_padded * scale)
    resized_img = cv2.resize(padded_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    prepared_img = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    dx, dy = (target_size - new_w) // 2, (target_size - new_h) // 2
    prepared_img[dy:dy+new_h, dx:dx+new_w, :] = resized_img
    return prepared_img, scale, dx, dy, pad_w, pad_h

def convert_coords_from_yolo_to_original(box, scale, dx, dy, pad_w, pad_h):
    """Converts YOLO's bounding box coordinates back to the original image's coordinate system."""
    x1, y1, x2, y2 = box
    x1, x2 = (x1 - dx) / scale - pad_w, (x2 - dx) / scale - pad_w
    y1, y2 = (y1 - dy) / scale - pad_h, (y2 - dy) / scale - pad_h
    return [int(coord) for coord in [x1, y1, x2, y2]]

if __name__ == "__main__":
    setup_korean_font()
    input_folder = './3_modern_plates'
    # Define separate output folders for clarity
    output_folder_cropped = 'results_cropped'
    output_folder_viz = 'results_viz'
    
    try:
        model = YOLO('best.pt')
        print("[Info] YOLOv8 model 'best.pt' loaded successfully.")
    except Exception as e:
        print(f"[Fatal Error] Could not load YOLOv8 model. Error: {e}"); exit()

    if not os.path.isdir(input_folder):
        print(f"[Fatal Error] Input folder not found at '{input_folder}'."); exit()
    os.makedirs(output_folder_cropped, exist_ok=True)
    os.makedirs(output_folder_viz, exist_ok=True)

    print("\n" + "="*50)
    print("🚀 Starting Batch Processing Session 🚀")
    print(f"👉 Reading from: '{input_folder}'")
    print(f"👉 Saving cropped plates to: '{output_folder_cropped}'")
    print(f"👉 Saving visualizations to: '{output_folder_viz}'")
    print("="*50 + "\n")
    
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            
        print(f"\n--- Processing file: {filename} ---")
        img_path = os.path.join(input_folder, filename)
        original_image = cv2.imread(img_path)
        if original_image is None:
            print(f"  - [Error] Failed to read image file."); continue

        prepared_img, scale, dx, dy, pad_w, pad_h = prepare_image_for_yolo(original_image, YOLO_INPUT_SIZE)
        results = model(prepared_img, conf=CONF_THRESHOLD)
        
        final_plate_crop = None
        img_to_show = original_image.copy()
        
        if len(results[0].boxes) > 0:
            best_box = results[0].boxes[0]
            yolo_coords = best_box.xyxy[0].cpu().numpy()
            
            orig_coords = convert_coords_from_yolo_to_original(yolo_coords, scale, dx, dy, pad_w, pad_h)
            x1, y1, x2, y2 = orig_coords
            
            # Aspect Ratio Filter
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                aspect_ratio = max(w, h) / min(w, h)
                if aspect_ratio < MIN_ASPECT_RATIO_FILTER:
                    print(f"  - [Info] Detection skipped. Aspect ratio {aspect_ratio:.2f} is below threshold {MIN_ASPECT_RATIO_FILTER}.")
                else:
                    print(f"  - [Info] Detection valid. Aspect ratio: {aspect_ratio:.2f}")
                    # Crop the final plate from the original image
                    final_plate_crop = original_image[y1:y2, x1:x2]
                    # Draw detection box for visualization
                    cv2.rectangle(img_to_show, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    conf_score = best_box.conf[0].item()
                    cv2.putText(img_to_show, f"Plate: {conf_score:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                 print("  - [Warning] Invalid box dimensions detected (width or height is zero).")
        else:
            print("  - [Info] No license plate detected in this image.")

        # --- Save Cropped Image ---
        if final_plate_crop is not None and final_plate_crop.size > 0:
            save_path_crop = os.path.join(output_folder_cropped, os.path.splitext(filename)[0] + ".png")
            cv2.imwrite(save_path_crop, final_plate_crop)
            print(f"  - [Success] Saved cropped plate to '{save_path_crop}'")
        else:
            print(f"  - [Failure] No valid plate was cropped.")

        # --- Create and Save Visualization Figure ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'File: {filename}', fontsize=16)
        
        axes[0].imshow(cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB))
        axes[0].set_title('YOLOv8 Detection Result')
        axes[0].axis('off')

        if final_plate_crop is not None and final_plate_crop.size > 0:
            axes[1].imshow(cv2.cvtColor(final_plate_crop, cv2.COLOR_BGR2RGB))
            axes[1].set_title('Cropped Plate')
        else:
            axes[1].text(0.5, 0.5, 'Detection Failed', ha='center', va='center', fontsize=14, color='red')
            axes[1].set_title('Result')
        axes[1].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the visualization figure
        save_path_viz = os.path.join(output_folder_viz, os.path.splitext(filename)[0] + "_viz.png")
        plt.savefig(save_path_viz)
        plt.close(fig) # Close the figure to free up memory

    print("\n🎉 All images have been processed automatically.")