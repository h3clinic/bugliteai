import cv2
import os
import glob
import numpy as np

# --- CONFIGURATION ---
# Update these to point to your Grounding DINO output folders
IMAGES_DIR = r"GD_biting_flies_images" 
LABELS_DIR = r"GD_biting_flies_labels"

# Filter classes? (Leave empty to show all)
# If you only want to see class 0 ('insect'), set VALID_CLASSES = [0]
VALID_CLASSES = [] 

# Visualization Settings
MAX_DISPLAY_WIDTH = 1600  # Max width of the window in pixels
MAX_DISPLAY_HEIGHT = 900  # Max height of the window in pixels

def load_yolo_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = list(map(float, line.strip().split()))
            if len(parts) >= 5:
                # YOLO format: class x_center y_center w h
                cls = int(parts[0])
                x, y, w, h = parts[1:5]
                boxes.append([cls, x, y, w, h])
    return boxes

def draw_boxes(img, boxes, color=(0, 255, 0), thickness=2):
    h_img, w_img, _ = img.shape
    for box in boxes:
        cls, x, y, w, h = box
        
        if VALID_CLASSES and cls not in VALID_CLASSES:
            continue

        # Convert normalized to pixel coords
        x1 = int((x - w/2) * w_img)
        y1 = int((y - h/2) * h_img)
        x2 = int((x + w/2) * w_img)
        y2 = int((y + h/2) * h_img)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, f"ID:{cls}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

def nms(boxes, iou_threshold=0.5):
    """ Simple Non-Maximum Suppression to merge overlapping boxes """
    if not boxes: return []
    
    # Convert to x1, y1, x2, y2 format for NMS calculation
    # We ignore the class ID for merging since they are synonyms (spider vs insect)
    # Storing as [x1, y1, x2, y2, original_index]
    box_list_coords = []
    for i, b in enumerate(boxes):
        cls, x, y, w, h = b
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        box_list_coords.append([x1, y1, x2, y2, i])

    box_list_coords = sorted(box_list_coords, key=lambda x: x[0]) # Sort doesn't matter much for simple NMS
    
    keep_indices = []
    
    while box_list_coords:
        current = box_list_coords.pop(0)
        keep_indices.append(current[4])
        
        # Compare current against rest
        remaining = []
        for candidate in box_list_coords:
            # Calculate IoU
            xx1 = max(current[0], candidate[0])
            yy1 = max(current[1], candidate[1])
            xx2 = min(current[2], candidate[2])
            yy2 = min(current[3], candidate[3])
            
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            
            inter = w * h
            area1 = (current[2]-current[0]) * (current[3]-current[1])
            area2 = (candidate[2]-candidate[0]) * (candidate[3]-candidate[1])
            iou = inter / (area1 + area2 - inter)
            
            # If IoU is low (not overlapping), keep it
            if iou < iou_threshold:
                remaining.append(candidate)
        
        box_list_coords = remaining

    return [boxes[i] for i in keep_indices]

def main():
    image_files = glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))
    if not image_files:
        print("No images found! Check path.")
        return

    print(f"Found {len(image_files)} images.")
    print("Controls: [D] Next Image, [A] Previous Image, [Q] Quit")

    idx = 0
    while True:
        img_path = image_files[idx]
        basename = os.path.basename(img_path)
        txt_name = basename.replace(".jpg", ".txt")
        label_path = os.path.join(LABELS_DIR, txt_name)

        # Load Image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load {basename}")
            idx += 1
            continue

        # Load Raw Boxes
        raw_boxes = load_yolo_labels(label_path)
        
        # Create Merged Boxes (NMS)
        clean_boxes = nms(raw_boxes, iou_threshold=0.7)

        # Draw Side-by-Side Comparison
        # Left: Raw (All 6 classes) | Right: Cleaned (merged)
        img_raw = img.copy()
        img_clean = img.copy()
        
        img_raw = draw_boxes(img_raw, raw_boxes, color=(0, 0, 255)) # Red
        img_clean = draw_boxes(img_clean, clean_boxes, color=(0, 255, 0)) # Green
        
        # Combine
        combined = np.hstack((img_raw, img_clean))
        
        # Add Text Labels
        # Scale font size based on image resolution so it's readable
        font_scale = max(0.5, img.shape[1] / 1000)
        thickness = max(1, int(img.shape[1] / 500))
        
        cv2.putText(combined, f"Raw: {len(raw_boxes)} boxes", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), thickness)
        cv2.putText(combined, f"Cleaned (NMS): {len(clean_boxes)} boxes", (img.shape[1] + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness)
        cv2.putText(combined, f"{idx+1}/{len(image_files)}: {basename}", (20, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.7, (255,255,255), thickness)

        # --- RESIZE FOR DISPLAY ---
        h, w = combined.shape[:2]
        scale = 1.0
        
        # Check restrictions
        if w > MAX_DISPLAY_WIDTH:
            scale = min(scale, MAX_DISPLAY_WIDTH / w)
        if h > MAX_DISPLAY_HEIGHT:
            scale = min(scale, MAX_DISPLAY_HEIGHT / h)
            
        if scale < 1.0:
            new_dim = (int(w * scale), int(h * scale))
            combined_display = cv2.resize(combined, new_dim)
        else:
            combined_display = combined

        # Show
        cv2.imshow("Grounding DINO Inspector", combined_display)

        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('d'): # Next
            idx = (idx + 1) % len(image_files)
        elif key == ord('a'): # Prev
            idx = (idx - 1) % len(image_files)
        elif key == ord('q'): # Quit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()  