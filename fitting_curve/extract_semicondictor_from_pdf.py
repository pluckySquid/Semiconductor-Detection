import cv2
import numpy as np
import os

# --------------------- CONFIGURATION ---------------------
INPUT_IMAGE_PATH   = "../data/test.png"  # Replace with your input image path
OUTPUT_IMAGE_PATH  = "../data/test/annotated_image_valid_boxes.png"
OUTPUT_DIR         = "../data/test/components_valid_boxes"

THRESHOLD_VALUE    = 200   # Initial threshold for binarization
MORPH_KERNEL_SIZE  = (3,3) # Kernel size for morphological operations
MIN_COMPONENT_AREA = 10    # Minimum area to accept a connected component
# ---------------------------------------------------------

def preprocess_image(image_path):
    """
    1. Read image in BGR
    2. Convert to grayscale
    3. Threshold (binary_inv so that objects are black=0, background=255)
    4. Perform morphological close to 'unify' broken lines
    :param image_path: str, path to image file
    :return: (binary_image, original_bgr)
    """
    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

    # Threshold so black lines/shapes become 0, background is 255
    _, binary_inv = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)

    # Morphological close (to fill small gaps/holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    closed = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel)

    return closed, original_bgr

def refine_bounding_box(binary_image, x, y, w, h):
    """
    Tighten the bounding box so it encloses all black pixels (== 0).
    1. Look at the sub-image for the bounding box
    2. Find the min/max row/col that contain black pixels
    3. Return a refined bounding box
    :param binary_image: np.ndarray (0=black, 255=white)
    :param x, y, w, h: bounding box from connectedComponents
    :return: (x_new, y_new, w_new, h_new) or None if no black pixels found
    """
    component = binary_image[y:y+h, x:x+w]
    # Locate all black pixel indices
    black_pts = np.where(component == 0)  # returns (rows, cols)
    if black_pts[0].size == 0:
        return None  # no black pixels
    
    min_row = np.min(black_pts[0])
    max_row = np.max(black_pts[0])
    min_col = np.min(black_pts[1])
    max_col = np.max(black_pts[1])
    
    # Convert local coords -> global coords
    new_x = x + min_col
    new_y = y + min_row
    new_w = (max_col - min_col) + 1
    new_h = (max_row - min_row) + 1
    
    return (new_x, new_y, new_w, new_h)

def edges_have_black_pixels(binary_image, x, y, w, h):
    """
    Optional check: ensure each edge (top, bottom, left, right)
    in the bounding box has at least one black pixel.
    :param binary_image: 2D array (0=black, 255=white)
    :param x, y, w, h: bounding box
    :return: bool
    """
    # Extract region
    roi = binary_image[y:y+h, x:x+w]
    # Edges
    top_edge    = roi[0, :]
    bottom_edge = roi[-1, :]
    left_edge   = roi[:, 0]
    right_edge  = roi[:, -1]

    if (0 in top_edge and
        0 in bottom_edge and
        0 in left_edge and
        0 in right_edge):
        return True
    return False

def find_tight_bounding_boxes(binary_image):
    """
    1. Use connectedComponentsWithStats to find all components
    2. For each, refine bounding box
    3. (Optional) Check area, edge blackness, etc.
    :param binary_image: 0=black, 255=white
    :return: list of bounding boxes (x, y, w, h)
    """
    # connectivity=8 to consider diagonal adjacency
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    valid_boxes = []
    for label_id in range(1, num_labels):  # skip label 0 (background)
        x, y, w, h, area = stats[label_id]

        # Skip too-small components
        if area < MIN_COMPONENT_AREA:
            continue
        
        # Refine bounding box to *exactly* enclose black pixels
        refined = refine_bounding_box(binary_image, x, y, w, h)
        if not refined:
            continue
        rx, ry, rw, rh = refined
        
        # Optionally ensure edges have black pixels
        if edges_have_black_pixels(binary_image, rx, ry, rw, rh):
            valid_boxes.append((rx, ry, rw, rh))
    
    return valid_boxes

def extract_and_draw_valid_boxes(binary_image, original_image, output_image_path, output_dir):
    """
    Finds bounding boxes and saves them + annotated image.
    :param binary_image: np.ndarray (0=black, 255=white)
    :param original_image: BGR image
    :param output_image_path: path for final annotated image
    :param output_dir: folder for extracted component images
    """
    os.makedirs(output_dir, exist_ok=True)

    # Copy for annotation
    annotated = original_image.copy()

    boxes = find_tight_bounding_boxes(binary_image)
    component_count = 0
    for (x, y, w, h) in boxes:
        # Draw bounding box in green
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract region
        roi_bgr = original_image[y:y+h, x:x+w]
        out_path = os.path.join(output_dir, f"component_{component_count}.png")
        cv2.imwrite(out_path, roi_bgr)
        component_count += 1

    cv2.imwrite(output_image_path, annotated)
    print(f"[INFO] Saved annotated image to: {output_image_path}")
    print(f"[INFO] Extracted {component_count} components to: {output_dir}")

def main():
    binary_img, original_bgr = preprocess_image(INPUT_IMAGE_PATH)
    extract_and_draw_valid_boxes(binary_img, original_bgr, OUTPUT_IMAGE_PATH, OUTPUT_DIR)

if __name__ == "__main__":
    main()
