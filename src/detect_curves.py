import os
import cv2
import numpy as np
import shutil
import time
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Parameters
IMAGE_PATH = '../data/cleaned_page_0.png'
# Update the TEMPLATES_DIR to point to the unique templates
TEMPLATES_DIR = '../data/auto_extract/unique/'
TEMPLATES_DIR = '../data/test/components_valid_boxes/'
OUTPUT_IMAGE_PATH = '../data/detected_curves.png'
CROPPED_X1, CROPPED_Y1 = 1000, 1000  # Adjust as needed
CROPPED_X2, CROPPED_Y2 = 8000, 5000
DPI = 100
THRESHOLD = 0.75  # Similarity threshold for template matching
NMS_THRESHOLD = 0.5  # IoU threshold for Non-Maximum Suppression
ROTATION_ANGLES = [0, 90, 180, 270]  # Degrees to rotate templates

# Function to load templates and generate rotated versions
def load_templates(templates_dir, rotation_angles):
    templates = []
    if not os.path.exists(templates_dir):
        print(f"Error: Templates directory '{templates_dir}' does not exist.")
        return templates

    for filename in os.listdir(templates_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            template_path = os.path.join(templates_dir, filename)
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template is None:
                print(f"Warning: Could not read template {template_path}. Skipping.")
                continue

            # Convert to grayscale
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            # Apply histogram equalization and thresholding
            equalized = cv2.equalizeHist(template_gray)
            _, template_thresh = cv2.threshold(equalized, 254, 255, cv2.THRESH_BINARY)

            template_type = os.path.splitext(filename)[0]
            # Generate rotated versions
            for angle in rotation_angles:
                rotated = rotate_image(template_thresh, angle)
                if rotated is not None:
                    templates.append({
                        'type': template_type,
                        'angle': angle,
                        'image': rotated
                    })
                    print(f"Loaded template '{template_type}' rotated by {angle}° with size {rotated.shape}")
                else:
                    print(f"Warning: Rotation angle {angle}° is not supported.")
    return templates

# Function to rotate an image by a given angle
def rotate_image(image, angle):
    if angle == 0:
        return image.copy()
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # For arbitrary angles, use warpAffine (not used here)
        print(f"Unsupported rotation angle: {angle}")
        return None

# Function for Non-Maximum Suppression
def non_max_suppression(detections, iou_threshold):
    if len(detections) == 0:
        return []

    # Convert detections to array format
    boxes = np.array([det['rect'] for det in detections])
    scores = np.array([det['score'] for det in detections])

    # Coordinates of bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # Compute areas
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort by scores descending
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(detections[i])

        # Compute IoU of the kept box with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Compute width and height of overlap
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute overlap ratio (IoU)
        overlap = (w * h) / areas[order[1:]]

        # Indices of boxes with overlap less than threshold
        inds = np.where(overlap <= iou_threshold)[0]

        # Update order
        order = order[inds + 1]

    return keep

def main():
    # Load and preprocess the main image
    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {IMAGE_PATH}")

    # Crop the image without resizing
    cropped_image = image[CROPPED_Y1:CROPPED_Y2, CROPPED_X1:CROPPED_X2]
    original_height, original_width = cropped_image.shape

    # Calculate figsize based on the original resolution and desired dpi
    figsize = (original_width / DPI, original_height / DPI)

    # Enhance contrast using histogram equalization
    equalized = cv2.equalizeHist(cropped_image)

    # Apply binary thresholding
    _, thresh = cv2.threshold(equalized, 254, 255, cv2.THRESH_BINARY)

    # Optional: Apply morphological operations to reduce noise
    # kernel = np.ones((3,3), np.uint8)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Load templates from the unique directory with rotated versions
    templates = load_templates(TEMPLATES_DIR, ROTATION_ANGLES)
    if not templates:
        raise ValueError(f"No valid templates found in {TEMPLATES_DIR}")

    # Perform template matching
    detections = []
    for template in templates:
        template_image = template['image']
        template_h, template_w = template_image.shape

        if template_h == 0 or template_w == 0:
            print(f"Warning: Template '{template['type']}' with rotation {template['angle']}° has invalid dimensions. Skipping.")
            continue

        # Perform template matching
        res = cv2.matchTemplate(thresh, template_image, cv2.TM_CCOEFF_NORMED)

        # Find locations where the matching result exceeds the threshold
        loc = np.where(res >= THRESHOLD)

        print(f"Matching template '{template['type']}' rotated by {template['angle']}° found {len(loc[0])} potential matches with size ({template_w}x{template_h}).")

        for pt in zip(*loc[::-1]):  # Switch x and y
            detections.append({
                'type': template['type'],
                'angle': template['angle'],
                'rect': [pt[0], pt[1], pt[0] + template_w, pt[1] + template_h],
                'score': res[pt[1], pt[0]]
            })

    print(f"Total raw detections before NMS: {len(detections)}")

    # Apply Non-Maximum Suppression
    final_detections = non_max_suppression(detections, NMS_THRESHOLD)
    print(f"Detections after Non-Maximum Suppression: {len(final_detections)}")

    # Visualization
    # Convert cropped image to BGR for colored annotations
    output_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)

    for det in final_detections:
        x1, y1, x2, y2 = det['rect']
        curve_type = det['type']
        angle = det['angle']
        score = det['score']

        # Choose a color based on curve type (consistent coloring)
        np.random.seed(abs(hash(curve_type)) % (2**32))
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # Draw rectangle
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

        # Put text label with rotation angle and score
        label = f"{curve_type} ({angle}) {score:.2f}"
        cv2.putText(output_image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save the output image without resizing
    success = cv2.imwrite(OUTPUT_IMAGE_PATH, output_image)
    if success:
        print(f"Detected curves have been saved to {OUTPUT_IMAGE_PATH}")
    else:
        print(f"Failed to save the detected curves image to {OUTPUT_IMAGE_PATH}")

    # Display the result using matplotlib
    plt.figure(figsize=figsize, dpi=DPI)
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Curves')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
