import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

# Increase the maximum image pixel size to prevent DecompressionBombError
Image.MAX_IMAGE_PIXELS = None  # Allow images of any size

def pdf_to_image(pdf_file, output_image_path, dpi=300):
    pages = convert_from_path(pdf_file, dpi=dpi)
    output_dir = os.path.dirname(output_image_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pages[0].save(output_image_path, 'PNG')
    return output_image_path

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    original_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply advanced noise reduction
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # Enhance edges using adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)

    # Use adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 10)

    # Morphological operations to enhance features
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Edge detection with adjusted thresholds
    edges = cv2.Canny(morph, 50, 150, apertureSize=3)

    return edges, original_img

def detect_straight_lines(edges, original_img):
    # Use the Line Segment Detector
    lsd = cv2.createLineSegmentDetector(0)
    lines, width, prec, nfa = lsd.detect(edges)

    line_img = np.zeros_like(original_img)
    straight_lines_drawn = original_img.copy()

    if lines is not None:
        print(f"Number of straight lines detected: {len(lines)}")
        for line in lines:
            x0, y0, x1, y1 = map(int, line[0])
            # Draw lines on black background for masking
            cv2.line(line_img, (x0, y0), (x1, y1), (255, 255, 255), 2)
            # Draw lines on original image for visualization
            cv2.line(straight_lines_drawn, (x0, y0), (x1, y1), (0, 255, 0), 2)
    else:
        print("No straight lines detected")
    return line_img, straight_lines_drawn

def remove_straight_lines(edges, line_img):
    # Convert line image to grayscale if necessary
    if len(line_img.shape) == 3:
        gray_lines = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_lines = line_img
    # Invert the line image
    inverted_lines = cv2.bitwise_not(gray_lines)
    # Mask the edges image to remove straight lines
    edges_without_lines = cv2.bitwise_and(edges, edges, mask=inverted_lines)
    return edges_without_lines

def detect_curved_lines(edges, original_img):
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    curve_img = original_img.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # Adjust as needed
            continue

        # Fit an ellipse if possible
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            # Calculate aspect ratio to filter out elongated shapes
            (center, axes, angle) = ellipse
            major_axis_length = max(axes)
            minor_axis_length = min(axes)
            if minor_axis_length == 0:
                continue
            aspect_ratio = major_axis_length / minor_axis_length
            if aspect_ratio < 5:
                cv2.ellipse(curve_img, ellipse, (0, 0, 255), 2)
    return curve_img

def process_image(image_path, output_dir):
    edges, original_img = preprocess_image(image_path)
    # Save the edges image
    cv2.imwrite(os.path.join(output_dir, 'edges.png'), edges)

    # Detect straight lines
    line_img, straight_lines_drawn = detect_straight_lines(edges, original_img)
    cv2.imwrite(os.path.join(output_dir, 'straight_lines.png'), straight_lines_drawn)

    # Remove straight lines from edges
    edges_no_lines = remove_straight_lines(edges, line_img)
    cv2.imwrite(os.path.join(output_dir, 'edges_no_straight_lines.png'), edges_no_lines)

    # Detect curved lines
    curved_lines_img = detect_curved_lines(edges_no_lines, original_img)
    cv2.imwrite(os.path.join(output_dir, 'curved_lines.png'), curved_lines_img)

    # Combine straight and curved lines
    combined_img = original_img.copy()
    # Overlay straight lines in green
    combined_img = cv2.addWeighted(combined_img, 1, straight_lines_drawn, 1, 0)
    # Overlay curved lines in red
    combined_img = cv2.addWeighted(combined_img, 1, curved_lines_img, 1, 0)
    cv2.imwrite(os.path.join(output_dir, 'combined_lines.png'), combined_img)

    print(f"Processed images saved in {output_dir}")

def process_pdf_to_image_and_detect_lines(pdf_file, output_dir, dpi=300):
    output_image_path = os.path.join(output_dir, 'converted_image.png')
    image_path = pdf_to_image(pdf_file, output_image_path, dpi)
    process_image(image_path, output_dir)

# Example usage
pdf_file = '/mnt/e/Work/Virginia/straight_line_detection/BZ_50D2AA (9-15-2023).pdf'  # Correct path to PDF file
output_dir = '/mnt/e/Work/Virginia/straight_line_detection/image'  # Output directory for the images

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process the PDF to image and detect lines
process_pdf_to_image_and_detect_lines(pdf_file, output_dir, dpi=300)
