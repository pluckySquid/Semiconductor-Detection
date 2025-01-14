import cv2
import os
import numpy as np

def load_manual_templates(manual_dir):
    """
    Load all manually saved semiconductor templates from a directory.
    :param manual_dir: Path to the directory containing manual semiconductor images
    :return: List of templates with their filenames
    """
    templates = []
    for filename in os.listdir(manual_dir):
        filepath = os.path.join(manual_dir, filename)
        if filename.endswith(".png") or filename.endswith(".jpg"):
            template = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            templates.append((template, filename))
    print(f"Loaded {len(templates)} templates from {manual_dir}")
    return templates

def identify_semiconductors(target_image_path, templates, output_dir, match_threshold=0.8):
    """
    Identify semiconductors in the target image using template matching.
    :param target_image_path: Path to the cleaned image
    :param templates: List of templates (images) with their filenames
    :param output_dir: Directory to save matched semiconductor regions
    :param match_threshold: Threshold for template matching (0 to 1)
    """
    # Load the target image
    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    annotated_image = cv2.imread(target_image_path)  # For drawing bounding boxes

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    semiconductor_counts = {}  # Dictionary to store counts for each template
    total_matches = 0

    for template, template_name in templates:
        # Perform template matching
        result = cv2.matchTemplate(target_image, template, cv2.TM_CCOEFF_NORMED)

        # Get locations where matching exceeds the threshold
        locations = np.where(result >= match_threshold)
        match_count = 0

        for pt in zip(*locations[::-1]):  # Switch x and y coordinates
            h, w = template.shape

            # Draw a rectangle around the matched region on the annotated image
            cv2.rectangle(annotated_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

            # Save the matched region
            matched_region = target_image[pt[1]:pt[1] + h, pt[0]:pt[0] + w]
            match_path = os.path.join(output_dir, f"match_{total_matches}_{template_name}")
            cv2.imwrite(match_path, matched_region)

            match_count += 1
            total_matches += 1

        # Update the count for this template
        semiconductor_counts[template_name] = match_count

    # Save the annotated image with bounding boxes
    annotated_image_path = os.path.join(output_dir, "annotated_image.png")
    cv2.imwrite(annotated_image_path, annotated_image)
    print(f"Annotated image saved to {annotated_image_path}")

    # Print the count for each semiconductor template
    for template_name, count in semiconductor_counts.items():
        print(f"Template '{template_name}' identified {count} matches.")

    return semiconductor_counts

if __name__ == "__main__":
    # Directory containing manually saved semiconductors
    manual_dir = "../data/manual"

    # Path to the cleaned target image
    target_image_path = "../data/cleaned_page_0.png"

    # Directory to save the output
    output_dir = "../data/matched_semiconductors"

    # Load manual templates
    templates = load_manual_templates(manual_dir)

    # Identify semiconductors in the target image
    semiconductor_counts = identify_semiconductors(target_image_path, templates, output_dir, match_threshold=0.8)

    # Print total matches
    print(f"Total semiconductors identified: {sum(semiconductor_counts.values())}")
