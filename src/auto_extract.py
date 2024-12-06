import os
import cv2
import numpy as np
import shutil
import time
from skimage.metrics import structural_similarity as ssim

def follow_line(binary_img, visited, x, y, direction):
    """Follow the line in a given direction until the slope changes or the line ends."""
    height, width = binary_img.shape
    dx, dy = direction  # Direction to follow
    line_pixels = [(x, y)]  # Store all the pixels in this line
    visited[x, y] = True  # Mark as visited

    while True:
        nx, ny = x + dx, y + dy  # Move to the next pixel in the same direction
        if not (0 <= nx < height and 0 <= ny < width):  # Check bounds
            break  # Out of bounds
        if binary_img[nx, ny] == 255 or visited[nx, ny]:  # Stop if not black or already visited
            break

        visited[nx, ny] = True  # Mark the pixel as visited
        line_pixels.append((nx, ny))  # Add pixel to the current line
        x, y = nx, ny  # Move to the next pixel

    return line_pixels

def find_straight_lines(binary_img, output_image):
    """Detect and color all lines in the binary image."""
    height, width = binary_img.shape
    visited = np.zeros((height, width), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    line_count = 0
    curve_count = 0

    for x in range(height):
        for y in range(width):
            if binary_img[x, y] == 0 and not visited[x, y]:
                # Check all directions from this pixel
                for direction in directions:
                    line_pixels = follow_line(binary_img, visited, x, y, direction)

                    if len(line_pixels) >= 8:  # Valid straight line
                        color = [0, 255, 0]  # Green for straight lines
                        line_count += 1
                    else:  # If not a straight line, treat it as a curve
                        color = [255, 0, 0]  # Blue for curved lines
                        curve_count += 1

                    # Mark the line/curve pixels with the chosen color
                    for px, py in line_pixels:
                        output_image[px, py] = color

                    # Mark the endpoints of straight lines in yellow
                    if color == [0, 255, 0]:  # Only for straight lines
                        start_px, start_py = line_pixels[0]
                        end_px, end_py = line_pixels[-1]
                        output_image[start_px, start_py] = [0, 255, 255]  # Yellow start point
                        output_image[end_px, end_py] = [0, 255, 255]  # Yellow end point

    print(f"Number of straight lines detected: {line_count}")
    print(f"Number of curved lines detected: {curve_count}")
    return output_image

def find_island_from_pixel(output_image, visited, x, y):
    """Use DFS to find all connected blue pixels forming an island."""
    stack = [(x, y)]
    island_pixels = []
    visited[x, y] = True

    while stack:
        cx, cy = stack.pop()
        island_pixels.append((cx, cy))
        
        # Check 8 neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < output_image.shape[0] and 0 <= ny < output_image.shape[1]
                    and output_image[nx, ny].tolist() == [255, 0, 0]  # Check for blue pixel
                    and not visited[nx, ny]):
                visited[nx, ny] = True
                stack.append((nx, ny))

    return island_pixels

def save_island_image(island_pixels, output_dir, index):
    """Create an image of the island, crop to bounding box, and save it."""
    if not island_pixels:
        print(f"Island {index} has no pixels. Skipping save.")
        return None
    
    # Extract the x and y coordinates of the island pixels
    x_coords = [px for px, _ in island_pixels]
    y_coords = [py for _, py in island_pixels]
    
    # Define the bounding box around the island
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # Create a blank white image for the bounding box of the island
    island_img = np.ones((max_x - min_x + 1, max_y - min_y + 1, 3), dtype=np.uint8) * 255
    
    # Place the island pixels into the bounding box image
    for px, py in island_pixels:
        island_img[px - min_x, py - min_y] = [0, 0, 0]  # Mark pixels in black for the island

    # Save the cropped island image
    output_path = os.path.join(output_dir, f"{index}.png")
    success = cv2.imwrite(output_path, island_img)
    
    if success and os.path.exists(output_path):
        print(f"Island image saved at: {output_path} (Size: {island_img.shape})")
        return output_path
    else:
        print(f"Failed to save island image at: {output_path}")
        return None

def is_duplicate(img1, img2):
    """Check if two images are duplicates by resizing and comparing SSIM."""
    try:
        # Resize both images to a fixed size
        img1_resized = cv2.resize(img1, (50, 50))
        img2_resized = cv2.resize(img2, (50, 50))
        
        # Convert images to grayscale
        img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        score, _ = ssim(img1_gray, img2_gray, full=True)
        return score > 0.9  # Threshold for similarity
    except Exception as e:
        print(f"Error in is_duplicate: {e}")
        return False

def categorize_islands(island_paths, unique_dir, duplicate_dir):
    """Compare each island to categorize into unique or duplicate."""
    unique_images = {}
    for index, path in enumerate(island_paths):
        if not os.path.exists(path):
            print(f"File path does not exist: {path}. Skipping.")
            continue

        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Unable to read image at path {path}. Skipping.")
            continue

        duplicate_found = False

        for unique_img_name, unique_img_path in unique_images.items():
            unique_img = cv2.imread(unique_img_path)
            if unique_img is None:
                print(f"Warning: Unable to read unique image at path {unique_img_path}. Skipping comparison.")
                continue

            if is_duplicate(img, unique_img):
                duplicate_found = True
                # Generate a unique name with suffix
                suffix = 1
                base_name = os.path.splitext(unique_img_name)[0]
                while True:
                    duplicate_name = f"{base_name}_{suffix}.png"
                    duplicate_path = os.path.join(duplicate_dir, duplicate_name)
                    if not os.path.exists(duplicate_path):
                        break
                    suffix += 1
                # Move duplicate image to the duplicate directory
                try:
                    shutil.move(path, duplicate_path)
                    print(f"Duplicate found: {path} moved to {duplicate_dir} as {duplicate_name}")
                except Exception as e:
                    print(f"Error moving duplicate {path} to {duplicate_dir}: {e}")
                break

        if not duplicate_found:
            # No duplicates found; move to unique directory
            unique_name = os.path.basename(path)
            unique_dest_path = os.path.join(unique_dir, unique_name)
            try:
                shutil.move(path, unique_dest_path)
                unique_images[unique_name] = unique_dest_path
                print(f"Unique image: {path} moved to {unique_dir} as {unique_name}")
            except Exception as e:
                print(f"Error moving unique {path} to {unique_dir}: {e}")

def find_and_save_curved_islands(output_image, output_dir):
    """Detect and save each island of curved blue lines starting from blue or yellow points."""
    height, width, _ = output_image.shape
    visited = np.zeros((height, width), dtype=bool)
    curve_count = 0
    island_paths = []

    os.makedirs(output_dir, exist_ok=True)

    for x in range(height):
        for y in range(width):
            # Start a new island search if we find a blue or yellow pixel and it hasn't been visited
            if not visited[x, y] and (output_image[x, y].tolist() == [255, 0, 0] or output_image[x, y].tolist() == [0, 255, 255]):
                island_pixels = find_island_from_pixel(output_image, visited, x, y)
                
                # Only save if the island has more than 30 blue pixels
                if len(island_pixels) > 30:
                    path = save_island_image(island_pixels, output_dir, curve_count)
                    if path:  # Only add if the file was successfully saved
                        island_paths.append(path)
                    curve_count += 1  # Increment the island index

    print(f"Number of curved line islands saved: {curve_count}")
    return island_paths

# Main execution
def main():
    # Get the root directory of the project
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Define the path to the data folder
    data_folder = os.path.join(root_dir, 'data')

    # Ensure the data folder exists
    os.makedirs(data_folder, exist_ok=True)
    print("data_folder:", data_folder)


    # Define paths relative to the data folder
    image_path = os.path.join(data_folder, 'cleaned_page_0.png')
    output_dir = os.path.join(data_folder, 'pixel_by_pixel_output')
    auto_extract_dir = os.path.join(data_folder, 'auto_extract')
    temp_islands_dir = os.path.join(auto_extract_dir, 'temp_islands')
    unique_dir = os.path.join(auto_extract_dir, 'unique')
    duplicate_dir = os.path.join(auto_extract_dir, 'duplicates')

    # Ensure all directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_islands_dir, exist_ok=True)
    os.makedirs(unique_dir, exist_ok=True)
    os.makedirs(duplicate_dir, exist_ok=True)

    # Load and process the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read the image file at {image_path}. Please check the path.")
        return

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)

    # Detect and mark straight and curved lines in the image
    output_image = find_straight_lines(binary_img, img.copy())

    # Detect and save islands of curved lines to temp_islands_dir
    island_paths = find_and_save_curved_islands(output_image, temp_islands_dir)

    if not island_paths:
        print("No islands were found to categorize.")
        return

    # Categorize islands into unique and duplicate
    categorize_islands(island_paths, unique_dir, duplicate_dir)

    # Optionally, you can move or delete the temp_islands_dir after categorization
    # shutil.rmtree(temp_islands_dir)

    # Save the processed output image with marked lines
    output_path = os.path.join(output_dir, 'islands_with_straight_and_curved_lines.png')
    success = cv2.imwrite(output_path, output_image)
    if success:
        print(f"Final output image saved at: {output_path}")
    else:
        print(f"Failed to save the final output image at: {output_path}")

if __name__ == "__main__":
    main()
