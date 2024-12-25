import re
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def remove_letters_and_numbers(pdf_path, output_path):
    """
    Detect and remove (via redaction) all words that contain letters (A-Z, a-z) or digits (0-9).
    
    Args:
        pdf_path (str): Path to the input PDF file.
        output_path (str): Path to save the redacted PDF file.
    """
    # Compile a pattern to match any letter or digit
    pattern = re.compile(r'[A-Za-z0-9]')

    doc = fitz.open(pdf_path)

    for page in doc:
        # Extract words: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        words = page.get_text("words")

        for w in words:
            x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
            # Check if the text contains letters or digits
            if pattern.search(text):
                rect = fitz.Rect(x0, y0, x1, y1)
                page.add_redact_annot(rect)

        # Apply all redactions on this page
        page.apply_redactions()

    doc.save(output_path)
    doc.close()
    print(f"[INFO] Letters and numbers removed from PDF. Saved: {output_path}")


def plot_bezier_curve(start, control_points, end, n_points=100):
    """
    Computes points on a cubic Bézier curve.

    Args:
        start (tuple): Starting point (x0, y0).
        control_points (list): List of two control points [(x1, y1), (x2, y2)].
        end (tuple): Ending point (x3, y3).
        n_points (int): Number of points to compute on the curve.

    Returns:
        tuple: Arrays of x and y coordinates of the curve.
    """
    t = np.linspace(0, 1, n_points)
    start = np.array(start)
    control1 = np.array(control_points[0])
    control2 = np.array(control_points[1])
    end = np.array(end)
    
    # Cubic Bézier formula
    bezier_curve = (
        ((1 - t)**3)[:, np.newaxis] * start +
        3 * ((1 - t)**2)[:, np.newaxis] * (t[:, np.newaxis]) * control1 +
        3 * ((1 - t))[:, np.newaxis] * (t**2)[:, np.newaxis] * control2 +
        (t**3)[:, np.newaxis] * end
    )
    return bezier_curve[:, 0], bezier_curve[:, 1]


def ensure_directory_exists(directory_path):
    """
    Ensures that the specified directory exists. If not, creates it.

    Args:
        directory_path (str): Path to the directory.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"[INFO] Created directory: {directory_path}")


def extract_vector_data_pymupdf(pdf_path):
    """
    Extracts vector data (lines, cubic Bézier curves, quadrilaterals) from a PDF using PyMuPDF.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: A list of dictionaries containing vector data.
    """
    doc = fitz.open(pdf_path)
    vector_data = []

    def get_coordinates(point):
        """
        Converts a Point object or a tuple/list to a list of coordinates.
        
        Args:
            point (object): A Point object with .x and .y or a tuple/list with two elements.
        
        Returns:
            list: [x, y]
        """
        if hasattr(point, 'x') and hasattr(point, 'y'):
            return [point.x, point.y]
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            return [point[0], point[1]]
        else:
            raise ValueError("Invalid point format")

    for page_number in range(len(doc)):
        page = doc[page_number]
        drawings = page.get_drawings()

        for drawing in drawings:
            for item in drawing["items"]:
                op = item[0]
                operands = item[1:]
                try:
                    if op == "l":  # Line
                        start = get_coordinates(item[1])
                        end = get_coordinates(item[2])
                        vector_data.append({
                            "page": page_number + 1,
                            "type": "line",
                            "start_point": start,
                            "end_point": end,
                            "control_points": []
                        })
                    elif op == "c":  # Cubic Bézier Curve
                        start = get_coordinates(item[1])
                        control1 = get_coordinates(item[2])
                        control2 = get_coordinates(item[3])
                        end = get_coordinates(item[4])
                        vector_data.append({
                            "page": page_number + 1,
                            "type": "cubic_curve",
                            "start_point": start,
                            "end_point": end,
                            "control_points": [control1, control2]
                        })
                    elif op == "qu":  # Quadrilateral
                        quad = operands[0]  # Quad(Point(...), Point(...), Point(...), Point(...))
                        p1 = get_coordinates(quad[0])
                        p2 = get_coordinates(quad[1])
                        p3 = get_coordinates(quad[2])
                        p4 = get_coordinates(quad[3])

                        # Calculate min and max to define the bounding rectangle
                        x_coords = [p1[0], p2[0], p3[0], p4[0]]
                        y_coords = [p1[1], p2[1], p3[1], p4[1]]
                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)
                        width = max_x - min_x
                        height = max_y - min_y

                        vector_data.append({
                            "page": page_number + 1,
                            "type": "quadrilateral",
                            "position": [min_x, min_y],
                            "width": width,
                            "height": height
                        })
                    else:
                        print("a rare op", op, "with operands", operands)
                except Exception as e:
                    print(f"[WARN] Error processing operator '{op}' on page {page_number +1}: {e}")

    return vector_data


def save_vector_data(vector_data, output_dir):
    """
    Saves lines, cubic curves, and quadrilaterals data into separate JSON files.

    Args:
        vector_data (list): List of dictionaries containing vector data.
        output_dir (str): Directory where JSON files will be saved.
    """
    lines = [item for item in vector_data if item['type'] == 'line']
    cubic_curves = [item for item in vector_data if item['type'] == 'cubic_curve']
    quadrilaterals = [item for item in vector_data if item['type'] == 'quadrilateral']

    lines_filepath = os.path.join(output_dir, 'lines.json')
    cubic_curves_filepath = os.path.join(output_dir, 'cubic_curves.json')
    quadrilaterals_filepath = os.path.join(output_dir, 'quadrilaterals.json')

    with open(lines_filepath, 'w') as f:
        json.dump(lines, f, indent=4)
    print(f"[INFO] Saved lines data to {lines_filepath}")

    with open(cubic_curves_filepath, 'w') as f:
        json.dump(cubic_curves, f, indent=4)
    print(f"[INFO] Saved cubic curves data to {cubic_curves_filepath}")

    with open(quadrilaterals_filepath, 'w') as f:
        json.dump(quadrilaterals, f, indent=4)
    print(f"[INFO] Saved quadrilaterals data to {quadrilaterals_filepath}")


def plot_vector_data(vector_data, output_dir):
    """
    Plots all vector data (lines, cubic curves, quadrilaterals) in a single plot 
    and saves it as a PDF.
    
    Args:
        vector_data (list): List of dictionaries containing vector data.
        output_dir (str): Directory where the PDF plot will be saved.
    """
    plt.figure(figsize=(10, 10))

    # To keep track of labels and avoid duplicates in the legend
    label_set = set()

    for item in vector_data:
        if item['type'] == 'line':
            x = [item['start_point'][0], item['end_point'][0]]
            y = [item['start_point'][1], item['end_point'][1]]
            label = f"Line (Page {item['page']})"
            if label not in label_set:
                plt.plot(x, y, 'b-', label=label)
                label_set.add(label)
            else:
                plt.plot(x, y, 'b-')
        elif item['type'] == 'cubic_curve':
            if item['control_points']:
                x_vals, y_vals = plot_bezier_curve(
                    item['start_point'],
                    item['control_points'],
                    item['end_point']
                )
                label = f"Cubic Curve (Page {item['page']})"
                if label not in label_set:
                    plt.plot(x_vals, y_vals, 'r-', label=label)
                    label_set.add(label)
                else:
                    plt.plot(x_vals, y_vals, 'r-')
        elif item['type'] == 'quadrilateral':
            x = item['position'][0]
            y = item['position'][1]
            width = item['width']
            height = item['height']
            quad = plt.Rectangle(
                (x, y), width, height, linewidth=1, edgecolor='green', 
                facecolor='none', label=f"Quadrilateral (Page {item['page']})"
            )
            if f"Quadrilateral (Page {item['page']})" not in label_set:
                plt.gca().add_patch(quad)
                label_set.add(f"Quadrilateral (Page {item['page']})")
            else:
                plt.gca().add_patch(quad)
        else:
            # Handle additional vector types if needed
            pass

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Vector Data from PDF")
    
    # To avoid duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small')
    
    plt.grid(True)

    # Invert Y-axis to correct the orientation
    plt.gca().invert_yaxis()

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Save the plot as PDF
    plot_pdf_path = os.path.join(output_dir, 'vector_data.pdf')
    try:
        plt.savefig(plot_pdf_path, format='pdf', bbox_inches='tight')
        print(f"[INFO] Saved collective plot as PDF: {plot_pdf_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save plot as PDF: {e}")

    # Display the plot interactively
    plt.show()

    # Close the plot to free memory
    plt.close()


# Main logic
if __name__ == "__main__":
    # Specify your input PDF
    pdf_path = 'NEW CATHAY.pdf'  # Replace with your actual PDF path

    # Directory to store JSON outputs and the final plot
    output_directory = os.path.join('..', 'data')
    ensure_directory_exists(output_directory)

    # Step 1: Remove letters and numbers from the original PDF
    redacted_pdf_path = os.path.join(output_directory, 'redacted.pdf')
    remove_letters_and_numbers(pdf_path, redacted_pdf_path)

    # Step 2: Extract vector data from the redacted PDF
    try:
        data = extract_vector_data_pymupdf(redacted_pdf_path)
    except Exception as e:
        print(f"[ERROR] An error occurred while extracting vector data: {e}")
        data = []

    # Step 3: If we have vector data, save it and plot it
    if data:
        try:
            save_vector_data(data, output_directory)
        except Exception as e:
            print(f"[ERROR] An error occurred while saving vector data: {e}")

        try:
            plot_vector_data(data, output_directory)
        except Exception as e:
            print(f"[ERROR] An error occurred while plotting vector data: {e}")

        # Optionally, you can print out details about curves/quadrilaterals:
        for idx, item in enumerate(data, start=1):
            if item["type"] == "cubic_curve":
                print(f"Cubic Curve {idx}:")
                print(f"  Page {item['page']}:")
                print(f"    Start Point: {item['start_point']}")
                print(f"    Control Points: {item['control_points']}")
                print(f"    End Point: {item['end_point']}\n")
            elif item["type"] == "quadrilateral":
                print(f"Quadrilateral {idx}:")
                print(f"  Page {item['page']}:")
                print(f"    Position: {item['position']}")
                print(f"    Width: {item['width']}")
                print(f"    Height: {item['height']}\n")
    else:
        print("[INFO] No vector data found.")
