import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def extract_vector_data_pymupdf(pdf_path):
    """
    Extracts vector data (lines and cubic Bézier curves) from a PDF using PyMuPDF.

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
                elif op == "c":  # Curve (Cubic Bézier)
                    start = get_coordinates(item[1])
                    control1 = get_coordinates(item[2])
                    control2 = get_coordinates(item[3])
                    end = get_coordinates(item[4])
                    vector_data.append({
                        "page": page_number + 1,
                        "type": "curve",
                        "start_point": start,
                        "end_point": end,
                        "control_points": [control1, control2]
                    })
                # You can handle other operations (e.g., quadratic curves) here if needed

    return vector_data

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
    
    # Cubic Bézier formula with correct parentheses
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
        print(f"Created directory: {directory_path}")

def save_vector_data(vector_data, output_dir):
    """
    Saves lines and curves data into separate JSON files.

    Args:
        vector_data (list): List of dictionaries containing vector data.
        output_dir (str): Directory where JSON files will be saved.
    """
    lines = [item for item in vector_data if item['type'] == 'line']
    curves = [item for item in vector_data if item['type'] == 'curve']

    lines_filepath = os.path.join(output_dir, 'lines.json')
    curves_filepath = os.path.join(output_dir, 'curves.json')

    with open(lines_filepath, 'w') as f:
        json.dump(lines, f, indent=4)
    print(f"Saved lines data to {lines_filepath}")

    with open(curves_filepath, 'w') as f:
        json.dump(curves, f, indent=4)
    print(f"Saved curves data to {curves_filepath}")

def plot_vector_data(vector_data, output_dir):
    """
    Plots all vector data (lines and curves) in a single plot and saves it as a PDF.
    
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
        elif item['type'] == 'curve':
            if item['control_points']:
                x, y = plot_bezier_curve(
                    item['start_point'],
                    item['control_points'],
                    item['end_point']
                )
                label = f"Curve (Page {item['page']})"
                if label not in label_set:
                    plt.plot(x, y, 'r-', label=label)
                    label_set.add(label)
                else:
                    plt.plot(x, y, 'r-')
            else:
                # If no control points, treat as a straight line
                x = [item['start_point'][0], item['end_point'][0]]
                y = [item['start_point'][1], item['end_point'][1]]
                label = f"Approx Line (Page {item['page']})"
                if label not in label_set:
                    plt.plot(x, y, 'g--', label=label)
                    label_set.add(label)
                else:
                    plt.plot(x, y, 'g--')

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

    # Save the plot as PDF before showing it
    plot_pdf_path = os.path.join(output_dir, 'vector_data.pdf')
    try:
        plt.savefig(plot_pdf_path, format='pdf', bbox_inches='tight')
        print(f"Saved collective plot as PDF: {plot_pdf_path}")
    except Exception as e:
        print(f"Failed to save plot as PDF: {e}")

    # Display the plot interactively
    plt.show()

    # Close the plot to free memory
    plt.close()


# Example usage
if __name__ == "__main__":
    pdf_path = 'ori.pdf'  # Replace with your PDF file path
    output_directory = os.path.join('..', 'data')  # Output directory for JSON files and PDF plot

    # Ensure the output directory exists
    ensure_directory_exists(output_directory)

    # Extract vector data from PDF
    try:
        data = extract_vector_data_pymupdf(pdf_path)
    except Exception as e:
        print(f"An error occurred while extracting vector data: {e}")
        data = []

    if data:
        # Save lines and curves data into separate JSON files
        try:
            save_vector_data(data, output_directory)
        except Exception as e:
            print(f"An error occurred while saving vector data: {e}")

        # Plot all vector data and save as PDF
        try:
            plot_vector_data(data, output_directory)
        except Exception as e:
            print(f"An error occurred while plotting vector data: {e}")

        # Optionally, print curve data with control points
        for idx, item in enumerate(data, start=1):
            if item["type"] == "curve":
                print(f"Curve {idx}:")
                print(f"  Page {item['page']}:")
                print(f"    Start Point: {item['start_point']}")
                print(f"    Control Points: {item['control_points']}")
                print(f"    End Point: {item['end_point']}\n")
    else:
        print("No vector data found.")
