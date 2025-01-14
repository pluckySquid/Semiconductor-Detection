import re
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np
import os
import json


def ensure_directory_exists(directory_path):
    """
    Ensures that the specified directory exists. If not, creates it.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"[INFO] Created directory: {directory_path}")


def save_bottom_left_to_pdf(pdf_path, output_path):
    """
    Save the bottom-left 12% width and 6% height of each page in a new PDF.
    """
    doc = fitz.open(pdf_path)
    output_pdf = fitz.open()  # Create a new PDF

    for page_number in range(len(doc)):
        page = doc[page_number]
        rect = page.rect
        new_width = rect.width * 0.12
        new_height = rect.height * 0.06
        bottom_left_rect = fitz.Rect(0, rect.height - new_height, new_width, rect.height)

        # Add cropped area to the new PDF
        new_page = output_pdf.new_page(width=new_width, height=new_height)
        new_page.show_pdf_page(fitz.Rect(0, 0, new_width, new_height), doc, page_number, clip=bottom_left_rect)

    output_pdf.save(output_path)
    output_pdf.close()
    doc.close()
    print(f"[INFO] Bottom-left portion saved to {output_path}")


def remove_letters_and_numbers(pdf_path, output_path):
    """
    Detect and redact all words containing letters or digits from the PDF.
    """
    pattern = re.compile(r'[A-Za-z0-9]')
    doc = fitz.open(pdf_path)

    for page in doc:
        words = page.get_text("words")
        for w in words:
            x0, y0, x1, y1, text = w[:5]
            if pattern.search(text):
                rect = fitz.Rect(x0, y0, x1, y1)
                page.add_redact_annot(rect)

        page.apply_redactions()

    doc.save(output_path)
    doc.close()
    print(f"[INFO] Letters and numbers removed from PDF. Saved: {output_path}")


def extract_vector_data_pymupdf(pdf_path):
    """
    Extracts vector data (lines, Bézier curves, quadrilaterals) from a PDF using PyMuPDF.
    """
    doc = fitz.open(pdf_path)
    vector_data = []

    def get_coordinates(point):
        if hasattr(point, 'x') and hasattr(point, 'y'):
            return [point.x, point.y]
        elif isinstance(point, (list, tuple)) and len(point) == 2:
            return [point[0], point[1]]
        else:
            raise ValueError("Invalid point format")

    for page_number in range(len(doc)):
        page = doc[page_number]
        drawings = page.get_drawings()

        for drawing in drawings:
            for item in drawing["items"]:
                op = item[0]
                try:
                    if op == "l":  # Line
                        vector_data.append({
                            "page": page_number + 1,
                            "type": "line",
                            "start_point": get_coordinates(item[1]),
                            "end_point": get_coordinates(item[2]),
                        })
                    elif op == "c":  # Cubic Bézier Curve
                        vector_data.append({
                            "page": page_number + 1,
                            "type": "cubic_curve",
                            "start_point": get_coordinates(item[1]),
                            "control_points": [get_coordinates(item[2]), get_coordinates(item[3])],
                            "end_point": get_coordinates(item[4]),
                        })
                    elif op == "qu":  # Quadrilateral
                        quad = item[1]
                        points = [get_coordinates(p) for p in quad]
                        vector_data.append({
                            "page": page_number + 1,
                            "type": "quadrilateral",
                            "points": points,
                        })
                except Exception as e:
                    print(f"[WARN] Error processing operator '{op}' on page {page_number + 1}: {e}")

    doc.close()
    return vector_data


def save_vector_data(vector_data, output_dir):
    """
    Save vector data (lines, Bézier curves, quadrilaterals) to JSON files.
    """
    with open(os.path.join(output_dir, "vector_data.json"), "w") as f:
        json.dump(vector_data, f, indent=4)
    print(f"[INFO] Vector data saved to {os.path.join(output_dir, 'vector_data.json')}")


def plot_vector_data(vector_data, output_dir):
    """
    Plot vector data and save it as a PDF. After saving, crop the bottom-left 12% width and 6% height.
    """
    plt.figure(figsize=(10, 10))

    for item in vector_data:
        if item["type"] == "line":
            x = [item["start_point"][0], item["end_point"][0]]
            y = [item["start_point"][1], item["end_point"][1]]
            plt.plot(x, y, "b-", label="Line")
        elif item["type"] == "cubic_curve":
            x_vals, y_vals = plot_bezier_curve(
                item["start_point"], item["control_points"], item["end_point"]
            )
            plt.plot(x_vals, y_vals, "r-", label="Bézier Curve")
        elif item["type"] == "quadrilateral":
            points = item["points"]
            x = [p[0] for p in points] + [points[0][0]]
            y = [p[1] for p in points] + [points[0][1]]
            plt.plot(x, y, "g-", label="Quadrilateral")

    plt.gca().invert_yaxis()
    plt.legend()
    plot_pdf_path = os.path.join(output_dir, "vector_plot.pdf")
    plt.savefig(plot_pdf_path, format="pdf", bbox_inches="tight")
    print(f"[INFO] Vector plot saved to {plot_pdf_path}")
    plt.close()

    # Crop the bottom-left portion of the plot PDF
    cropped_pdf_path = os.path.join(output_dir, "vector_plot_cropped.pdf")
    save_bottom_left_to_pdf(plot_pdf_path, cropped_pdf_path)


def plot_bezier_curve(start, control_points, end, n_points=100):
    """
    Compute Bézier curve points.
    """
    t = np.linspace(0, 1, n_points)
    start = np.array(start)
    control1, control2 = map(np.array, control_points)
    end = np.array(end)
    bezier = ((1 - t) ** 3)[:, None] * start + \
             3 * ((1 - t) ** 2)[:, None] * t[:, None] * control1 + \
             3 * (1 - t)[:, None] * (t ** 2)[:, None] * control2 + \
             (t ** 3)[:, None] * end
    return bezier[:, 0], bezier[:, 1]


# Main logic
if __name__ == "__main__":
    pdf_path = "ori.pdf"
    #pdf_path = "output.pdf"
    pdf_path = 'NEW CATHAY.pdf'
    output_directory = os.path.join("output")
    ensure_directory_exists(output_directory)

    # Step 1: Save bottom-left portion to a new PDF
    bottom_left_pdf_path = os.path.join(output_directory, "bottom_left.pdf")
    save_bottom_left_to_pdf(pdf_path, bottom_left_pdf_path)

    # Step 2: Remove letters and numbers
    redacted_pdf_path = os.path.join(output_directory, "redacted.pdf")
    remove_letters_and_numbers(bottom_left_pdf_path, redacted_pdf_path)

    # Step 3: Extract vector data
    vector_data = extract_vector_data_pymupdf(redacted_pdf_path)
    save_vector_data(vector_data, output_directory)

    # Step 4: Plot vector data and save cropped plot
    plot_vector_data(vector_data, output_directory)
