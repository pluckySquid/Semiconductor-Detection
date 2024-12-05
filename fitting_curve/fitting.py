import numpy as np
import cv2
import json
from scipy.optimize import least_squares

# --------------------------- Bézier Curve Functions ---------------------------

def bezier_curve(t, control_points):
    """
    Evaluate a cubic Bézier curve at parameter t.
    """
    P0, P1, P2, P3 = control_points
    t = np.atleast_1d(t)
    return ((1 - t)**3)[:, np.newaxis] * P0 + \
           3 * ((1 - t)**2 * t)[:, np.newaxis] * P1 + \
           3 * ((1 - t) * t**2)[:, np.newaxis] * P2 + \
           (t**3)[:, np.newaxis] * P3

def residuals(control_points, t, points):
    """
    Calculate residuals between the Bézier curve and target points for optimization.
    """
    cp = control_points.reshape((4, 2))
    curve_points = bezier_curve(t, cp)
    return (curve_points - points).ravel()

def calculate_control_points_for_logo(r, center, k):
    """
    Calculate initial control points for a quarter-circle Bézier curve approximation.
    """
    P0 = np.array([center[0] - r, center[1]])
    P1 = np.array([P0[0] + k * r, P0[1]])
    P2 = np.array([P0[0] + r, P0[1] - k * r])
    P3 = np.array([P0[0] + r, P0[1] - r])
    return np.array([P0, P1, P2, P3])

def fit_cubic_bezier(points, initial_cp):
    """
    Fit a cubic Bézier curve to a set of points using least-squares optimization.
    """
    distances = np.sqrt(((points[1:] - points[:-1])**2).sum(axis=1))
    cumulative = np.insert(np.cumsum(distances), 0, 0)
    t = cumulative / cumulative[-1] if cumulative[-1] != 0 else np.linspace(0, 1, len(points))
    result = least_squares(residuals, initial_cp, args=(t, points))
    return result.x.reshape((4, 2)) if result.success else initial_cp.reshape((4, 2))

def create_parametric_equation(control_points):
    """
    Generate parametric equations for a cubic Bézier curve.
    """
    P0, P1, P2, P3 = control_points
    equation = {
        "x(t)": f"(1-t)^3*{P0[0]:.4f} + 3*(1-t)^2*t*{P1[0]:.4f} + 3*(1-t)*t**2*{P2[0]:.4f} + t^3*{P3[0]:.4f}",
        "y(t)": f"(1-t)^3*{P0[1]:.4f} + 3*(1-t)^2*t*{P1[1]:.4f} + 3*(1-t)*t**2*{P2[1]:.4f} + t^3*{P3[1]:.4f}"
    }
    return equation

def draw_bezier(image, control_points, color=(0, 0, 255), thickness=2):
    """
    Draw a cubic Bézier curve on an image.
    """
    t_vals = np.linspace(0, 1, 100)
    bezier_pts = bezier_curve(t_vals, control_points).astype(int)
    for i in range(len(bezier_pts) - 1):
        pt1 = tuple(bezier_pts[i])
        pt2 = tuple(bezier_pts[i + 1])
        cv2.line(image, pt1, pt2, color, thickness)

# ------------------------------ Main Function ------------------------------

def main():
    # Load the McDonald's logo image
    image_path = 'mcdonald.png'  # Adjust path as needed
    logo = cv2.imread(image_path)

    # Define new background size
    background_height = logo.shape[0] * 2
    background_width = logo.shape[1] * 2

    # Create a larger white background
    background = np.ones((background_height, background_width, 3), dtype=np.uint8) * 255

    # Calculate the position to center the logo
    center_y = (background_height - logo.shape[0]) // 2
    center_x = (background_width - logo.shape[1]) // 2

    # Place the logo in the center of the background
    background[center_y:center_y + logo.shape[0], center_x:center_x + logo.shape[1]] = logo

    # Rotate the combined image by 180 degrees
    image_rotated_180 = cv2.rotate(background, cv2.ROTATE_180)

    # Define radius and center positions for the half-circles on the new background
    r = logo.shape[1] // 4.4  # Approximate radius for the half-circle fitting
    k = 0.5522847498
    center_right = (background_width // 2 - r, background_height // 2)
    center_left = (center_right[0] + 2 * r, center_right[1])

    # Calculate control points for each half-circle
    control_points_right = calculate_control_points_for_logo(r, center_right, k)
    control_points_left = calculate_control_points_for_logo(r, center_left, k)

    # Define points along the half-circle for fitting
    theta = np.linspace(np.pi, 0, 100)
    half_circle_points_right = np.stack((center_right[0] + r * np.cos(theta), center_right[1] + r * np.sin(theta)), axis=1)
    half_circle_points_left = np.stack((center_left[0] + r * np.cos(theta), center_left[1] + r * np.sin(theta)), axis=1)

    # Fit Bézier curves to the half-circles
    optimized_cp_right = fit_cubic_bezier(half_circle_points_right, control_points_right.flatten())
    optimized_cp_left = fit_cubic_bezier(half_circle_points_left, control_points_left.flatten())

    # Apply upward translation to the control points to make the curves overlap with the logo
    y_translation = int(0.43 * logo.shape[0])  # Adjust this value as needed
    optimized_cp_right[:, 1] -= y_translation
    optimized_cp_left[:, 1] -= y_translation

    # Create parametric equations for JSON output
    equation_right = create_parametric_equation(optimized_cp_right)
    equation_left = create_parametric_equation(optimized_cp_left)

    # Prepare data for JSON output
    bezier_data = {
        "right_quarter_circle": {
            "control_points": optimized_cp_right.tolist(),
            "parametric_equation": equation_right
        },
        "left_quarter_circle": {
            "control_points": optimized_cp_left.tolist(),
            "parametric_equation": equation_left
        }
    }

    # Save the parametric equations and control points to a JSON file
    output_json_path = 'circle_bezier_equations.json'
    with open(output_json_path, 'w') as f:
        json.dump(bezier_data, f, indent=4)

    # Draw the fitted Bézier curves on the rotated image
    overlap_color = (0, 0, 255)  # Use a single color for overlap effect
    draw_bezier(image_rotated_180, optimized_cp_right, color=overlap_color, thickness=2)
    draw_bezier(image_rotated_180, optimized_cp_left, color=overlap_color, thickness=2)

    # Save the final image with overlapping curves on a larger background
    output_image_path = 'mcdonald_with_bezier_adjusted.png'  # Adjust the path as needed
    cv2.imwrite(output_image_path, image_rotated_180)

    print(f"Parametric equations saved in {output_json_path}")
    print(f"Image with Bézier curves saved in {output_image_path}")

# Run the main function
if __name__ == "__main__":
    main()
