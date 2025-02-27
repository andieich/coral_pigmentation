import os
import cv2
import numpy as np
import argparse
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Define constants
ROWS = 6
COLS = 4
RESOLUTION = 300
MARGIN = 0.1

def point_select_callback(event, points, texts, image_shape):
    """Callback function to handle mouse click events for selecting points."""
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        if len(points) < 4:
            points.append((x, y))
            text = plt.text(x, y, f'{len(points)}', color='red', fontsize=12)
            texts.append(text)
            plt.scatter(x, y, color='red')
            plt.draw()

def draw_polygon(event, points):
    """Draw polygon connecting the selected points."""
    if event.key == 'enter' and len(points) == 4:
        polygon = plt.Polygon(points, closed=True, fill=None, edgecolor='red')
        plt.gca().add_patch(polygon)
        plt.draw()

def undo_last_point(event, points, texts):
    """Undo the last selected point."""
    if event.key == 'backspace' and points:
        points.pop()
        text = texts.pop()
        text.remove()
        plt.gca().lines = plt.gca().lines[:-1]
        plt.draw()

def on_close(event, points):
    """Check the number of points on close event."""
    if len(points) != 4:
        raise ValueError("Four points must be selected.")

def annotate_color_checker(image_filename, csv_writer):
    """ Deskew and crop the color checker from the provided image """
    points = []
    texts = []

    # Read the image
    image = cv2.imread(image_filename)
    image_shape = image.shape

    # Display the image
    fig = plt.figure()
    plt.title('Select the four corners of the color checker, starting from the top left corner and moving clockwise!')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Set the window to full screen
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

    # Define the mouse click event to select four corners
    plt.connect('button_press_event', lambda event: point_select_callback(event, points, texts, image_shape))

    # Define the key press event for drawing the polygon and undoing the last point
    plt.connect('key_press_event', lambda event: draw_polygon(event, points))
    plt.connect('key_press_event', lambda event: undo_last_point(event, points, texts))

    # Define the close event to check the number of points
    fig.canvas.mpl_connect('close_event', lambda event: on_close(event, points))

    plt.show()

    # Write points to CSV
    for i, (x, y) in enumerate(points):
        csv_writer.writerow([image_filename, i, x, y])

    return points

def deskew_and_crop_color_checker(image_filename, points):
    """ Deskew and crop the color checker from the provided image """
    image = cv2.imread(image_filename)  # Read the image

    src_pts = np.array(points, dtype='float32')
    dst_pts = np.array([[0, 0], [0, RESOLUTION], [RESOLUTION, RESOLUTION], [RESOLUTION, 0]], dtype='float32')

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    color_checker = cv2.warpPerspective(image, M, (RESOLUTION, RESOLUTION))

    return color_checker

def analyze_color_checker(image, method='dominant', correction_method='Finlayson 2015', degree=2):
    """ Method to crop the patches for each color from the cropped color checker image
        It returns a dictionary of the average RGB values for each color patch
    """

    margin_px = int(MARGIN * RESOLUTION)  # Compute margin in pixels

    patches = {}  # Dictionary to store the average RGB values for each color patch

    plt.subplots(ROWS, COLS)

    # Crop the patches for each color from the cropped color checker image
    # + There are ROWS x COLS colors in the color checker
    # + The color patches are arranged in a grid pattern
    # + The color patches are of equal size

    # First, find the center of each color patch
    for i in range(ROWS):
        for j in range(COLS):
            patch_upper_left  = (i * RESOLUTION // ROWS + margin_px, j * RESOLUTION // COLS + margin_px)
            patch_lower_right = ((i + 1) * RESOLUTION // ROWS - margin_px, (j + 1) * RESOLUTION // COLS - margin_px)

            # Crop the patch from the image
            patch = image[patch_upper_left[1]:patch_lower_right[1], patch_upper_left[0]:patch_lower_right[0]]

            # Calculate the RGB values for the patch based on the selected method
            if method == 'mean':
                rgb_values = np.mean(patch, axis=(0, 1))
            elif method == 'min':
                rgb_values = np.min(patch, axis=(0, 1))
            elif method == 'max':
                rgb_values = np.max(patch, axis=(0, 1))
            elif method == 'dominant':
                pixels = patch.reshape(-1, 3)
                kmeans = KMeans(n_clusters=1).fit(pixels)
                rgb_values = kmeans.cluster_centers_[0]
            else:
                raise ValueError(f"Unknown method: {method}")

            # Swap the values compensating for the BGR to RGB conversion
            rgb_values = rgb_values[[2, 1, 0]]
            
            color = f'Patch {i * COLS + j + 1}'
            
            # Store the uncorrected patch values
            patches[color] = rgb_values

    return patches

def add_annotations(warped, patch_fraction, reference_values):
    """ Add annotations to the color checker image """
    # Ensure the image is a NumPy array
    if not isinstance(warped, np.ndarray):
        raise ValueError("The 'warped' image must be a NumPy array")

    # Ensure the image is in the correct data type
    if warped.dtype != np.uint8:
        warped = warped.astype(np.uint8)

    # Define the grid size (e.g., 6x4 for a standard color chart)
    grid_size = (6, 4)
    h, w = warped.shape[:2]
    patch_height = h // grid_size[1]
    patch_width = w // grid_size[0]
    
    for i in range(grid_size[1]):
        for j in range(grid_size[0]):
            x_start = j * patch_width
            y_start = i * patch_height
            x_end = x_start + patch_width
            y_end = y_start + patch_height
            
            # Calculate the center portion of the patch
            x_center_start = int(x_start + (1 - patch_fraction) / 2 * patch_width)
            y_center_start = int(y_start + (1 - patch_fraction) / 2 * patch_height)
            x_center_end = int(x_end - (1 - patch_fraction) / 2 * patch_width)
            y_center_end = int(y_end - (1 - patch_fraction) / 2 * patch_height)
            
            # Draw the center portion rectangle
            cv2.rectangle(warped, (x_center_start, y_center_start), (x_center_end, y_center_end), (0, 242, 0), 1)
                
            # Fill half of the patch with the reference color
            ref_color = reference_values[i * grid_size[0] + j]
            ref_color_bgr = (int(ref_color[2]), int(ref_color[1]), int(ref_color[0]))  # Convert RGB to BGR
            cv2.rectangle(warped, (x_center_start, y_center_start), (x_center_start + (x_center_end - x_center_start) // 2, y_center_end), ref_color_bgr, -1)
    
    return warped

def map_gamut(image_filename, output_filename, targets, patches, correction_method, degree, output_debug):
    """ Method to map the color gamut of the image to the target colors and save the corrected patches alongside the target colors """

    # Create the A and B matrices for the color correction
    A = np.zeros((len(targets), 3))
    B = np.zeros((len(targets), 3))
    for j, (color, target_rgb) in enumerate(targets.items()):
        if color in patches:
            A[j] = patches[color][:3]
            B[j] = target_rgb[:3]

    # Read the image
    image = cv2.imread(image_filename)

    # Transform image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform color correction
    transformed_image = np.clip(colour.colour_correction(image, A, B, method=correction_method, degree=degree), 0, 255)

    # White balance the image
    factors = np.zeros(3)

    wb_targets = [
        'Neutral 8',
        'White'
    ]

    for target in wb_targets:
        white = np.clip(colour.colour_correction(patches[target][:3], A, B), 0, 255)
        white_target = targets[target][:3]

        factors += white_target / white

    # White balance factors
    factors /= len(wb_targets)

    for i in range(3):
        transformed_image[:, :, i] = np.clip(transformed_image[:, :, i] * factors[i], 0, 255)
    # Transform image back to BGR
    transformed_image = cv2.cvtColor(transformed_image.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Save the transformed image
    cv2.imwrite(output_filename, transformed_image)

    # Extract the corrected colors from the patches
    corrected_patches = {}
    for color, rgb_values in patches.items():
        corrected_rgb = np.clip(colour.colour_correction(rgb_values, A, B), 0, 255)
        corrected_patches[color] = corrected_rgb

    # Create an image to display the corrected colors alongside the target colors
    patch_size = 50
    comparison_image = np.zeros((patch_size * ROWS, patch_size * COLS * 2, 3), dtype=np.uint8)

    for i in range(ROWS):
        for j in range(COLS):
            color = f'Patch {i * COLS + j + 1}'
            target_rgb = targets[color]
            corrected_rgb = corrected_patches[color]

            target_patch = np.full((patch_size, patch_size, 3), target_rgb, dtype=np.uint8)
            corrected_patch = np.full((patch_size, patch_size, 3), corrected_rgb, dtype=np.uint8)

            comparison_image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size, :] = target_patch
            comparison_image[i * patch_size:(i + 1) * patch_size, (j + COLS) * patch_size:(j + COLS + 1) * patch_size, :] = corrected_patch

            # Add color names
            cv2.putText(comparison_image, color, (j * patch_size + 5, i * patch_size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(comparison_image, color, ((j + COLS) * patch_size + 5, i * patch_size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Save the comparison image in the output_debug folder
    comparison_filename = os.path.join(output_debug, f'{os.path.splitext(os.path.basename(output_filename))[0]}_comparison.jpg')
    cv2.imwrite(comparison_filename, comparison_image)

def process_images(input_dir, output_dir_corrected, output_dir_debug, csv_writer, method, correction_method, degree):
    """ Process all images in the input directory and save the results to the specified output directories """
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_filename = os.path.join(input_dir, filename)
            output_filename = os.path.join(output_dir_corrected, f'{os.path.splitext(filename)[0]}_corrected.jpg')
            
            # Isolate the color checker from the image
            points = annotate_color_checker(image_filename, csv_writer)
            color_checker = deskew_and_crop_color_checker(image_filename, points)

            # Extract the color patches from the color checker and perform color correction
            patches = analyze_color_checker(color_checker, method, correction_method, degree)
            map_gamut(image_filename, output_filename, targets, patches, correction_method, degree, output_dir_debug)

def main(input_dir, output_dir_corrected, output_dir_debug, csv_filename, method, correction_method, degree):
    with open(csv_filename, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['image_filename', 'point_id', 'x', 'y'])
        process_images(input_dir, output_dir_corrected, output_dir_debug, csv_writer, method, correction_method, degree)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images for color correction.')
    parser.add_argument('--input', '-i', required=True, help='Path to the input directory')
    parser.add_argument('--output_corrected', '-oc', required=True, help='Path to the output directory for corrected images')
    parser.add_argument('--output_debug', '-od', required=True, help='Path to the output directory for debug images')
    parser.add_argument('--csv', '-c', required=True, help='Path to the CSV file to save points')
    parser.add_argument('--method', '-m', default='dominant', choices=['mean', 'median', 'min', 'max', 'dominant'], help='Method to calculate the RGB values for each patch (default: dominant)')
    parser.add_argument('--correction_method', '-cm', default='Cheung 2004', choices=['Finlayson 2015', 'Vandermonde', 'Cheung 2004'], help='Method to use for color correction (default: Cheung 2004)')
    parser.add_argument('--degree', '-d', type=int, default=2, choices=[1, 2, 3, 4], help='Degree for the Finlayson2015 and Vandermonde methods (default: 2)')

    args = parser.parse_args()

    main(args.input, args.output_corrected, args.output_debug, args.csv, args.method, args.correction_method, args.degree)