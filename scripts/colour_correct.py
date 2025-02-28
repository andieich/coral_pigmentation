import os
os.environ['QT_MAC_WANTS_LAYER'] = '1'
import cv2
import colour
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PyQt5.QtWidgets import QApplication, QMessageBox
import argparse
import csv
from sklearn.cluster import KMeans
from datetime import datetime

# Define some constants regarding the color checker
ROWS   = 6
COLS   = 4
MARGIN = 0.06  # 6% margin/border around the color patches

# Resample the color checker to a fixed resolution
RESOLUTION = 600

# Colors in the order of the color checker patches if annotated starting from the top left corner and moving clockwise
colors = ['Bluish Green', 'Orange Yellow', 'Cyan', 'Black',         # Row 1
          'Blue Flower', 'Yellow Green', 'Magenta', 'Neutral 3.5',  # Row 2
          'Foliage', 'Purple', 'Yellow', 'Neutral 5',               # Row 3
          'Blue Sky', 'Moderate Red', 'Red', 'Neutral 6.5',         # Row 4
          'Light Skin', 'Purplish Blue', 'Green', 'Neutral 8',      # Row 5
          'Dark Skin', 'Orange', 'Blue', 'White']                   # Row 6

# Define targets in RGB format (from the X-Rite Classic color checker documentation)
targets =   {
            'Dark Skin':     [115,  82,  68],
            'Light Skin':    [194, 150, 130],
            'Blue Sky':      [ 98, 122, 157],
            'Foliage':       [ 87, 108,  67],
            'Blue Flower':   [133, 128, 177],
            'Bluish Green':  [103, 189, 170],
            'Orange':        [214, 126,  44],
            'Purplish Blue': [ 80,  91, 166],
            'Moderate Red':  [193,  90,  99],
            'Purple':        [ 94,  60, 108],
            'Yellow Green':  [157, 188,  64],
            'Orange Yellow': [224, 163,  46],
            'Blue':          [ 56,  61, 150],
            'Green':         [ 70, 148,  73],
            'Red':           [175,  54,  60],
            'Yellow':        [231, 199,  31],
            'Magenta':       [187,  86, 149],
            'Cyan':          [  8, 133, 161],
            'White':         [243, 243, 242],
            'Neutral 8':     [200, 200, 200],
            'Neutral 6.5':   [160, 160, 160],
            'Neutral 5':     [122, 122, 121],
            'Neutral 3.5':   [ 85,  85,  85],
            'Black':         [ 52,  52,  52],
            }

def point_select_callback(event, points, texts, ax):
    """ Callback for mouse click event to select four corners of the color checker """
    if plt.get_current_fig_manager().toolbar.mode == '':
        if len(points) < 4:
            ix, iy = event.xdata, event.ydata
            points.append((ix, iy))
            point = plt.plot(ix, iy, 'ro')
            text = plt.text(ix, iy, str(len(points)), color='white', fontsize=12, ha='center', va='center')
            texts.append((point, text))
            plt.draw()
            if len(points) == 4:
                draw_polygon(points, ax)

def undo_last_point(event, points, texts, ax):
    """ Callback for key press event to undo the last point """
    if event.key == 'u' and len(points) < 4:
        if points:
            points.pop()
            if texts:
                point, text = texts.pop()
                point[0].remove()
                text.remove()
            plt.draw()

def reset_points(event, points, texts, ax):
    """ Callback for key press event to reset all points and the polygon """
    if event.key == 'r' and len(points) == 4:
        if points:
            points.clear()
            while texts:
                point, text = texts.pop()
                point[0].remove()
                text.remove()
            for patch in ax.patches:
                patch.remove()  # Clear the polygon
            plt.draw()

def draw_polygon(points, ax):
    """ Draw the polygon around the color checker """
    polygon = Polygon(points, closed=True, fill=None, edgecolor='r', linewidth=2)
    ax.add_patch(polygon)
    plt.draw()

def on_close(event, points):
    """ Callback for close event to check the number of points """
    if len(points) < 4:
        show_error_message("Less than 4 points were chosen to extract the color checker.")

def show_error_message(message):
    """ Show an error message using PyQt5 """
    app = QApplication([])
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(message)
    msg.setWindowTitle("Error")
    msg.exec_()

def check_enter_key(event, points):
    """ Callback for key press event to check if Enter is pressed and 4 points are selected """
    if event.key == 'enter' and len(points) == 4:
        plt.close()

def annotate_color_checker(image_filename, csv_writer):
    """ Deskew and crop the color checker from the provided image """
    points = []
    texts = []

    # Read the image
    image = cv2.imread(image_filename)

    # Display the image
    fig, ax = plt.subplots()
    plt.title('Select the four corners of the color checker, starting from the top left corner and moving clockwise!')
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Set the window to full screen
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

    # Select the magnifier tool
    toolbar = manager.toolbar
    toolbar.zoom()

    # Define the mouse click event to select four corners
    fig.canvas.mpl_connect('button_press_event', lambda event: point_select_callback(event, points, texts, ax))

    # Define the key press event for undoing the last point
    fig.canvas.mpl_connect('key_press_event', lambda event: undo_last_point(event, points, texts, ax))

    # Define the key press event for resetting all points and the polygon
    fig.canvas.mpl_connect('key_press_event', lambda event: reset_points(event, points, texts, ax))

    # Define the key press event for checking if Enter is pressed
    fig.canvas.mpl_connect('key_press_event', lambda event: check_enter_key(event, points))

    # Define the close event to check the number of points
    fig.canvas.mpl_connect('close_event', lambda event: on_close(event, points))

    # Define the event to deactivate the zoom tool after use
    def deactivate_zoom(event):
        if toolbar.mode == 'zoom rect':
            toolbar.zoom()
    
    fig.canvas.mpl_connect('button_release_event', deactivate_zoom)

    plt.show()

    # Write points to CSV
    for i, (x, y) in enumerate(points):
        csv_writer.writerow([image_filename, i + 1, x, y])

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
    patch_positions = {}  # Dictionary to store the positions of each color patch

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
                kmeans = KMeans(n_clusters=1)
                kmeans.fit(pixels)
                rgb_values = kmeans.cluster_centers_[0]
            else:  # default to median
                rgb_values = np.median(patch, axis=(0, 1))
            
            # Swap the values compensating for the BGR to RGB conversion
            rgb_values = rgb_values[[2, 1, 0]]
            
            color   = colors[i * COLS + j]
            
            # Store the uncorrected patch values
            patches[color] = rgb_values
            patch_positions[color] = (patch_upper_left, patch_lower_right)


    return patches, patch_positions

def map_gamut(image_filename, output_filename, targets, patches, patch_positions, correction_method, degree, points, output_debug):
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
        white = np.clip(colour.colour_correction(patches[target][:3], A, B, method=correction_method, degree=degree), 0, 255)
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

    # Crop and de-warp the color checker from the corrected image
    corrected_color_checker = deskew_and_crop_color_checker(output_filename, points)

    # Rotate the dewarped color checker by 90 degrees
    corrected_color_checker = cv2.rotate(corrected_color_checker, cv2.ROTATE_90_CLOCKWISE)

    # Annotate the corrected color checker with the target colors
    patch_height = RESOLUTION // ROWS
    patch_width = RESOLUTION // COLS
    margin_px = int(MARGIN * patch_height)

    for i in range(ROWS):
        for j in range(COLS):
            color = colors[i * COLS + (COLS - 1 - j)]  # Reverse the column index
            target_rgb = targets[color]
            target_bgr = (int(target_rgb[2]), int(target_rgb[1]), int(target_rgb[0]))  # Convert RGB to BGR

            # Calculate the center portion of the patch
            x_start = j * patch_width + margin_px
            y_start = i * patch_height + margin_px
            x_end = (j + 1) * patch_width - margin_px
            y_end = (i + 1) * patch_height - margin_px

            # Calculate the smaller rectangle for the target color
            target_x_start = x_start + (x_end - x_start) // 4
            target_y_start = y_start + (y_end - y_start) // 4
            target_x_end = target_x_start + (x_end - x_start) // 2
            target_y_end = target_y_start + (y_end - y_start) // 2

            # Fill a small fraction of the patch with the target color
            cv2.rectangle(corrected_color_checker, (target_x_start, target_y_start), (target_x_end, target_y_end), target_bgr, -1)

    # Save the cropped and de-warped color checker
    color_checker_filename = os.path.join(output_debug, f'{os.path.splitext(os.path.basename(output_filename))[0]}_color_checker.jpg')
    cv2.imwrite(color_checker_filename, corrected_color_checker)

def process_images(input_dir, output_dir_corrected, output_dir_debug, csv_writer, method, correction_method, degree):
    """ Process all images in the input directory and save the results to the specified output directories """
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_filename = os.path.join(input_dir, filename)
            output_filename = os.path.join(output_dir_corrected, f'{os.path.splitext(filename)[0]}_corrected.jpg')
            
            # Check if the output file already exists
            if os.path.exists(output_filename):
                print(f"Skipping {filename} as it has already been processed.")
                continue
            
            # Isolate the color checker from the image
            points = annotate_color_checker(image_filename, csv_writer)
            color_checker = deskew_and_crop_color_checker(image_filename, points)

            # Extract the color patches from the color checker and perform color correction
            patches, patch_positions = analyze_color_checker(color_checker, method, correction_method, degree)
            map_gamut(image_filename, output_filename, targets, patches, patch_positions, correction_method, degree, points, output_dir_debug)

def main(input_dir, output_dir_corrected, output_dir_debug, csv_filename, method, correction_method, degree):
    # Create output directories if they don't exist
    os.makedirs(output_dir_corrected, exist_ok=True)
    os.makedirs(output_dir_debug, exist_ok=True)

    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        # Write the header only if the file is being created
        if not file_exists:
            csv_writer.writerow(['image_filename', 'point_id', 'x', 'y'])
        process_images(input_dir, output_dir_corrected, output_dir_debug, csv_writer, method, correction_method, degree)
        
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images for color correction.')
    parser.add_argument('--input', '-i', required=True, help='Path to the input directory')
    parser.add_argument('--output_corrected', '-oc', default='./corrected_images', help='Path to the output directory for corrected images (default: ./corrected_images)')
    parser.add_argument('--output_debug', '-od', default='./debugging_images', help='Path to the output directory for debug images (default: ./debugging_images)')
    parser.add_argument('--csv', '-c', default=datetime.now().strftime('%Y_%m_%d_colourchart_positions.csv'), help='Path to the CSV file to save points (default: YYYY_MM_DD_colourchart_positions.csv)')
    parser.add_argument('--method', '-m', default='mean', choices=['mean', 'median', 'min', 'max', 'dominant'], help='Method to calculate the RGB values for each patch (default: dominant)')
    parser.add_argument('--correction_method', '-cm', default='Cheung 2004', choices=['Finlayson 2015', 'Vandermonde', 'Cheung 2004'], help='Method to use for color correction (default: Cheung 2004)')
    parser.add_argument('--degree', '-d', type=int, default=1, choices=[1, 2, 3, 4], help='Degree for the Finlayson 2015 and Vandermonde methods (default: 1)')

    args = parser.parse_args()

    main(args.input, args.output_corrected, args.output_debug, args.csv, args.method, args.correction_method, args.degree)