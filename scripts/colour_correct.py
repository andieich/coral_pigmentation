import os
import cv2
import colour
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PyQt5.QtWidgets import QApplication, QMessageBox
import argparse
import csv

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
targets = {
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

def point_select_callback(event, points, texts, image_shape):
    """ Callback for mouse click event to select four corners of the color checker """
    if plt.get_current_fig_manager().toolbar.mode == '':
        if len(points) < 4:
            ix, iy = event.xdata, event.ydata
            if ix is not None and iy is not None:
                # Clip the coordinates to the image boundaries
                ix = np.clip(ix, 0, image_shape[1] - 1)
                iy = np.clip(iy, 0, image_shape[0] - 1)
                points.append((ix, iy))
                plt.plot(ix, iy, 'ro')
                text = plt.text(ix, iy, str(len(points)), color='white', fontsize=12, ha='center', va='center')
                texts.append(text)
                plt.draw()

def undo_last_point(event, points, texts):
    """ Callback for key press event to undo the last point """
    if event.key == 'u' and points:
        points.pop()
        if plt.gca().lines:
            plt.gca().lines[-1].remove()  # Remove the last point from the plot
        if texts:
            texts[-1].remove()  # Remove the last text annotation
            texts.pop()
        plt.draw()

def draw_polygon(event, points):
    """ Callback for key press event to draw the polygon around the color checker and close the interactive plot """
    if event.key == 'enter':
        if len(points) < 4:
            show_error_message("Less than 4 points were chosen to extract the color checker.")
            return
        points = np.array(points)
        polygon = Polygon(points, closed=True, fill=None, edgecolor='r', linewidth=2)
        plt.gca().add_patch(polygon)
        plt.draw()

        # Wait for the plot to update
        plt.pause(1.0)
        plt.close()

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

def analyze_color_checker(image, palette_filename=None, debug_filename=None, method='mean'):
    """ Method to crop the patches for each color from the cropped color checker image
        It returns a dictionary of the average RGB values for each color patch
        Creates debug images for the patch selection and the resulting color palette
    """
    debug_image = image.copy()    # Create a copy of the image for debugging

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
            else:  # default to median
                rgb_values = np.median(patch, axis=(0, 1))

            # Swap the values compensating for the BGR to RGB conversion
            rgb_values = rgb_values[[2, 1, 0]]
            color   = colors[i * COLS + j]

            # Perform color correction on the patch
            corrected_patch = colour.colour_correction(patch, np.array([rgb_values]), np.array([targets[color]]))
            corrected_patch = np.clip(corrected_patch, 0, 255).astype(np.uint8)

            # Create a target patch
            target_patch = np.full_like(patch, targets[color])

            # Convert patches to RGB
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

            # Combine the patches in the correct order
            combined_patch = np.hstack((patch, corrected_patch, target_patch))
            # Show the patch
            plt.subplot(ROWS, COLS, i * COLS + j + 1)
            plt.imshow(combined_patch)
            plt.title(f'{color}')
            plt.axis('off')

            patches[color] = rgb_values

            # Draw the position of the patch on the image
            cv2.rectangle(debug_image, patch_upper_left, patch_lower_right, (0, 255, 0), 1, cv2.LINE_AA)

            # Put a text label with the color name on the patch
            cv2.putText(debug_image, colors[i * COLS + j], [p - margin_px//2 for p in patch_upper_left], cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust for title
    plt.suptitle('Color Checker Patches (left=actual, middle=corrected, right=target)', fontsize=16)
    if palette_filename:
        plt.savefig(palette_filename)
    plt.close()

    # Save the debug image
    if debug_filename:
        cv2.imwrite(debug_filename, debug_image)

    return patches

def map_gamut(image_filename, output_filename, targets, patches, white_balance_target):
    """ Method to map the color gamut of the image to the target colors """

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

    transformed_image = np.clip(colour.colour_correction(image, A, B), 0, 255)

    if white_balance_target:
        # White balance the image
        factors = np.zeros(3)

        white = np.clip(colour.colour_correction(patches[white_balance_target][:3], A, B), 0, 255)
        white_target = targets[white_balance_target][:3]

        factors += white_target / white

        for i in range(3):
            transformed_image[:, :, i] = np.clip(transformed_image[:, :, i] * factors[i], 0, 255)

    # Transform image back to BGR
    transformed_image = cv2.cvtColor(transformed_image.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Save the transformed image
    cv2.imwrite(output_filename, transformed_image)

def process_images(input_dir, output_dir_corrected, output_dir_debug, csv_writer, white_balance_target, method):
    """ Process all images in the input directory and save the results to the specified output directories """
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_filename = os.path.join(input_dir, filename)
            output_filename = os.path.join(output_dir_corrected, f'{os.path.splitext(filename)[0]}_corrected.jpg')
            palette_filename = os.path.join(output_dir_debug, f'{os.path.splitext(filename)[0]}_palette.jpg')
            debug_filename = os.path.join(output_dir_debug, f'{os.path.splitext(filename)[0]}_debug.jpg')

            # Isolate the color checker from the image
            points = annotate_color_checker(image_filename, csv_writer)
            color_checker = deskew_and_crop_color_checker(image_filename, points)

            # Extract the color patches from the color checker and perform color correction
            patches = analyze_color_checker(color_checker, palette_filename, debug_filename, method)
            map_gamut(image_filename, output_filename, targets, patches, white_balance_target)

def main(input_dir, output_dir_corrected, output_dir_debug, csv_filename, white_balance_target, method):
    with open(csv_filename, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['image_filename', 'point_id', 'x', 'y'])
        process_images(input_dir, output_dir_corrected, output_dir_debug, csv_writer, white_balance_target, method)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images for color correction.')
    parser.add_argument('--input', '-i', required=True, help='Path to the input directory')
    parser.add_argument('--output_corrected', '-oc', required=True, help='Path to the output directory for corrected images')
    parser.add_argument('--output_debug', '-od', required=True, help='Path to the output directory for debug images')
    parser.add_argument('--csv', '-c', required=True, help='Path to the CSV file to save points')
    parser.add_argument('--white_balance', '-wb', default='White', help='Patch to use for white balancing (default: White). Use "None" to disable white balancing.')
    parser.add_argument('--method', '-m', default='mean', choices=['mean', 'median', 'min', 'max'], help='Method to calculate the RGB values for each patch (default: mean)')

    args = parser.parse_args()

    white_balance_target = None if args.white_balance.lower() == 'none' else args.white_balance

    main(args.input, args.output_corrected, args.output_debug, args.csv, white_balance_target, args.method)