############################################################
# 12/23/2024
# Kendell Cottam (kendellwc@protonmail.com)
# Course: Fundamentals of CV and IP
# Week 6
# Assignment 2 - Blemish Removal
# When you run the app it will display an image of a face
# that has some skin blemishes on it. Click on a blemish,
# and it will be replaced with a smooth patch of skin from
# the surrounding area.
############################################################

import cv2
import numpy as np

window_name = "Blemish Remover (click to remove, opposite click to undo last)"
cv2.namedWindow(window_name)

sample_radius = 15
number_of_samples = 16
patch_size = 32
half_patch_size = np.int32(patch_size / 2)
last_patch = []
last_patch_location = (-1, -1)

# Set to True for testing, which will display the patches from the surrounding area
# along with their Fourier Transforms used for checking 'smoothness'.
# Two lists are displayed: un-ordered and ordered (by smoothness)
should_display_sample_images = False


def get_blank_mask():
    mask = np.zeros(shape=[patch_size, patch_size], dtype='uint8')
    return mask


def setup_patch_mask():
    mask = get_blank_mask()
    # draw a filled circle on a blank image
    radius = sample_radius
    center = (half_patch_size - 1, half_patch_size - 1)
    cv2.circle(mask, center=center, radius=radius, color=[255], thickness=-1, lineType=cv2.LINE_8)
    return mask


def get_patch(x, y, get_color):
    # return a patch sample from the image, either gray scale or color depending on get_color
    row_start = y - half_patch_size
    row_end = y + half_patch_size
    column_start = x - half_patch_size
    column_end = x + half_patch_size
    if get_color:
        patch = blemish[row_start:row_end, column_start:column_end, :]
        return patch
    else:
        blemish_gray = cv2.cvtColor(blemish, cv2.COLOR_BGR2GRAY)
        patch =  blemish_gray[row_start:row_end, column_start:column_end]
        return patch


def apply_patch(patch, location):
    global blemish
    blemish = cv2.seamlessClone(patch, blemish, patch_mask, location, cv2.NORMAL_CLONE)
    cv2.imshow(window_name, blemish)


def is_point_within_safe_boundary_of_image(x, y):
    # check to see if point is far enough away from the edge of the image as to not generate errors,
    # complicated logic would be needed to prevent errors when sampling or applying patches very near the edge
    if (x - half_patch_size >= 0) and (x + half_patch_size < max_x) and (y - half_patch_size >= 0) and (y + half_patch_size < max_y):
        return True
    else:
        return False


def var_fft(image_gray):
    # Compute the discrete Fourier Transform of the image
    dft = cv2.dft(np.float32(image_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    # Shift the zero-frequency component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)
    # calculate the magnitude of the Fourier Transform
    magnitude_spectrum = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    # Scale (normalize) the magnitude
    magnitude_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    # Return the fft image and the variance
    return magnitude_normalized, magnitude_normalized.var()


def get_array_of_patches_from_surrounding_area(x, y):
    # loop through several locations surrounding the circular area around a 'blemish' in the image
    samples = []

    two_pi = np.pi * 2
    angle_increment = two_pi / number_of_samples
    current_angle = 0

    while current_angle < two_pi:
        # compute location to sample
        sample_x = np.int32((np.cos(current_angle) * (sample_radius * 2)) + x)
        sample_y = np.int32((np.sin(current_angle) * (sample_radius * 2)) + y)

        # ensure this patch is within the safe boundaries of the image space
        if is_point_within_safe_boundary_of_image(sample_x, sample_y):

            # get the patches at this location
            patch_color = get_patch(sample_x, sample_y, True)
            patch_gray = get_patch(sample_x, sample_y, False)

            patch_magnitude, smoothness = var_fft(patch_gray)

            # Sometimes we will get a divide by zero error in the log() function within the var_fft() function, ignore these samples
            if smoothness > 0:  # variance will be zero if there is a problem in var_fft()

                # convert to contain 3 channels so this patch can be displayed side-by-side with a color patch, for displaying the list of patch samples (used for testing)
                patch_magnitude = cv2.merge([patch_magnitude, patch_magnitude, patch_magnitude])

                # add patch sample data to array
                samples.append((patch_color, smoothness, patch_magnitude, (sample_x, sample_y)))

        current_angle += angle_increment

    # Display the unsorted list of patch samples (testing only, see flag: should_display_sample_images)
    display_sample_images('Unsorted', samples)

    # Sort the array according to the variance (smoothness)
    samples.sort(key=lambda tup: tup[1])
    return samples


def clear_samples_window(new_window_name):
    if should_display_sample_images:
        blank_image = np.zeros(shape=[1, patch_size * 2], dtype='uint8')
        cv2.imshow(new_window_name, blank_image)


def display_sample_images(new_window_name, samples):
    if not should_display_sample_images:
        return

    if len(samples) < 1:
        clear_samples_window(new_window_name)
        return

    #print(samples[0][1])   # smoothness
    sample_plus_fft = cv2.hconcat([samples[0][0], samples[0][2]])
    samples_image = sample_plus_fft
    for i in range(1, len(samples)):
        #print(samples[i][1])   # smoothness
        sample_plus_fft = cv2.hconcat([samples[i][0], samples[i][2]])
        samples_image = cv2.vconcat([samples_image, sample_plus_fft])
    cv2.imshow(new_window_name, samples_image)


def handleMouse(action, x, y, flags, userdata):
    global last_patch
    global last_patch_location

    if action == cv2.EVENT_LBUTTONDOWN:

        # ensure that the click is not too close to the edge, to prevent errors and complicated logic
        if is_point_within_safe_boundary_of_image(x, y):

            last_patch = get_patch(x, y, True)
            last_patch_location = (x, y)

            samples = get_array_of_patches_from_surrounding_area(x, y)
            smoothest_patch = samples[0][0]

            apply_patch(smoothest_patch, (x, y))

            # Display the sorted list of patch samples (testing only, see flag: should_display_sample_images)
            display_sample_images('Sorted', samples)
        else:
            clear_samples_window('Unsorted')
            clear_samples_window('Sorted')

    if action == cv2.EVENT_RBUTTONDOWN:
        # Undo the last blemish removal
        if last_patch_location[0] < 0:
            return
        # replace the last location with the last patch saved
        apply_patch(last_patch, last_patch_location)
        last_patch_location = (-1, -1)


patch_mask = setup_patch_mask()
cv2.setMouseCallback(window_name, handleMouse)

blemish = cv2.imread("blemish.png")

# pre-calculate some image related variables
max_x = blemish.shape[1]
max_y = blemish.shape[0]

cv2.imshow(window_name, blemish)

key = cv2.waitKey(0)
cv2.destroyAllWindows()
