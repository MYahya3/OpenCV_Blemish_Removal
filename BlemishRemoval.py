import cv2 as cv2
import numpy as np

# Returns patches of image from around our blemished area
def find_neighbor_patches(center_patch_coords,number_of_neighbors,radius):
    x = center_patch_coords[0]
    y = center_patch_coords[1]

    # Finding specific number of evenly spaced points around our patch
    t = np.linspace(0, 2 * np.pi, number_of_neighbors + 1)  # +1 because, the first and the last one will be the same
    neighbor_x = (np.round(2 * radius * np.cos(t)) + x).astype(int)
    neighbor_y = (np.round(2 * radius * np.sin(t)) + y).astype(int)
    neighbor_centers = np.c_[neighbor_x, neighbor_y][:-1]  # We remove the last one

    # Remove the patches that dont fit into our image
    condition = np.array([0 + radius < center[0] < image.shape[1] - radius
                          and 0 + radius < center[1] < image.shape[0] - radius
                          for center in neighbor_centers])
    neighbor_centers_filtered = neighbor_centers[condition]

    # These are neighbor patches that we get using the center coordinates and radius,
    # which we use here as half of the square side length
    neighbor_patches = [image[y - radius:y + radius, x - radius:x + radius] for [x, y] in neighbor_centers_filtered]

    return neighbor_patches

# Calculates a single number gradient score (vertical or horizontal) for a single image
def calculate_mean_gradient(image_patch, xorder, yorder):
    sobel = cv2.Sobel(image_patch, cv2.CV_64F, xorder, yorder, ksize=3)
    abs_sobel = np.abs(sobel)  # Because sobel results can be negative
    mean_gradient = np.mean(np.uint8(abs_sobel))
    return mean_gradient

# Calculates single number gradient score for every patch
def calculate_neighbor_gradient_measures(patches):
    # Calculate horizontal and vertical sobel gradients:
    sobel_x = [calculate_mean_gradient(patch, 1, 0) for patch in patches]
    sobel_y = [calculate_mean_gradient(patch, 0, 1) for patch in patches]

    total_gradients = np.add(sobel_x, sobel_y)

    return total_gradients

number_of_neighbors = 30
radius = 15
cursor_radius = 15  # You can adjust this value as needed
# Clears blemish around place where point and click our cursor
def clear_blemish(action, x, y, flags, userdata):
    global image
    global cursor_radius

    if action == cv2.EVENT_MOUSEMOVE:
        # This part draws a circle at the cursor's position
        cursor_image = image.copy()
        cv2.circle(cursor_image, (x, y), cursor_radius, (0, 0, 255), 2)  # Circle in red color

        cv2.imshow("Window", cursor_image)
    elif action == cv2.EVENT_LBUTTONDOWN:
        blemish_center = (x, y)
        patch_fits = 0 + radius < x < image.shape[1] - radius and 0 + radius < y < image.shape[0] - radius

        if not patch_fits:
            return

        neighbor_patches = find_neighbor_patches(blemish_center, number_of_neighbors=number_of_neighbors, radius=radius)

        neighbor_gradients = calculate_neighbor_gradient_measures(neighbor_patches)

        minimum_gradient_index = np.argmin(neighbor_gradients)
        minimum_gradient_patch = neighbor_patches[int(minimum_gradient_index)]

        mask = np.full_like(minimum_gradient_patch, 255, dtype=minimum_gradient_patch.dtype)

        image = cv2.seamlessClone(minimum_gradient_patch, image, mask, blemish_center, cv2.NORMAL_CLONE)

        cv2.imshow("Window", image)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Blemish Removal Script")
    parser.add_argument("image_path", help="Path to the image file")
    args = parser.parse_args()

    # Read the image from the provided path
    image = cv2.imread(args.image_path, 1)
    image = cv2.resize(image, (0,0), fx=2, fy=2)
    if image is None:
        print("Error: Could not load the image.")
        exit(1)

    # Make a dummy image, will be useful to reset the photo
    dummy = image.copy()

    # Create window and show image
    window_name = cv2.namedWindow("Window")
    cv2.imshow("Window", image)

    # highgui function called when mouse events occur
    cv2.setMouseCallback("Window", clear_blemish)

    k = 0
    # loop until the escape character is pressed
    while k != 27:
        k = cv2.waitKey(20) & 0xFF

        if k == 99:
            image = dummy.copy()
            cv2.imshow("Window", image)

    cv2.destroyAllWindows()