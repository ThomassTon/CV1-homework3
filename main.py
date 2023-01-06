import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from PIL import Image

import os




from problem1 import *
def problem1():
    """Example code implementing the steps in Problem 2"""

    def show_images(ims, hw, title='', size=(8, 2)):
        assert ims.shape[0] < 10, "Too many images to display"
        n = ims.shape[0]

        # visualising the result
        fig = plt.figure(figsize=size)
        for i, im in enumerate(ims):
            fig.add_subplot(1, n, i + 1)
            plt.imshow(im.reshape(*hw), "gray")
            plt.axis("off")
        fig.suptitle(title)


    # Load images
    imgs = load_faces("data/yale_faces")
    y = vectorize_images(imgs)
    hw = imgs.shape[1:]
    print("Loaded array: ", y.shape)

    # Using 2 random images for testing
    test_face2 = y[0, :]
    test_face = y[-1, :]
    show_images(np.stack([test_face, test_face2], 0), hw,  title="Sample images")

    # Compute PCA
    mean_face, u, cumul_var = compute_pca(y)

    # Compute PCA reconstruction
    # percentiles of total variance
    ps = [0.5, 0.75, 0.9, 0.95]
    ims = []
    for i, p in enumerate(ps):
        b = basis(u, cumul_var, p)
        a = compute_coefficients(test_face2, mean_face, b)
        ims.append(reconstruct_image(a, mean_face, b))

    show_images(np.stack(ims, 0), hw, title="PCA reconstruction")

    # fix some basis
    b = basis(u, cumul_var, 0.95)

    # Image search
    top5 = search(y, test_face2, b, mean_face, 5)
    show_images(top5, hw, title="Image Search")

    # Interpolation
    ints = interpolate(test_face2, test_face, b, mean_face, 5)
    show_images(ints, hw, title="Interpolation")

    plt.show()




from problem2 import *
def problem2():
    def load_img(path):
        color = Image.open(path)
        gray = color.convert("L")
        color = np.array(color) / 255
        gray = np.array(gray) / 255
        return color, gray

    def show_points(img, rows, cols):
        plt.imshow(img, interpolation="none")
        plt.plot(cols, rows ,"xr", linewidth=8)
        plt.axis("off")


    def plot_heatmap(img, title=""):
        plt.imshow(img, "jet", interpolation="none")
        plt.axis("off")
        plt.title(title)


    # Set paramters and load the image
    sigma = 2
    threshold = 3e-3
    img_dir = os.path.join("data", "a3p2")
    imgs_data = {}

    for img_name in os.listdir(img_dir):
        color, gray = load_img(os.path.join(img_dir, img_name))

        # Generate filters and compute Hessian
        fx, fy = derivative_filters()
        gauss = gauss2d(sigma, (10, 10))
        I_xx, I_yy, I_xy = compute_hessian(gray, gauss, fx, fy)




        # Show components of Hessian matrix
        plt.figure()
        plt.subplot(1,4,1)
        plot_heatmap(I_xx, "I_xx")
        plt.subplot(1,4,2)
        plot_heatmap(I_yy, "I_yy")
        plt.subplot(1,4,3)
        plot_heatmap(I_xy, "I_xy")

        # Compute and show Hessian criterion
        criterion = compute_criterion(I_xx, I_yy, I_xy, sigma)
        plt.subplot(1,4,4)
        plot_heatmap(criterion, "Determinant of Hessian")

        # Show all interest points where criterion is greater than threshold
        rows, cols = np.nonzero(criterion > threshold)
        plt.figure()
        show_points(color, rows, cols)

        # Apply non-maximum suppression and show remaining interest points
        rows, cols = nonmaxsuppression(criterion, threshold)
        plt.figure()
        show_points(color, rows, cols)
        plt.show()

        # Get image patches around feature points as local descriptors
        descriptors = imagepatch_descriptors(gray, rows, cols)

        # Save computed interest points
        imgs_data[img_name] = [color, gray, rows, cols, descriptors]


    # Load image data, interest points and descriptors
    color1, _, rows1, cols1, descriptors1 = imgs_data["a3p2_0.png"]
    color2, _, rows2, cols2, descriptors2 = imgs_data["a3p2_1.png"]
    color3, _, rows3, cols3, descriptors3 = imgs_data["a3p2_2.png"]

    # Show images
    plt.figure()
    plt.subplot(1, 3, 1)
    show_points(color1, rows1, cols1)
    plt.axis("off")
    plt.subplot(1, 3, 2)
    show_points(color2, rows2, cols2)
    plt.axis("off")
    plt.subplot(1, 3, 3)
    show_points(color3, rows3, cols3)
    plt.axis("off")

    # Get matched interest points for image pairs
    matches_12 = match_interest_points(descriptors1, descriptors2)
    matches_13 = match_interest_points(descriptors1, descriptors3)

    # Show matched interest points
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title("Matched interst points for a3p2_0.png and a3p2_1.png")
    plt.imshow(np.hstack((color1, color2)))
    for idx1, idx2 in matches_12[:15]:
        plt.plot([cols1[idx1],cols2[idx2]+color1.shape[1]], [rows1[idx1],rows2[idx2]],'xr--')
    plt.axis("off")
    plt.subplot(2, 1, 2)
    plt.title("Matched interst points for a3p2_0.png and a3p2_2.png")
    plt.imshow(np.hstack((color1, color3)))
    for idx1, idx2 in matches_13[:15]:
        plt.plot([cols1[idx1],cols3[idx2]+color1.shape[1]], [rows1[idx1],rows3[idx2]],'xr--')
    plt.axis("off")
    plt.show()




if __name__ == "__main__":
  problem1()
  problem2()


