import numpy as np
from scipy.ndimage import convolve, maximum_filter
import cv2


def gauss2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (w, h) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (h, w) np.array
    """
    m, n = fsize
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)


def derivative_filters():
    """ Create derivative filters for x and y direction

    Returns:
        fx: derivative filter in x direction
        fy: derivative filter in y direction
    """
    fx = np.array([[0.5, 0, -0.5]])
    print(fx.shape)
    fy = fx.transpose()
    return fx, fy


def compute_hessian(img, gauss, fx, fy):
    """ Compute elements of the Hessian matrix

    Args:
        img:
        gauss: Gaussian filter
        fx: derivative filter in x direction
        fy: derivative filter in y direction

    Returns:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
    """
    #

    img0 = convolve(img, gauss, mode='mirror')
    I_x = convolve(img0, fx, mode='mirror')
    I_xx = convolve(I_x, fx, mode='mirror')
    I_y = convolve(img0, fy, mode='mirror')
    I_yy = convolve(I_y, fy, mode='mirror')
    I_xy = convolve(I_x, fy, mode='mirror')
    # print(I_xx.shape)  h=400, w= 300
    return I_xx, I_yy, I_xy
    #



def compute_criterion(I_xx, I_yy, I_xy, sigma):
    """ Compute criterion function

    Args:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
        sigma: scaling factor

    Returns:
        criterion: (h, w) np.array of scaled determinant of Hessian matrix
    """
    #
    H = I_xx * I_yy - I_xy * I_xy
    H = sigma ** 4 * H
    return H
    #


def nonmaxsuppression(criterion, threshold):
    """ Apply non-maximum suppression to criterion values
        and return Hessian interest points

        Args:
            criterion: (h, w) np.array of criterion function values
            threshold: criterion threshold
        Returns:
            rows: (n,) np.array with y-positions of interest points
            cols: (n,) np.array with x-positions of interest points
    """

    #
    rows = []
    cols = []
    h, w = criterion.shape
    L0 = maximum_filter(criterion, size=(10,10), mode='mirror')
    nonmaxCriterion = np.zeros(criterion.shape)
    nonmaxCriterion[np.where(criterion == L0)] = criterion[np.where(criterion == L0)]
    rows, cols = np.nonzero(nonmaxCriterion > threshold)
    return rows, cols

    #


def imagepatch_descriptors(gray, rows, cols):
    """ Get image patch descriptors for every interes point

        Args:
            img: (h, w) np.array with image gray values
            rows: (n,) np.array with y-positions of interest points
            cols: (n,) np.array with x-positions of interest points
        Returns:
            descriptors: (n, patch_size**2) np.array with image patch feature descriptors
    """

    #
    n, = rows.shape
    des = np.zeros((n, 11 ** 2))
    h, w = gray.shape
    for i in range(n):
      d = []
      for j in range(11):
        for k in range(11):
          r = rows[i] - 5 + j
          c = cols[i] - 5 + k
          if r >= h:
            r = 2 * h - 1 - r
          if c >= w:
            c = 2 * w - 1 - c
          d.append(gray[r, c])

      des[i, :] = np.asarray(d)

    return des
    #


def match_interest_points(descriptors1, descriptors2):
    """ Brute-force match the interest points descriptors of two images using the cv2.BFMatcher function.
    Select a reasonable distance measurement to be used and set "crossCheck=True".

    Args:
        descriptors1: (n, patch_size**2) np.array with image patch feature descriptors
        descriptors2: (n, patch_size**2) np.array with image patch feature descriptors
    Returns:
        matches: (m) list of matched descriptor pairs
    """
    #
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    des1 = descriptors1.astype("float32")
    des2 = descriptors2.astype("float32")
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    mat = []
    for i in range(len(matches)):
      mat.append([matches[i].queryIdx, matches[i].trainIdx])
    return mat
    #
