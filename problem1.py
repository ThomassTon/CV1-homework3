import numpy as np
import os
from PIL import Image


def load_faces(path, ext=".pgm"):
    """Load faces into an array (N, H, W),
    where N is the number of face images and
    H, W are height and width of the images.

    Hint: os.walk() supports recursive listing of files
    and directories in a path

    Args:
        path: path to the directory with face images
        ext: extension of the image files (you can assume .pgm only)

    Returns:
        imgs: (N, H, W) numpy array
    """

    #
    imgs=[]
    for root, dirs, files in os.walk(path, topdown=True, followlinks=ext):
      for name in files:
        im = Image.open(os.path.join(root, name))
        imgs.append(np.asarray(im))

    imgs = np.asarray(imgs)
    return imgs
    #


def vectorize_images(imgs):
    """Turns an  array (N, H, W),
    where N is the number of face images and
    H, W are height and width of the images into
    an (N, M) array where M=H*W is the image dimension.

    Args:
        imgs: (N, H, W) numpy array

    Returns:
        x: (N, M) numpy array
    """
    #
    N = imgs.shape[0]
    M = imgs.shape[1] * imgs.shape[2]
    x = np.zeros((N, M))
    for i in range(N):
      x[i] = imgs[i].reshape(M, )
    return x
    #


def compute_pca(X):
    """PCA implementation

    Args:
        X: (N, M) an numpy array with N M-dimensional features

    Returns:
        mean_face: (M,) numpy array representing the mean face
        u: (M, M) numpy array, bases with D principal components
        cumul_var: (N, ) numpy array, corresponding cumulative variance
    """

    #

    N = X.shape[0]  # 760
    M = X.shape[1]  # 8064
    u = np.zeros((M,M))
    cumul_var=np.zeros((N,))
    X0=np.transpose(X)
    X1=np.zeros((X0.shape))
    mean_face=np.mean(X0,axis=1)
    for i in range(N):
      X1[:,i]=X0[:,i]-mean_face
    U, S, VT = np.linalg.svd(X1)
    for i in range(N):
      cumul_var[i]=S[i]**2/N

    return mean_face, U, cumul_var
    #


def basis(u, cumul_var, p = 0.5):
    """Return the minimum number of basis vectors
    from matrix U such that they account for at least p percent
    of total variance.

    Hint: Do the singular values really represent the variance?

    Args:
        u: (M, M) numpy array containing principal components.
        For example, i'th vector is u[:, i]
        cumul_var: (N, ) numpy array, variance along the principal components.

    Returns:
        v: (M, D) numpy array, contains M principal components from N
        containing at most p (percentile) of the variance.

    """

    #
    N=cumul_var.shape[0]
    lm = cumul_var
    lm_D = 0
    Ds = []
    for i in range(N):
      lm_D += lm[i]
      if lm_D >= p * sum(lm):
        Ds.append(i)
    D = min(Ds)
    v = u[:, :D]
    return v
    #


def compute_coefficients(face_image, mean_face, u):
    """Computes the coefficients of the face image with respect to
    the principal components u after projection.

    Args:
        face_image: (M, ) numpy array (M=h*w) of the face image a vector
        mean_face: (M, ) numpy array, mean face as a vector
        u: (M, D) numpy array containing D principal components.
        For example, (:, 1) is the second component vector.

    Returns:
        a: (D, ) numpy array, containing the coefficients
    """
    #
    m=np.transpose(u)
    x = face_image-mean_face
    a = np.dot(m,x)
    return a


def reconstruct_image(a, mean_face, u):
    """Reconstructs the face image with respect to
    the first D principal components u.

    Args:
        a: (D, ) numpy array containings the image coefficients w.r.t
        the principal components u
        mean_face: (M, ) numpy array, mean face as a vector
        u: (M, D) numpy array containing D principal components.
        For example, (:, 1) is the second component vector.

    Returns:
        image_out: (M, ) numpy array, projected vector of face_image on
        principal components
    """

    #
    sum =0
    for i in range(a.shape[0]):
      sum+= a[i]*u[:,i]
    image_out=mean_face+sum
    return image_out



def compute_similarity(Y, x, u, mean_face):
    """Compute the similarity of an image x to the images in Y
    based on the cosine similarity.

    Args:
        Y: (N, M) numpy array with N M-dimensional features
        x: (M, ) image we would like to retrieve
        u: (M, D) bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector


    Returns:
        sim: (N, ) numpy array containing the cosine similarity values
    """
    #
    N = Y.shape[0]
    sim = np.zeros((N,))
    a = compute_coefficients(x,mean_face,u)
    out = reconstruct_image(a,mean_face,u)
    for i in range(N):
      sim[i]=np.dot(Y[i],out)/(np.linalg.norm(Y[i])*np.linalg.norm(out))

    return sim
    #


def search(Y, x, u, mean_face, top_n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.

    Args:
        Y: (N, M) numpy array with N M-dimensional features
        x: (M, ) numpy array, image we would like to retrieve
        u: (M, D) numpy arrray, bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector
        top_n: integer, return top_n closest images in L2 sense.

    Returns:
        Y: (top_n, M) numpy array containing the top_n most similar images
        sorted by similarity
    """
    #
    sim = compute_similarity(Y, x, u, mean_face)
    M=Y.shape[1]
    Y1=np.zeros((top_n,M))
    index = np.argsort(sim)
    for i in range(top_n):
      Y1[i,:]=Y[index[i],:]

    return Y1
    #


def interpolate(x1, x2, u, mean_face, n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.

    Args:
        x1: (M, ) numpy array, the first image
        x2: (M, ) numpy array, the second image
        u: (M, D) numpy array, bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector
        n: number of interpolation steps (including x1 and x2)

    Hint: you can use np.linspace to generate n equally-spaced points on a line

    Returns:
        Y: (n, M) numpy arrray, interpolated results.
        The first dimension is in the index into corresponding
        image; Y[0] == project(x1, u); Y[-1] == project(x2, u)
    """

    #
    a1 = compute_coefficients(x1, mean_face, u)
    a2 = compute_coefficients(x2, mean_face, u)
    M = x1.shape[0]
    Y = np.zeros((n, M))
    D = a1.shape[0]
    a3 = np.zeros((n, D))
    for i in range(D):
      a3[:, i] = np.linspace(a1[i], a2[i], n)
    for i in range(n):
      Y[i,:]= reconstruct_image(a3[i], mean_face, u)
    return Y

    #
