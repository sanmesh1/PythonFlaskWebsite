import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

#parts of code taken from https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html

def quantizeRGB(origImg, k):
    ### Convert to floats instead of the default 8 bits integer coding. Dividing by
    ### 255 is important so that plt.imshow behaves works well on float data (need to
    ### be in the range [0-1])
    origImg = np.array(origImg, dtype=np.float64) / 255

    # Load Image and transform to a 2D numpy array.
    w, h, d = original_shape = tuple(origImg.shape)
    image_array = np.reshape(origImg, (w * h, d))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(image_array)

    def recreate_image(codebook, labels, w, h):
        """Recreate the (compressed) image from the code book & labels"""
        d = codebook.shape[1]
        image = np.zeros((w, h, d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = codebook[labels[label_idx]]
                label_idx += 1
        return image


    outputImg = (255*recreate_image(kmeans.cluster_centers_, kmeans.labels_, w, h)).astype('uint8')
    meanColors = kmeans.cluster_centers_*255
    return outputImg, meanColors
