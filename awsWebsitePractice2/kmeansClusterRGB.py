import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

#parts of code taken from https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html

def kmeansClusterRGBRawImage(origImg, k):
    ### Convert to floats instead of the default 8 bits integer coding. Dividing by
    ### 255 is important so that plt.imshow behaves works well on float data (need to
    ### be in the range [0-1])
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
    if(isinstance(np.max(origImg), np.integer)): 
        origImg = np.array(origImg, dtype=np.float64) / 255

        # Load Image and transform to a 2D numpy array.
        w, h, d = original_shape = tuple(origImg.shape)
        image_array = np.reshape(origImg, (w * h, d))
        kmeans = KMeans(n_clusters=k, random_state=0).fit(image_array)

        outputImg = (255*recreate_image(kmeans.cluster_centers_, kmeans.labels_, w, h)).astype('uint8')
        meanColors = kmeans.cluster_centers_*255
        return outputImg, meanColors
    else:
        print("origImg", origImg)
        origImg = np.array(origImg, dtype=np.float64)

        # Load Image and transform to a 2D numpy array.
        w, h, d = original_shape = tuple(origImg.shape)
        image_array = np.reshape(origImg, (w * h, d))
        kmeans = KMeans(n_clusters=k, random_state=0).fit(image_array)
        
        outputImg = (recreate_image(kmeans.cluster_centers_, kmeans.labels_, w, h))
        meanColors = kmeans.cluster_centers_
        return outputImg, meanColors

####################################################################################
########clusterImageFuncton
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
import os
##from kmeansClusterRGB import kmeansClusterRGB

def kmeansClusterRGBImagePath(imageName, k_rgb1):

    filename, file_extension = os.path.splitext(imageName)
    #load image
    img1 = np.array(plt.imread(imageName))
    img2 = np.array(plt.imread(imageName))
    imgCopy = np.array(plt.imread(imageName))

    quantizeRGBImg1, quantizeRGBmeanColors1 = np.array(kmeansClusterRGBRawImage(img1, k_rgb1))
    clusteredFileName = filename+'_kmeansIs'+str(k_rgb1)+file_extension
    print("clusteredFileNameNow", clusteredFileName)
    plt.imsave(clusteredFileName, quantizeRGBImg1)
    return clusteredFileName
