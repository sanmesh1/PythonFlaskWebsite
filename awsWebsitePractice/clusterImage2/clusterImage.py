import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
import quantizeRGB

def clusterImage(imageName, k_rgb1):

    ##quantizeRGB#####################################
    #k_rgb1 = 8
    #imageName = 'beautiful'

    #load image
    img1 = np.array(plt.imread(imageName + '.jpg'))
    img2 = np.array(plt.imread(imageName + '.jpg'))
    imgCopy = np.array(plt.imread(imageName + '.jpg'))

    quantizeRGBImg1, quantizeRGBmeanColors1 = np.array(quantizeRGB.quantizeRGB(img1, k_rgb1))
    clusteredFileName = imageName+'_kmeansIs'+str(k_rgb1)+'.jpg'
    plt.imsave(clusteredFileName, quantizeRGBImg1)
    return clusteredFileName
if __name__ == "__main__":
    clusterImage("upload/houseMiyazaki", 3)
