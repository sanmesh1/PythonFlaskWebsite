###########################################################################
########original code
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
##from clusterImage import clusterImage
from flask import render_template

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'jpg', 'png'}

application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
application.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024    # 2 Mb limit
##application.config['MAX_CONTENT_LENGTH'] = 14 * 1024    # 2 Mb limit

@application.errorhandler(413)
def error413(e):
    return redirect(url_for('upload_file'))
                    
#############################################################################
##########QuantizeRGBFunction
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
##
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
##import quantizeRGB

def clusterImage(imageName, k_rgb1):

    ##quantizeRGB#####################################
    #k_rgb1 = 8
    #imageName = 'beautiful'


    filename, file_extension = os.path.splitext(imageName)
    #load image
    img1 = np.array(plt.imread(imageName))
    img2 = np.array(plt.imread(imageName))
    imgCopy = np.array(plt.imread(imageName))

    quantizeRGBImg1, quantizeRGBmeanColors1 = np.array(quantizeRGB(img1, k_rgb1))
##    clusteredFileName = imageName[0:-4]+'_kmeansIs'+str(k_rgb1)+imageName[-4:]
    clusteredFileName = filename+'_kmeansIs'+str(k_rgb1)+file_extension
    print("clusteredFileNameNow", clusteredFileName)
    plt.imsave(clusteredFileName, quantizeRGBImg1)
    return clusteredFileName
###########################################################################

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@application.route('/', methods=['GET', 'POST'])
def upload_file():
##############################
#clean files
    import os, time, sys
    print("Entered clean files")

    now = time.time()
    path = "static"
    numMinutes = 10

    for f in os.listdir(path):
        f1 = os.path.join(path, f)
        if (now - os.stat(f1).st_mtime) > 60*numMinutes:
            if os.path.isfile(f1) and os.path.isdir(f1) == False and f1.lower().endswith(('.png', '.jpg', '.jpeg')):
                os.remove(f1)
###clean files
################################

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        clusters = request.form['quantity']
        print(clusters)
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename, clusters = clusters))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <br>
      Number of Clusters: <input type="number" name="quantity" min="1" max="30">
      <br>
      <input type=submit value=Upload>
    </form>
    '''
from flask import send_from_directory

@application.route('/uploads/<filename>/<clusters>')
def uploaded_file(filename, clusters):
    clusters = int(clusters)
    print("filename", filename)
    clusteredFileName = clusterImage("static/"+filename, clusters)
    print('/' + clusteredFileName)
    print('/' + "static/"+filename)
    return render_template("uploaded.html", input_image = '/' + "static/"+filename, clustered_image =  '/' + clusteredFileName)
if __name__ == "__main__":
    application.debug = True
    application.run()	

