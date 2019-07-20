# -*- coding: utf-8 -*-
"""
DR+SVM classification

@author: Sudhanshu
"""
from sklearn import svm
import file_op
import rasterio
import numpy as np
import os
import gdal
import ntpath
from gdal import osr
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import h5py

#--------------------------------------------------------------------------------------------------------------------------
#reading mat file
from scipy.io import loadmat
im_file=file_op.file_op('OPEN THE IMAGE TO BE CLASSIFIED')
im_dict = loadmat(im_file)
mat_image=im_dict['paviaU'] #lines (rows), samples (cols),bands
rows=mat_image.shape[0]
cols=mat_image.shape[1]
bands=mat_image.shape[2]

cor=mat_image.T
plt.imshow(cor[1,:,:].T)
c=[]
for i in range(cor.shape[0]):
    c.append(cor[i,:,:].T)

c=np.array(c)

out_file=im_file.split('/')[-1]
dst_filename = out_file+str('.tif')
corr=c
gdal_datatype = gdal.GDT_Float32
np_datatype = np.float32
driver = gdal.GetDriverByName( "GTiff" )

dst_ds = driver.Create( dst_filename, cols, rows, bands, gdal_datatype ) #cols,rows bands
for i, image in enumerate(corr, 1):
        dst_ds.GetRasterBand(i).WriteArray( image )
dst_ds.FlushCache()
dst_ds = None

#reading the MAT GT
from scipy.io import loadmat
im_file=file_op.file_op('OPEN THE IMAGE TO BE CLASSIFIED')
im_dict = loadmat(im_file)
mat_image=im_dict['salinasA_gt']
cor=mat_image
out_file=im_file.split('/')[-1]
dst_filename = out_file+str('.tif')
rows=cor.shape[0]
cols=cor.shape[1]
output_data = rasterio.open(dst_filename, 'w', driver='GTiff', height=rows, width=cols, count=1,dtype=mat_image.dtype)
output_data.write(cor, 1)
output_data.close()



#for matlab grater than 7.5 using h5py
arrays = {}
f = h5py.File(im_file)
for k, v in f.items():
    arrays[k] = np.array(v)
mat_image= np.array(arrays['sgt']) #check the sizes properly

#-----------------------------------------------------------------------------------------------------------------------------
#open the image file

im_file=file_op.file_op('OPEN THE IMAGE TO BE CLASSIFIED')
img_ds = gdal.Open(im_file, gdal.GA_ReadOnly)
rows=img_ds.RasterYSize
cols=img_ds.RasterXSize
bands=img_ds.RasterCount


#open the roi class image
ROI_file=file_op.file_op('OPEN THE ROI IMAGE')
roi_ds = gdal.Open(ROI_file, gdal.GA_ReadOnly)
roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
 


#run the DR method
dataset_matrix=np.array(img)
dataset_matrix=np.reshape(dataset_matrix,(bands,(rows*cols))).T
DRC=10
pca = PCA(n_components=DRC, whiten = False)
param= pca.fit(dataset_matrix)
dataset_pca = pca.fit_transform(dataset_matrix)
print(pca.explained_variance_ratio_)  
print (np.sum(pca.explained_variance_ratio_))
outdata_inter=np.reshape(dataset_pca,(rows,cols,DRC))
DR_reduced=outdata_inter
#display the roi image and input image
plt.subplot(121)
plt.imshow(img[:, :, 4], cmap=plt.cm.Greys_r)
plt.title('Input Image')
plt.subplot(122)
plt.imshow(roi, cmap=plt.cm.Spectral)
plt.title('ROI Training Data')

#preparing X and y data set for DR reduced data
X = DR_reduced[roi > 0,:] 
y = roi[roi > 0]


#normal classification without DR
X = img[roi > 0,:] 
y = roi[roi > 0]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5) # 70% training and 30% test (random_state=42)
from sklearn import svm
#Create a svm Classifier
clf = svm.SVC(gamma='scale',C=100) # Linear Kernel


#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


#predict for complete image
"1. resize the image  to 2-d array of dims [(rows*cols),bands]
image_pred_mat= np.reshape(img, ((rows*cols),bands)) #without DR

image_pred_mat= np.reshape(DR_reduced, ((rows*cols),DRC)) #with DR


class_prediction = clf.predict(image_pred_mat)
classified=np.reshape(class_prediction,(rows,cols))
plt.imshow(classified)
#output_data = rasterio.open(str('classified_image') + '.tif', 'w', driver='GTiff', height=rows, width=cols, count=1,dtype=classified.dtype, crs=Crs, transform=trans)
output_data = rasterio.open(str('classified_image') + '.tif', 'w', driver='GTiff', height=rows, width=cols, count=1,dtype=classified.dtype)
output_data.write(classified, 1)
output_data.close()


#visualization
def color_stretch(image, index, minmax=(0, 10000)):
    colors = image[:, :, index].astype(np.float64)

    max_val = minmax[1]
    min_val = minmax[0]

    # Enforce maximum and minimum values
    colors[colors[:, :, :] > max_val] = max_val
    colors[colors[:, :, :] < min_val] = min_val

    for b in range(colors.shape[2]):
        colors[:, :, b] = colors[:, :, b] * 1 / (max_val - min_val)
        
    return colors
    
img543 = color_stretch(img, [4, 3, 2], (0, 8000))

# See https://github.com/matplotlib/matplotlib/issues/844/
n = class_prediction.max()
# Next setup a colormap for our map
colors = dict((
    (0, (0, 0, 0, 255)),  # Nodata
    (1, (0, 150, 0, 255)),  # veg
    (2, (0, 0, 255, 255)),  # urban
#    (3, (0, 255, 0, 255)),  # Herbaceous
#    (4, (160, 82, 45, 255)),  # Barren
#    (5, (255, 0, 0, 255))  # Urban
))
# Put 0 - 255 as float 0 - 1
for k in colors:
    v = colors[k]
    _v = [_v / 255.0 for _v in v]
    colors[k] = _v
    
index_colors = [colors[key] if key in colors else 
                (255, 255, 255, 0) for key in range(1, n + 1)]
cmap = plt.matplotlib.colors.ListedColormap(index_colors, 'Classification', int(n))

# Now show the classmap next to the image
plt.subplot(121)
plt.imshow(img543)

plt.subplot(122)
plt.imshow(classified, cmap=cmap, interpolation='none')

plt.show()


#confusion matirx
import pandas as pd
df = pd.DataFrame()
df['truth'] = y
df['predict'] = rf.predict(X)
print(pd.crosstab(df['truth'], df['predict'], margins=True))



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
classes = classes[unique_labels(y_test, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)