# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:27:32 2019

@author: sudhanshu
Dimension reduction methods in python---Linear Vs Non linear
"""

'''input the data'''
from __future__ import division
import rasterio
import numpy as np
import os
import sklearn as sk
import gdal
from gdal import *
from tkinter import Tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from gdal import osr

from sklearn.decomposition import PCA
from sklearn import manifold




Tk().withdraw()
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print (filename)
dataset = rasterio.open(filename)
print (dataset)
Crs = dataset.crs
trans = dataset.transform
cols = dataset.width
rows = dataset.height
bands= dataset.count
print (cols,rows,bands)

dataset_list=[]
for b in range(bands):
    with rasterio.open(filename) as f:
        r=f.read(b+1)
        print('read band ='+str(b))
        dataset_list.append(r)


dataset_matrix=np.array(dataset_list)
dataset_matrix=np.reshape(dataset_matrix,(bands,(rows*cols))).T

#PCA DR method
n_components = 4
pca = PCA(n_components, whiten = False)
param= pca.fit(dataset_matrix)
dataset_pca = pca.fit_transform(dataset_matrix)
print(pca.explained_variance_ratio_)  
print (np.sum(pca.explained_variance_ratio_))
outdata_inter=dataset_pca.T
outdata=np.reshape(outdata_inter,(n_components,rows,cols))



#LLE DR method

n_neighbors = 5
n_components = 4
dataset_lle, err = manifold.locally_linear_embedding(dataset_matrix,n_neighbors,n_components, eigen_solver='auto',method='standard')
print("Reconstruction error: %g" % err)
outdata_inter=dataset_lle
outdata=np.reshape(outdata_inter,(n_components,rows,cols))
                                     
                                        


#ISOMAP Method
from sklearn.manifold import Isomap
n_neighbors = 5
n_components = 4
dataset_isomap = manifold.Isomap(n_neighbors,n_components).fit_transform(dataset_matrix)
outdata_inter=dataset_isomap
outdata=np.reshape(outdata_inter,(n_components,rows,cols))

#write the file
cor=outdata
dst_filename = 'isomap.tif'
dataset = gdal.Open(filename, gdal.GA_ReadOnly)
gdal_datatype = gdal.GDT_Float32
np_datatype = np.float32
driver = gdal.GetDriverByName( "GTiff" )
originX, pixelWidth, b, originY, d, pixelHeight = dataset.GetGeoTransform() 
dst_ds = driver.Create( dst_filename, cor.shape[2], cor.shape[1], cor.shape[0], gdal_datatype )
dst_ds.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
for i, image in enumerate(cor, 1):
        dst_ds.GetRasterBand(i).WriteArray( image )
prj=dataset.GetProjection()
outRasterSRS = osr.SpatialReference(wkt=prj)
dst_ds.SetProjection(outRasterSRS.ExportToWkt())
dst_ds.FlushCache()
dst_ds = None


