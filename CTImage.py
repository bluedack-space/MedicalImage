#from pydicom import dcmread
#from pydicom.data import get_testdata_file
#import matplotlib
#import matplotlib.pyplot as plt
#ds = dcmread("D:\\Development\\Python\\MedicalImage\\Images\\vhf.1502.dcm")
#print(ds)
#plt.imshow(ds.pixel_array)
#plt.show()

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob

#---------------------------------------------------------------------------------------------
# (01) Read Image Files
files = []
for fname in glob.glob("D:\\Development\\Python\\MedicalImage\\Images\\Head\\*", recursive=False):
    files.append(pydicom.dcmread(fname))

#---------------------------------------------------------------------------------------------
# (02) Generate Slices
slices = []
skipcount = 0
for f in files:
    InstanceNumber = str(f.InstanceNumber)
    if InstanceNumber.isnumeric():
        slices.append(f)
    else:
        skipcount = skipcount + 1

#---------------------------------------------------------------------------------------------
# (03) Evaluate Aspect Ratio
ps = slices[0].PixelSpacing
ss = slices[0].SliceThickness
ax_aspect = ps[0]/ps[1]
sag_aspect = ss/ps[0]
cor_aspect = ss/ps[1]

#---------------------------------------------------------------------------------------------
# (04) Generate 3D Empty Numpy Array 
img_shape = list(slices[0].pixel_array.shape)
img_shape.insert(0,len(slices))
img3d = np.zeros(img_shape)

#---------------------------------------------------------------------------------------------
# (05) Generate 3D Numpy Array 
for i, s in enumerate(slices):
    img2d = s.pixel_array
    img3d[i,:, :] = img2d

#---------------------------------------------------------------------------------------------
# (06) Do Plot
middle0 = img_shape[0]//2
middle1 = img_shape[1]//2
middle2 = img_shape[2]//2

a1 = plt.subplot(1, 3, 1)
plt.imshow(img3d[middle0,:, :])
a1.set_aspect(ax_aspect)

a2 = plt.subplot(1, 3, 2)
plt.imshow(img3d[:, middle1, :])
a2.set_aspect(cor_aspect)

a3 = plt.subplot(1, 3, 3)
plt.imshow(img3d[:, :, middle2]) # 
a3.set_aspect(sag_aspect)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

#---------------------------------------------------------------------------------------------
# (07) Run Marching Cube Method
verts, faces, normals, values = measure.marching_cubes_lewiner(img3d,level=400,spacing=(3,0.57, 0.57))

#fig = plt.figure(figsize=(10, 10))
#ax  = fig.add_subplot(111, projection='3d')
#mesh = Poly3DCollection(verts[faces])
#mesh.set_edgecolor('k')
#ax.add_collection3d(mesh)
#
#ax.set_xlabel("x-axis")
#ax.set_ylabel("y-axis")
#ax.set_zlabel("z-axis") 
#
#ax.set_xlim(0, 800)  
#ax.set_ylim(0, 800)  
#ax.set_zlim(0, 800)  
#
#plt.tight_layout()
#plt.show()

import numpy as np
from stl import mesh
mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        mesh.vectors[i][j] = verts[f[j],:]
mesh.save('mesh.stl')
