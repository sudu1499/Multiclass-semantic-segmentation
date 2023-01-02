import pickle as pkl
from glob import glob 
import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
x=[]
y=[]
for i in glob(r'Images\*'):
    img=cv2.imread(i)
    img=cv2.resize(img,(128,128),interpolation=cv2.INTER_NEAREST)
    x.append(img)

for i in glob(r'labels\*'):
    img=cv2.imread(i)
    img=cv2.resize(img,(128,128),interpolation=cv2.INTER_NEAREST)
    y.append(img)

y_label=[]
for img in  y:
    row=[]
    for i in range(128):
        column=[]
        for j in range(128):
            t=img[i,j,:]
            if (t==(0,0,255)).all():
                column.append(0)

            elif (t==(0,255,255)).all():
                column.append(1)

            elif (t==(255,255,0)).all():
                column.append(2)

            elif (t==(255,0,0)).all():
                column.append(3)

            elif (t==(255,255,255)).all():
                column.append(4)
            else:
                column.append(5)
        row.append(column)
    y_label.append(row)

y_label=np.array(y_label)
# pkl.dump(y_label,open("y_label_encoded.dat",'wb'))

ohe=OneHotEncoder()
ohe.fit([[0],[1],[2],[3],[4],[5]])
y_true=[]
for img in y_label:
    row=[]
    for i in range(128):
        column=[]
        for j in range(128):
            l=ohe.transform([[img[i,j]]]).toarray()
            column.append(l)
        row.append(column)
    y_true.append(row)
y_true=np.array(y_true)
x=np.array(x)
y_true=np.reshape(y_true,(424,128,128,6))
pkl.dump(y_true,open("y_OHE.dat","wb"))
pkl.dump(x,open("X.dat","wb"))





