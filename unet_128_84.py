from keras.layers import Dense,Conv2D,Conv2DTranspose,Cropping2D,Input,LeakyReLU,BatchNormalization,MaxPool2D,Concatenate,Dropout
from keras.models import Model
import tensorflow as tf
import pickle as pkl
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    
x=pkl.load(open("X.dat","rb"))
y=pkl.load(open("y_OHE.dat","rb"))
x=np.array(x)
y=np.array(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)


inp=Input(shape=(128,128,3))

lay1=Conv2D(32,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(inp)
lay1=Dropout(.3)(lay1)
lay1=BatchNormalization()(lay1)
lay1=Conv2D(64,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay1)
lay1=Dropout(.3)(lay1)
lay1=BatchNormalization()(lay1)

lay2=MaxPool2D(pool_size=(2,2))(lay1)
lay2=Conv2D(128,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay2)
lay2=Dropout(.3)(lay2)
lay2=BatchNormalization()(lay2)
lay2=Conv2D(128,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay2)
lay2=Dropout(.3)(lay2)
lay2=BatchNormalization()(lay2)

lay3=MaxPool2D(pool_size=(2,2))(lay2)
lay3=Conv2D(256,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay3)
lay3=Dropout(.3)(lay3)
lay3=BatchNormalization()(lay3)
lay3=Conv2D(256,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay3)
lay3=Dropout(.3)(lay3)
lay3=BatchNormalization()(lay3)

lay4=MaxPool2D(pool_size=(2,2))(lay3)
lay4=Conv2D(512,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay4)
lay4=Dropout(.3)(lay4)
lay4=BatchNormalization()(lay4)
lay4=Conv2D(512,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay4)
lay4=Dropout(.3)(lay4)
lay4=BatchNormalization()(lay4)

lay5=Conv2DTranspose(256,(3,3),strides=(2,2),padding='same',activation='relu',kernel_initializer='he_normal')(lay4)
la5=Concatenate()([lay3,lay5])
lay5=Conv2D(256,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay5)
lay5=Dropout(.3)(lay5)
lay5=BatchNormalization()(lay5)
lay5=Conv2D(256,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay5)
lay5=Dropout(.3)(lay5)
lay5=BatchNormalization()(lay5)

lay6=Conv2DTranspose(256,(3,3),strides=(2,2),padding='same',activation='relu',kernel_initializer='he_normal')(lay5)
la6=Concatenate()([lay2,lay6])
lay6=Conv2D(128,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay6)
lay6=Dropout(.3)(lay6)
lay6=BatchNormalization()(lay6)
lay6=Conv2D(64,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay6)
lay6=Dropout(.3)(lay6)
lay6=BatchNormalization()(lay6)

lay7=Conv2DTranspose(32,(3,3),strides=(2,2),padding='same',activation='relu',kernel_initializer='he_normal')(lay6)
la7=Concatenate()([lay1,lay7])
lay7=Conv2D(32,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay7)
lay7=Dropout(.3)(lay7)
lay7=BatchNormalization()(lay7)
lay7=Conv2D(16,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay7)
lay7=Dropout(.3)(lay7)
lay7=BatchNormalization()(lay7)
op=Conv2D(6,(1,1),padding='same',activation='softmax',kernel_initializer='he_normal')(lay7)

model=Model(inp,op)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=15,batch_size=16)
model.save("multiclass_semantic_segmentation")


def test_model(item_no):
    test=np.array(x[item_no])
    test=np.reshape(test,(1,128,128,3))
    p=model.predict(test)[0]
    result=np.argmax(p,axis=2)
    color={"0":[0, 0 ,255],"1":[0, 255 ,255],"2":[255 ,255, 0],"3":[255, 0, 0],"4":[255 ,255 ,255],"5":[0 ,255 ,0]}
    act_result=[]
    for i in result:
        temp=[]
        for j in i:
            temp.append(color[str(j)])
        # temp=np.array(temp)
        act_result.append(temp)
    act_result=np.array(act_result)
    f=plt.figure("actual data")
    plt.imshow(test[0])

    g=plt.figure("predicted data")
    plt.imshow(act_result)
    plt.show()

test_model(30)