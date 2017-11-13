import cv2
import tensorflow as tf
import numpy as np
from os.path import isfile, join
#image = tf.placeholder(tf.float32, shape=[None, 40, 40])
#landmark = tf.placeholder(tf.float32, shape=[None, 10])
#gender = tf.placeholder(tf.float32, shape=[None, 2])
#smile = tf.placeholder(tf.float32, shape=[None, 2])
#glasses = tf.placeholder(tf.float32, shape=[None, 2])
#headpose = tf.placeholder(tf.float32, shape=[None, 5])


mypath = "/media/dongy/Windows7_OS/Users/Owner/Desktop/Life with Divine/MTFL/"
train= "training.txt"
#i,l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,g,s,gl,p = np.genfromtxt("/media/dongy/Windows7_OS/Users/Owner/Desktop/Life with Divine/MTFL/training.txt", delimiter=" ", unpack=True)

i,l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,g,s,gl,p = np.genfromtxt(join(mypath,train), delimiter=" ", unpack=True)
i= np.genfromtxt(join(mypath, train), delimiter=" ",usecols=0, dtype=str, unpack=True)


onlyfiles = [ f for f in i if isfile(join(mypath,f))]

images = np.empty(len(onlyfiles),dtype=object)
for n in range(0, 1000):
    images[n] = cv2.resize(cv2.imread( join(mypath, onlyfiles[n]),0),(40,40))




l1=np.transpose(np.reshape(l1,(-1,10000)))
l2=np.transpose(np.reshape(l2,(-1,10000)))
l3=np.transpose(np.reshape(l3,(-1,10000)))
l4=np.transpose(np.reshape(l4,(-1,10000)))
l5=np.transpose(np.reshape(l5,(-1,10000)))
l6=np.transpose(np.reshape(l6,(-1,10000)))
l7=np.transpose(np.reshape(l7,(-1,10000)))
l8=np.transpose(np.reshape(l8,(-1,10000)))
l9=np.transpose(np.reshape(l9,(-1,10000)))
l10=np.transpose(np.reshape(l10,(-1,10000)))
g=np.transpose(np.reshape(g,(-1,10000)))
s=np.transpose(np.reshape(s,(-1,10000)))
gl=np.transpose(np.reshape(gl,(-1,10000)))
p=np.transpose(np.reshape(p,(-1,10000)))

l=np.concatenate([l1,l2,l3,l4,l5,l6,l7,l8,l9,l10],axis=1)

gender = np.empty(10000,dtype=object)
smile = np.empty(10000,dtype=object)
glass = np.empty(10000,dtype=object)
gender = np.empty(10000,dtype=object)
pose = np.empty(10000,dtype=object)

for n in range(0,10000):
    if g[n]==1:
        gender[n]=np.array([1,0])
    else:
        gender[n]=np.array([0,1])

    if s[n]==1:
        smile[n]=np.array([1,0])
    else:
        smile[n]=np.array([0,1])

    if gl[n]==1:
        glass[n]=np.array([1,0])
    else:
        glass[n]=np.array([0,1])
    
    if p[n]==1:
        pose[n]=np.array([1,0,0,0,0])
    elif p[n]==2:
        pose[n]=np.array([0,1,0,0,0])
    elif p[n]==3:
        pose[n]=np.array([0,0,1,0,0])
    elif p[n]==4:
        pose[n]=np.array([0,0,0,1,0])
    else:
        pose[n]=np.array([0,0,0,0,1])
    
    
        


def get_next_batch(num):


    
    return np.expand_dims(images[num], axis=0),np.expand_dims(l[num], axis=0),np.expand_dims(gender[num], axis=0),np.expand_dims(smile[num], axis=0),np.expand_dims(glass[num], axis=0),np.expand_dims(pose[num], axis=0)








print (get_next_batch(0),"abcdefg\n",np.expand_dims(images[0], axis=0).shape)

    
