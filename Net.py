#implement the Facial Landmark Detection of TCDCN, this code is combinge keras and tensorflow, have a reference on flyingzhao/tfTCDCN



import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dropout, Reshape, Flatten
from keras.layers import Dense, Conv2D, MaxPooling2D
from PrepareData import get_next_batch

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#set Keras session to tensorflow to initiate from placeholder

sess = tf.Session()
K.set_session(sess)
K.set_learning_phase(1)
image = tf.placeholder(tf.float32, shape=[None, 40, 40])
landmark = tf.placeholder(tf.float32, shape=[None, 10])
gender = tf.placeholder(tf.float32, shape=[None, 2])
smile = tf.placeholder(tf.float32, shape=[None, 2])
glasses = tf.placeholder(tf.float32, shape=[None, 2])
headpose = tf.placeholder(tf.float32, shape=[None, 5])


#add layers using keras layers

input_image = tf.reshape(image,[-1,40,40,1])
#model = Sequential()
h_conv1 = Conv2D(16, (5,5), activation='tanh')(input_image)
h_pool1 = MaxPooling2D((2,2))(h_conv1)
#model.add(Conv2D(16, (5,5), activation='tenH' input_shape=(40,40,1)))
#model.add(MaxPooling2D((2,2)))

h_conv2 = Conv2D(48, (3,3), activation='tanh' )(h_pool1)
h_pool2 = MaxPooling2D((2,2))(h_conv2)
#model.add(Conv2D(48, (3,3), activation='tenH' ))
#model.add(MaxPooling2D((2,2)))

h_conv3 = Conv2D(64, (3,3), activation='tanh' )(h_pool2)
h_pool3 = MaxPooling2D((2,2))(h_conv3)

#model.add(Conv2D(64, (3,3), activation='tenH' ))
#model.add(MaxPooling2D((2,2)))
h_conv4 = Conv2D(64, (2,2), activation='tanh' )(h_pool3)

#model.add(Conv2D(64, (2,2), activation='tenH' ))
h_pool4_flat = tf.reshape(h_conv4, [-1, 2 * 2 * 64])
h_fc1 = Dense(100, activation='tanh')(h_pool4_flat)
h_fc1_drop = Dropout(0.5)(h_fc1)
#model.add(Flatten())
#model.add(Dense(100, activation='tenH' ))
#model.add(Dropout(0.5))

#h_fc1_drop= model.layer[5].output
# gender
W_fc_gender = weight_variable([100, 2])
b_fc_gender = bias_variable([2])
y_gender = tf.matmul(h_fc1_drop, W_fc_gender) + b_fc_gender
# smile
W_fc_smile = weight_variable([100, 2])
b_fc_smile = bias_variable([2])
y_smile = tf.matmul(h_fc1_drop, W_fc_smile) + b_fc_smile
# glasses
W_fc_glasses = weight_variable([100, 2])
b_fc_glasses = bias_variable([2])
y_glasses =tf.matmul(h_fc1_drop, W_fc_glasses) + b_fc_glasses
# headpose
W_fc_headpose = weight_variable([100, 5])
b_fc_headpose = bias_variable([5])
y_headpose = tf.matmul(h_fc1_drop, W_fc_headpose) + b_fc_headpose


#def custom_objective(y_true, y_pred):

#return 1 / 2 * K.reduce_sum(K.square(landmark - y_landmark)) + \
#        K.reduce_sum(K.nn.softmax_cross_entropy_with_logits(logits=y_gender,labels=gender)) + \
#        K.reduce_sum(K.nn.softmax_cross_entropy_with_logits(logits=y_smile, labels=smile)) + \
#        K.reduce_sum(K.nn.softmax_cross_entropy_with_logits(logits=y_glasses,labels= glasses)) + \
#        K.reduce_sum(K.nn.softmax_cross_entropy_with_logits(logits=y_headpose,labels= headpose))+\
#        2*K.nn.l2_loss(W_fc_landmark)+\
#        2*K.nn.l2_loss(W_fc_glasses)+\
#        2*K.nn.l2_loss(W_fc_gender)+\
#        2*K.nn.l2_loss(W_fc_headpose)+\
#        2*K.nn.l2_loss(W_fc_smile)

#model.compile(optimizer ='AdamOptimizer', loss='custom_objective', metrics=['accuracy'])


# landmark
W_fc_landmark = weight_variable([100, 10])
b_fc_landmark = bias_variable([10])
y_landmark = tf.matmul(h_fc1_drop, W_fc_landmark) + b_fc_landmark

# gender
W_fc_gender = weight_variable([100, 2])
b_fc_gender = bias_variable([2])
y_gender = tf.matmul(h_fc1_drop, W_fc_gender) + b_fc_gender
# smile
W_fc_smile = weight_variable([100, 2])
b_fc_smile = bias_variable([2])
y_smile = tf.matmul(h_fc1_drop, W_fc_smile) + b_fc_smile
# glasses
W_fc_glasses = weight_variable([100, 2])
b_fc_glasses = bias_variable([2])
y_glasses = tf.matmul(h_fc1_drop, W_fc_glasses) + b_fc_glasses
# headpose
W_fc_headpose = weight_variable([100, 5])
b_fc_headpose = bias_variable([5])
y_headpose = tf.matmul(h_fc1_drop, W_fc_headpose) + b_fc_headpose


error = 1 / 2 * tf.reduce_sum(tf.square(landmark - y_landmark)) + \
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_gender,labels=gender)) + \
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_smile, labels=smile)) + \
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_glasses,labels= glasses)) + \
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_headpose,labels= headpose))+\
        2*tf.nn.l2_loss(W_fc_landmark)+\
        2*tf.nn.l2_loss(W_fc_glasses)+\
        2*tf.nn.l2_loss(W_fc_gender)+\
        2*tf.nn.l2_loss(W_fc_headpose)+\
        2*tf.nn.l2_loss(W_fc_smile)

#train
train_step = tf.train.AdamOptimizer(1e-3).minimize(error)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for x in range(1000):
        i, j, k, l, m, n = get_next_batch(x)
        print(x, sess.run(error,
                          feed_dict={image: i, landmark: j, gender: k, smile: l, glasses: m, headpose: n
                    }))
        sess.run(train_step,
                 feed_dict={image: i, landmark: j, gender: k, smile: l, glasses: m, headpose: n})
