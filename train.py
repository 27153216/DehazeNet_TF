import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os

def preprocess_image(image, train=True):
    image = tf.image.decode_jpeg(image, channels=3)
    if not train: image = tf.image.resize(image, image.shape[:2])
    elif train: image = tf.image.resize(image, img_size[:2])
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_image(path, train=True):
    image = tf.io.read_file(path)
    return preprocess_image(image, train)

class MultiScaleMapping(tf.keras.Model):
  def __init__(self, kernel_size):
    super(MultiScaleMapping, self).__init__(name='')
    kernel_size_1, kernel_size_2, kernel_size_3 = kernel_size

    self.conv2a = tf.keras.layers.Conv2D(16, kernel_size_1, padding='same')
    self.conv2b = tf.keras.layers.Conv2D(16, kernel_size_2, padding='same')
    self.conv2c = tf.keras.layers.Conv2D(16, kernel_size_3, padding='same')

  def call(self, input_tensor, training=False):
    a = self.conv2a(input_tensor)
    b = self.conv2b(input_tensor)
    c = self.conv2c(input_tensor)

    x = tf.concat((a,b), axis=-1)
    x = tf.concat((x,c), axis=-1)
    return tf.nn.relu(x)



# img_path = './train'
# srcimg = load_and_preprocess_image(img_path+'/1_1_0.90179.png')
# srcimg = np.expand_dims(srcimg, axis=0) #因為要作成資料集，所以要加一維
# print(srcimg.shape)
# for file in os.listdir(img_path)[1:]:
#     temp = load_and_preprocess_image(img_path+ '/' +file)
#     temp = np.expand_dims(temp, axis=0)
#     srcimg = np.concatenate((srcimg, temp), axis=0)
# print(srcimg.shape)


# # gt = np.ones((1,300,450, 1))
# # gt *= 0.5
# # print(gt)
# img_path = './clear'
# gt = load_and_preprocess_image(img_path+'/1.png')
# gt = np.expand_dims(gt, axis=0) #因為要作成資料集，所以要加一維
# print(gt.shape)
# for file in os.listdir(img_path)[1:]:
#     temp = load_and_preprocess_image(img_path+ '/' +file)
#     temp = np.expand_dims(temp, axis=0)
#     gt = np.concatenate((gt, temp), axis=0)
# print(gt.shape)

img_size = [300, 300, 3]

# @tf.function(experimental_relax_shapes=True)
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(16, 5, activation='relu', padding="same", input_shape=img_size))
    model.add(layers.Reshape((-1, 16, 1)))
    model.add(layers.MaxPooling2D(pool_size=(1,4), strides=(1,4)))
    model.add(layers.Reshape((img_size[0], img_size[1], 4)))
    model.add(MultiScaleMapping([3,5,7]))
    model.add(layers.MaxPooling2D(pool_size=(7,7), strides=(1,1), padding="same"))
    model.add(layers.Conv2D(1, (6,6), activation='relu', padding='same'))
    def brelu(x):
        x = tf.clip_by_value(tf.keras.activations.relu(x), 0, 1)
        return x
    model.add(layers.Activation(brelu))
    
    def loss_fn(y_true, y_pred):
        I_pred = y_true*y_pred + 1*(1-y_pred)
        mse = tf.keras.losses.MeanSquaredError()
        return mse(srcimg, I_pred)
    
    # model.compile(loss=loss_fn, optimizer='sgd') #自訂loss
    model.compile(loss='mse', optimizer='sgd')
    
    return model

model = create_model()


# checkpoing先拿掉
# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                   save_weights_only=True,
#                                                   verbose=1,
#                                                   period=30000)


checkpoint_path = './checkpoints/'
checkn = 0
"""
path = '../../_IMG/RESIDE-standard/ITS/hazy/'
all_img = os.listdir(path)
np.random.shuffle(all_img)
for epoch in range(1):
    for file in all_img:
        img_path = path
        srcimg = load_and_preprocess_image(img_path + file)
        srcimg = np.expand_dims(srcimg, axis=0) #因為要作成資料集，所以要加一維
        img_path = '../../_IMG/RESIDE-standard/ITS/trans/'
        # gt = load_and_preprocess_image(img_path + file.split('_')[0] + '.' +file.split('.')[-1])
        gt = load_and_preprocess_image(img_path + file.split('_')[0] + '_' + file.split('_')[1] + '.' +file.split('.')[-1])
        gt = np.expand_dims(gt, axis=0) #因為要作成資料集，所以要加一維
        history = model.fit(srcimg, gt, epochs=1) #, callbacks=[cp_callback]檢查點先拿掉
    
        checkn += 1
        if checkn % 1000 == 0:  #每n個存一次
            model.save_weights(checkpoint_path + str(epoch) + '_' + str(checkn))
        print(checkn)

model.save_weights(checkpoint_path + 'lastest')
"""
# plt.imshow(model.predict(srcimg)[0,:,:,0], plt.cm.gray)
def testimg(path):
    global img_size, srcimg, model
    
    img_path = path
    srcimg = load_and_preprocess_image(img_path, train=False)
    
    img_size = srcimg.shape
    
    srcimg = np.expand_dims(srcimg, axis=0)
    
    tf.keras.backend.clear_session() #清除之前的model
    model = create_model()
    model.load_weights(checkpoint_path + 'lastest')
    
    t = model.predict(srcimg)[0]
    guide = cv.cvtColor(srcimg[0], cv.COLOR_RGB2GRAY)
    guide = guide.astype(np.float32)
    t = t.astype(np.float32)
    t = cv.ximgproc.guidedFilter(guide=guide, src=t, radius=60, eps=0.001, dDepth=-1)  #導向濾波
    t = np.clip(t, 0.1, 1)
    cv.imshow("t", t)
    
    t = np.expand_dims(t, axis=-1)
    res = cv.cvtColor((srcimg[0]-0.8)/t +0.8, cv.COLOR_RGB2BGR)
    cv.imshow("res", res)
    cv.waitKey(0)
    cv.destroyAllWindows()
    res *= 255
    res = res.astype(np.int32)
    cv.imwrite("./" + path.split('/')[-1], res)
testimg("./testimg/src 2.png")