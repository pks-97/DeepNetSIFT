
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, merge, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.models import load_model
from keras.callbacks import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# this part will prevent tensorflow to allocate all the avaliable GPU Memory
# backend
# import tensorflow as tf
# from keras import backend as k
# from tensorflow.python.keras import backend as k
from tensorflow.python.keras.backend import set_session

# Don't pre-allocate memory; allocate as-needed
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

# Create a session with the above options specified.
# k.backend.tensorflow_backend.set_session(tf.Session(config=config))
set_session(tf.compat.v1.Session(config=config))

# Hyperparameters

# number of classes
num_classes = 200
# input image dimensions
img_height, img_width = 32, 32
# The images are RGB.
img_channels = 3
















import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, SeparableConv2D, BatchNormalization, Activation, Input, concatenate, add, Conv2DTranspose, MaxPool2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from generators import *
from tensorflow.keras.utils import plot_model


log_dir = "/home/avnish/docker_mount/DenseRootSift/logs_siftv3TinyImageNet"


def space_to_depth_x2(x):
    import tensorflow as tf
    return tf.compat.v1.space_to_depth(x, block_size=2)


def DenseNet(input, filters, dilation_rate=(2,2)):
  """
  inception-like block with dilated separable convolutions
  """
  input = Input(shape=(None, None, img_channels))

# Block 1

  layer1 = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input)
  layer1 = BatchNormalization(name='norm_1')(layer1)
  layer1 = Activation("relu")(layer1)

  layer2 = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(layer1)
  layer2 = BatchNormalization(name='norm_2')(layer2)
  layer2 = Activation("relu")(layer2)

  layer3 = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(layer2)
  layer3 = BatchNormalization(name='norm_3')(layer3)
  layer3 = Activation("relu")(layer3)

  layer4 = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_4', use_bias=False)(layer3)
  layer4 = BatchNormalization(name='norm_4')(layer4)
  layer4 = Activation("relu")(layer4)

  layer5 = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(layer4)
  layer5 = BatchNormalization(name='norm_5')(layer5)
  layer5 = Activation("relu")(layer5)

  layer6 = MaxPooling2D(pool_size=(2, 2))(layer5)

  skip_connection_1 = layer6

# Block 2

  layer7 = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_7', use_bias=False)(layer6)
  layer7 = BatchNormalization(name='norm_7')(layer7)
  layer7 = Activation("relu")(layer7)

  layer8 = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(layer7)
  layer8 = BatchNormalization(name='norm_8')(layer8)
  layer8 = Activation("relu")(layer8)

  layer9 = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(layer8)
  layer9 = BatchNormalization(name='norm_9')(layer9)
  layer9 = Activation("relu")(layer9)

  layer10 = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_10', use_bias=False)(layer9)
  layer10 = BatchNormalization(name='norm_10')(layer10)
  layer10 = Activation("relu")(layer10)

  layer11 = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(layer10)
  layer11 = BatchNormalization(name='norm_11')(layer11)
  layer11 = Activation("relu")(layer11)

  layer12 = MaxPooling2D(pool_size=(2, 2))(layer11)

  skip_connection_1 = Lambda(space_to_depth_x2)(skip_connection_1)

  layer13 = concatenate([skip_connection_1, layer12])

  skip_connection_2 = layer13

# Block 3

  layer14 = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(layer13)
  layer14 = BatchNormalization(name='norm_14')(layer14)
  layer14 = Activation("relu")(layer14)

  layer15 = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_15', use_bias=False)(layer14)
  layer15 = BatchNormalization(name='norm_15')(layer15)
  layer15 = Activation("relu")(layer15)

  layer16 = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(layer15)
  layer16 = BatchNormalization(name='norm_16')(layer16)
  layer16 = Activation("relu")(layer16)

  layer17 = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_17', use_bias=False)(layer16)
  layer17 = BatchNormalization(name='norm_17')(layer17)
  layer17 = Activation("relu")(layer17)

  layer18 = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(layer17)
  layer18 = BatchNormalization(name='norm_18')(layer18)
  layer18 = Activation("relu")(layer18)

  layer19 = MaxPooling2D(pool_size=(2, 2))(layer18)

  skip_connection_2 = Lambda(space_to_depth_x2)(skip_connection_2)

  layer20 = concatenate([skip_connection_2, layer19])

  layer21 = Conv2D(num_classes, (1,1), name='conv_21', use_bias=False)(layer20)
  layer21 = BatchNormalization(name='norm_21')(layer21)

  layer22 = GlobalAveragePooling2D(data_format=None)(layer21)

  layer23 = Activation('softmax')(layer22)

  output = layer23
  iblock = concatenate([b1, b2, b3, pool])
  return iblock

def plot(model):
    plot_model(model, to_file='model_sift_tiny_imagenetv3.png',show_shapes=True, show_layer_names=True)



if __name__ == "__main__":
  # training strategy
  mirrored_strategy = tf.distribute.MirroredStrategy()

  with mirrored_strategy.scope():
    # train, test = get_generators(input_shape, dataset = "tiny_image_net", batch_size = 64)
    val_data = pd.read_csv('val/val_annotations.txt', sep='\t', header=None, names=['File', 'Class', 'X', 'Y', 'H', 'W'])
    val_data.drop(['X', 'Y', 'H', 'W'], axis=1, inplace=True)
    val_data.head(3)

    train_datagen = ImageDataGenerator(
    rescale= 1./255)

    valid_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory( r'./train/', target_size=(32, 32), color_mode='rgb', 
                                                    batch_size=256, class_mode='categorical', shuffle=True, seed=42)
    validation_generator = valid_datagen.flow_from_dataframe(val_data, directory='./val/images/', x_col='File', y_col='Class', target_size=(64, 64),
                                                    color_mode='rgb', class_mode='categorical', batch_size=256, shuffle=True, seed=42)


    model = Model(inputs=[input], outputs=[output])
    model.summary()
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.fit_generator(train_generator, epochs=15, steps_per_epoch=200, validation_steps=200, validation_data=validation_generator, verbose=1, callbacks=[lr_reducer])

    model.save('/gdrive/MyDrive/SIFT-Experiments/classification-experiments/Models/Exp_1_1.h5')
    plot_model(model)  
    # train, val = get_generators(input_shape, dataset = "tiny_image_net", batch_size = 128)
    # input = Input(shape=(None, None, 3))
    # sift_encodings = sift_encoder_v1(input)
    # model = Model(input, sift_encodings)
    # model = load_model('/home/avnish/docker_mount/DenseRootSift/siftv1-tinyimagenet.h5')

    # print(model.summary())
    # plot_model(model)

    # model.compile(optimizer = "adam", loss = ["mse", tf.keras.losses.MeanAbsoluteError()], metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
    # model.fit(train, validation_data = val, batch_size = 128, epochs = 100, callbacks = get_callbacks())
