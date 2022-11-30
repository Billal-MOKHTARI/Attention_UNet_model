# Import packages
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, UpSampling2D, BatchNormalization, Add, Multiply, concatenate
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Input
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow.image as tfi
import numpy as np
from keras.metrics import MeanIoU
from keras.callbacks import ModelCheckpoint


# Define custommed objects
class EncoderBlock(Layer):

    def __init__(self, filters, rate, pooling=True, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)

        self.filters = filters
        self.rate = rate
        self.pooling = pooling

        self.c1 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.drop = Dropout(rate)
        self.c2 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.pool = MaxPool2D()

    def call(self, X):
        x = self.c1(X)
        x = self.drop(x)
        x = self.c2(x)
        if self.pooling:
            y = self.pool(x)
            return y, x
        else:
            return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters":self.filters,
            'rate':self.rate,
            'pooling':self.pooling
        }



class DecoderBlock(Layer):

    def __init__(self, filters, rate, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)

        self.filters = filters
        self.rate = rate

        self.up = UpSampling2D()
        self.net = EncoderBlock(filters, rate, pooling=False)

    def call(self, X):
        X, skip_X = X
        x = self.up(X)
        c_ = concatenate([x, skip_X])
        x = self.net(c_)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters":self.filters,
            'rate':self.rate,
        }


class AttentionGate(Layer):

    def __init__(self, filters, bn, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)

        self.filters = filters
        self.bn = bn

        self.normal = Conv2D(filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.down = Conv2D(filters, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')
        self.learn = Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')
        self.resample = UpSampling2D()
        self.BN = BatchNormalization()

    def call(self, X):
        X, skip_X = X

        x = self.normal(X)
        skip = self.down(skip_X)
        x = Add()([x, skip])
        x = self.learn(x)
        x = self.resample(x)
        f = Multiply()([x, skip_X])
        if self.bn:
            return self.BN(f)
        else:
            return f
        # return f

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters":self.filters,
            "bn":self.bn
        }
    
    
    
    def show_mask(image, mask, cmap=None, alpha=0.4):
    plt.imshow(image)
    plt.imshow(tf.squeeze(mask), cmap=cmap, alpha=alpha)
    plt.axis('off')

def show_image(image, title=None, cmap=None, alpha=1):
    plt.imshow(image, cmap=cmap, alpha=alpha)
    if title is not None:
        plt.title(title)
    plt.axis('off')


def load_image(image, SIZE = (256, 256)):
    return np.round(tfi.resize(img_to_array(load_img(image))/255., SIZE), 4)

def load_images(image_paths, SIZE, mask=False, trim=None):
    if trim is not None:
        image_paths = image_paths[:trim]
    
    if mask:
        images = np.zeros(shape=(len(image_paths), SIZE, SIZE, 1))
    else:
        images = np.zeros(shape=(len(image_paths), SIZE, SIZE, 3))
    
    for i, image in enumerate(image_paths):
        img = load_image(image, SIZE)
        if mask:
            images[i] = img[:,: ,: 1]
        else:
            images[i] = img
    
    return images







#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################

def attentionUNetModel(shape, classes=1):
  # Inputs
  input_layer = Input(shape=shape)

  # Encoder
  p1, c1 = EncoderBlock(32, 0.1, name="Encoder1")(input_layer)
  p2, c2 = EncoderBlock(64, 0.1, name="Encoder2")(p1)
  p3, c3 = EncoderBlock(128, 0.2, name="Encoder3")(p2)
  p4, c4 = EncoderBlock(256, 0.2, name="Encoder4")(p3)

  # Encoding
  encoding = EncoderBlock(512,0.3, pooling=False, name="Encoding")(p4)

  # Attention + Decoder

  a1 = AttentionGate(256, bn=True, name="Attention1")([encoding, c4])
  d1 = DecoderBlock(256,0.2, name="Decoder1")([encoding, a1])

  a2 = AttentionGate(128, bn=True, name="Attention2")([d1, c3])
  d2 = DecoderBlock(128, 0.2, name="Decoder2")([d1, a2])

  a3 = AttentionGate(64, bn=True, name="Attention3")([d2, c2])
  d3 = DecoderBlock(64, 0.1, name="Decoder3")([d2, a3])


  a4 = AttentionGate(32, bn=True, name="Attention4")([d3, c1])
  d4 = DecoderBlock(32, 0.1, name="Decoder4")([d3, a4])

  # Output
  if classes == 1: 
    act_f = 'sigmoid'
  else :
    act_f = 'softmax'

  output_layer = Conv2D(classes, kernel_size=1, activation=act_f, padding='same')(d4)

  # Model
  model = Model(
      inputs=[input_layer],
      outputs=[output_layer]
  )

  # Configure the model for training
  if classes == 1 :
    num_classes = 2
    loss = 'binary_crossentropy'
  else :
    num_classes = classes
    loss = 'categorical_crossentropy'

  model.compile(loss=loss,
              optimizer='adam',
              metrics=['accuracy', MeanIoU(num_classes=num_classes, name='IoU')])

  return model

"""
The model should have the format of '*.h5'
"""
def train_model(model, images, masks, model_save_path = 'AttentionUNet.h5', epochs=15, batch_size=8, validation_split=0.2):
  # Callbacks
  cb = [
      # EarlyStopping(patience=3, restore_best_weight=True), # With Segmentation I trust on eyes rather than on metrics
      ModelCheckpoint(model_save_path, save_best_only=True),
  ] 


  # Train the model
  # Config Training

  SPE = len(images)//batch_size

  # Training
  results = model.fit(
      images, masks,
      validation_split=validation_split,
      epochs=epochs, # 15 will be enough for a good Model for better model go with 20+
      steps_per_epoch=SPE,
      batch_size=batch_size,
      callbacks=cb
  )

  return results

def predict(image_path, model):

    image = load_image(image_path)

    pred_mask = model.predict(image[np.newaxis,...])[0]
    processed_pred_mask = (pred_mask>0.5).astype('float')

    plt.figure(figsize=(20,25))

    plt.subplot(1,3,1)
    plt.title("Original Image")
    plt.imshow(image)

    plt.subplot(1,3,2)
    plt.title("Predicted Mask")
    show_mask(image, pred_mask)

    plt.subplot(1,3,3)
    plt.title("Processed Mask")
    show_mask(image, processed_pred_mask)

    plt.tight_layout()
    plt.show()
