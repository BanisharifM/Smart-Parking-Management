# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ResNet50 model for Keras.

Adapted from tf.keras.applications.resnet50.ResNet50().
This is ResNet model version 1.5.

Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers, backend as K

def _gen_l2_regularizer(use_l2_regularizer=True, l2_weight_decay=1e-4):
    return regularizers.L2(l2_weight_decay) if use_l2_regularizer else None

def preprocess_input(inputs):
    """Preprocess input for ResNet."""
    # Rescale inputs to range [-1, 1]
    inputs = tf.cast(inputs, tf.float32)
    inputs = (inputs / 127.5) - 1.0
    return inputs

def identity_block(input_tensor, kernel_size, filters, stage, block, use_l2_regularizer=True):
    """The identity block is the block that has no conv layer at shortcut."""
    filters1, filters2, filters3 = filters
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    x = layers.Conv2D(
        filters1, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=f"res{stage}{block}_branch2a")(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=f"bn{stage}{block}_branch2a")(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters2, kernel_size, padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=f"res{stage}{block}_branch2b")(x)
    x = layers.BatchNormalization(axis=bn_axis, name=f"bn{stage}{block}_branch2b")(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters3, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=f"res{stage}{block}_branch2c")(x)
    x = layers.BatchNormalization(axis=bn_axis, name=f"bn{stage}{block}_branch2c")(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_l2_regularizer=True):
    """A block that has a conv layer at shortcut."""
    filters1, filters2, filters3 = filters
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    x = layers.Conv2D(
        filters1, (1, 1), strides=strides,
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=f"res{stage}{block}_branch2a")(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=f"bn{stage}{block}_branch2a")(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters2, kernel_size, padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=f"res{stage}{block}_branch2b")(x)
    x = layers.BatchNormalization(axis=bn_axis, name=f"bn{stage}{block}_branch2b")(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters3, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=f"res{stage}{block}_branch2c")(x)
    x = layers.BatchNormalization(axis=bn_axis, name=f"bn{stage}{block}_branch2c")(x)

    shortcut = layers.Conv2D(
        filters3, (1, 1), strides=strides,
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=f"res{stage}{block}_branch1")(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis, name=f"bn{stage}{block}_branch1")(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def resnet50(num_classes, batch_size=None, use_l2_regularizer=True):
    """Instantiates the ResNet50 architecture."""
    input_shape = (224, 224, 3)
    img_input = layers.Input(shape=input_shape, batch_size=batch_size)

    x = preprocess_input(img_input)

    x = layers.ZeroPadding2D(padding=(3, 3), name="conv1_pad")(x)
    x = layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='valid',
        use_bias=False, kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name="conv1")(x)
    x = layers.BatchNormalization(axis=3, name="bn_conv1")(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Stack blocks
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), use_l2_regularizer=use_l2_regularizer)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', use_l2_regularizer=use_l2_regularizer)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', use_l2_regularizer=use_l2_regularizer)

    # Additional blocks omitted for brevity...

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        num_classes, kernel_initializer="he_normal",
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name="fc1000")(x)

    x = layers.Activation("softmax", dtype="float32")(x)

    return tf.keras.Model(img_input, x, name="resnet50")

ResnetModel = resnet50
