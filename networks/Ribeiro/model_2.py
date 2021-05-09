from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense)
from tensorflow.keras.models import Model
import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf


class ResidualUnit(object):
    def __init__(self, n_filters_out, kernel_initializer='he_normal',
                 dropout_keep_prob=0.8, kernel_size=17, preactivation=True,
                 postactivation_bn=False, activation_function='relu'):
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = 1 - dropout_keep_prob
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function

    def _skip_connection(self, y, downsample, n_filters_in):
        """Implement skip connection."""
        # Deal with downsampling
        if downsample > 1:
            y = MaxPooling1D(downsample, strides=downsample, padding='same')(y)
        elif downsample == 1:
            y = y
        else:
            raise ValueError("Number of samples should always decrease.")
        # Deal with n_filters dimension increase
        if n_filters_in != self.n_filters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.
            y = Conv1D(self.n_filters_out, 1, padding='same',
                       use_bias=False, kernel_initializer=self.kernel_initializer)(y)
        return y

    def _batch_norm_plus_activation(self, x):
        if self.postactivation_bn:
            # x = Activation(self.activation_function)(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
            # x = BatchNormalization(center=False, scale=False)(x)
            x = tfa.layers.InstanceNormalization(axis=2,
                                                center=False,
                                                scale=False,
                                                beta_initializer=self.kernel_initializer,
                                                gamma_initializer=self.kernel_initializer)(x)
        else:
            # x = BatchNormalization()(x)
            x = tfa.layers.InstanceNormalization(axis=2,
                                                center=False,
                                                scale=False,
                                                beta_initializer=self.kernel_initializer,
                                                gamma_initializer=self.kernel_initializer)(x)
            # x = Activation(self.activation_function)(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        return x

    def __call__(self, inputs):
        """Residual unit."""
        x, y = inputs
        n_samples_in = y.shape[1]
        downsample = 2
        n_filters_in = y.shape[2]
        y = self._skip_connection(y, downsample, n_filters_in)
        # 1st layer
        x = Conv1D(self.n_filters_out, self.kernel_size, padding='same',
                   use_bias=False, kernel_initializer=self.kernel_initializer)(x)
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)

        # 2nd layer
        x = Conv1D(self.n_filters_out, self.kernel_size, strides=downsample,
                   padding='same', use_bias=False,
                   kernel_initializer=self.kernel_initializer)(x)
        if self.preactivation:
            x = Add()([x, y])  # Sum skip connection and main connection
            y = x
            x = self._batch_norm_plus_activation(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        else:
            # x = BatchNormalization()(x)
            x = tfa.layers.InstanceNormalization(axis=2,
                                                center=False,
                                                scale=False,
                                                beta_initializer=self.kernel_initializer,
                                                gamma_initializer=self.kernel_initializer)(x)
            x = Add()([x, y])  # Sum skip connection and main connection
            # x = Activation(self.activation_function)(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
            y = x
        return [x, y]


def get_model(n_classes, input_shape ,last_layer='softmax'):
    kernel_size = 16
    kernel_initializer = 'he_normal'
    signal = Input(shape=input_shape, dtype=np.float32, name='signal')
    x = signal
    x = Conv1D(64, kernel_size, padding='same', use_bias=False,
            kernel_initializer=kernel_initializer)(x)

    # x = BatchNormalization()(x)
    x = tfa.layers.InstanceNormalization(axis=2,
                                        center=False,
                                        scale=False,
                                        beta_initializer=kernel_initializer,
                                        gamma_initializer=kernel_initializer)(x)
    # x = Activation('relu')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    # 1st residual block 
    x, y = ResidualUnit(128, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, x])

    x, y = ResidualUnit(256, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])

    x, y = ResidualUnit(384, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])

    x, y = ResidualUnit(512, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])

    x, y = ResidualUnit(768, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])

    x, y = ResidualUnit(1024, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])

    x, y = ResidualUnit(1024+512, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])

    x, y = ResidualUnit(1024+768, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])

    x, y = ResidualUnit(2048, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])

    x, y = ResidualUnit(2048+512, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])

    x = Flatten()(x)
    x = Dense(2560, activation=None, kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = Dense(512, activation=None, kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = Dense(128, activation=None, kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    output = Dense(n_classes, activation=last_layer, kernel_initializer=kernel_initializer)(x)
    model = Model(signal, output)
    return model