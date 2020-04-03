import keras
from keras.layers import  Dense, Conv2D, MaxPool2D, \
    AveragePooling2D, Lambda, Input, Flatten, LeakyReLU, Activation, Dot, Reshape, UpSampling2D, Deconvolution2D,Maximum
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def discriminator():
    state_shape, action_dim = [80, 80, 4], 2
    state = Input(state_shape)

    x = Conv2D(32, kernel_size=8, strides=4, padding="same")(state)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(64, kernel_size=4, strides=2, padding="same")(x)
    x = Activation('relu')(x)

    x = Conv2D(64, kernel_size=3, strides=1, padding="same")(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)

    # 输出价值
    out = Dense(action_dim, name="action_value")(x)
    model = keras.Model(state, out)
    return model


def generator():
    state_out = Input([102])
    x = Dense(512)(state_out)
    x = Activation("relu")(x)

    x = Dense(1600)(x)
    x = Activation("relu")(x)

    x = Reshape([5, 5, 64])(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding="same")(x)
    x = Activation("relu")(x)

    x = Deconvolution2D(32, kernel_size=4, strides=2, padding="same")(x)
    x = Activation("relu")(x)

    x = UpSampling2D()(x)
    x = Deconvolution2D(4, kernel_size=8, strides=4, padding="same")(x)
    out = Activation("sigmoid")(x)

    model = keras.Model(state_out, out)
    model.compile()
    model.summary()
    return model


def combine_model(d, g):
    model = keras.Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


class GAN:
    def __init__(self):
        pass

    def next_data(self):
        act = np.eye(2)
        rw = np.array([-1, 0.1, 1])

        while True:
            current_state = np.random.rand(32, 80, 80, 4)
            action = act[np.random.randint(0, 2, 32)]
            reward = rw[np.random.randint(0, 3, [32, 1])]
            terminal = np.random.randint(0, 2, [32, 1])
            next_state = np.random.rand(32, 80, 80, 4)
            yield current_state, next_state, action, reward, terminal

    def train_on_epoch(self):
        batch_size = 32

        disc = discriminator()
        gene = generator()

        action = Input([2])
        d_out = Maximum(-1)(disc.output)
        d = keras.Model(disc.input, d_out)
        d.compile(optimizer="adam", loss='mse')

        combined_tmp = combine_model(disc, gene)
        combined_out = Dot(-1)([action, combined_tmp.output])
        combined = keras.Model([gene.input, action], combined_out)

        combined.compile(optimizer="adam", loss="mse")
        combined.summary()

        for current_state, next_state, action, reward, terminal in self.next_data():
            d_current_value = d.predict(current_state)

            noisy = np.random.normal(0, 1, [32, 100])
            batch = np.concatenate([reward, terminal, d_current_value, noisy], axis=-1)

            g_value = combined.predict(batch)
            d_value = d.predict(next_state)

            d_loss = d.train_on_batch(next_state, g_value)

            g_loss = combined.train_on_batch([batch, action], d_value)



gan = GAN()
gan.train_on_epoch()
