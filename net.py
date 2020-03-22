import keras
from keras.layers import Dense, Conv2D, MaxPool2D, Input, Multiply, Flatten, Dot, BatchNormalization, ReLU


import numpy as np


def conv(x, num_uits, kernel_size, stride, bn=True):
    x = Conv2D(num_uits, kernel_size=kernel_size, strides=stride, padding="same", activation="relu")(x)
    if bn:
        x = BatchNormalization()(x)
    return x


def Net(bn=True):
    action_dim = 2
    state_shape = [80, 80, 4]
    actions = Input([action_dim])
    state = Input(state_shape)

    x = conv(state, 32, 8, 4, bn)
    x = MaxPool2D(pool_size=2)(x)
    x = conv(x, 64, 5, 2, bn)
    x = conv(x, 64, 3, 2, bn)
    x = MaxPool2D(pool_size=2, padding="same")(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    out1 = Dense(action_dim, activation=None)(x)

    reward = Multiply()([actions, out1])

    out2 = Dense(1, kernel_initializer="ones", bias_initializer="zeros", trainable=False)(reward)

    model = keras.Model([state, actions], out2)
    # optimizer = keras.optimizers.SGD(lr=1e-6, momentum=0.9, nesterov=True)
    optimizer = keras.optimizers.Adam(learning_rate=1e-6,beta_1=0.9, beta_2=0.999)
    loss = keras.losses.mse
    model.compile(optimizer=optimizer, loss=loss)
    return model

def gen():
    import time
    for i in range(10):
        a = np.random.rand(3, 2)
        s = np.random.rand(3, 80, 80, 4)
        y = np.random.randint(0, 2, [3, 1])
        yield [s, a], y


if __name__ == '__main__':
    train_net = Net()
    a = np.random.rand(3, 2)
    s = np.random.rand(3, 80, 80, 4)
    y = np.random.randint(0, 2, [3, 1])


    f = keras.backend.function(train_net.input[0], train_net.layers[-3].output) #keras.Model(train_net.input[0], train_net.layers[-3].output)
    t = keras.Model(train_net.input[0], train_net.layers[-3].output).predict
    o = t(s)
    print(o)
    print(o.max(-1))

    train_net.fit_generator(gen(), epochs=1, steps_per_epoch=2)  # , callbacks=[ckpt])

    o = t(s)
    print(o)
    o = f(s)
    print(o)



