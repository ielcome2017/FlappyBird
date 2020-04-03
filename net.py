import keras
from keras.layers import Dense, Conv2D, MaxPool2D, \
    Input, Flatten, Dot, Activation


def NetV1():
    state_shape, action_dim = [80, 80, 4], 2
    actions = Input([action_dim])
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

    out1 = Dense(action_dim)(x)

    out2 = Dot(-1)([actions, out1])
    model = keras.Model([state, actions], out2)
    optimizer = keras.optimizers.Adam(1e-6)
    # optimizer = keras.optimizers.SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
    loss = keras.losses.mse
    model.compile(optimizer=optimizer, loss=loss)
    model.summary()
    return model


def NetV2():
    state_shape, action_dim = [80, 80, 4], 2
    actions = Input([action_dim])
    state = Input(state_shape)

    x = Conv2D(32, kernel_size=8, strides=4, padding="same")(state)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(64, kernel_size=5, strides=2, padding="same")(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=2, padding="same")(x)

    x = Conv2D(64, kernel_size=3, strides=1, padding="same")(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=2, padding="same")(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)

    out1 = Dense(action_dim)(x)

    out2 = Dot(-1)([actions, out1])
    model = keras.Model([state, actions], out2)
    optimizer = keras.optimizers.Adam(1e-6)
    # optimizer = keras.optimizers.SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
    loss = keras.losses.mse
    model.compile(optimizer=optimizer, loss=loss)
    model.summary()
    return model


# import h5py
# f = h5py.File("bak/weight.292.h5", "r")
# n = []
# f.visit(n.append)
# for i in n:
#     if i.endswith("0"):
#         print(i, f[i].shape)

# net = NetV2()
# net.load_weights("bak/weight.292.h5")
# net = Net()
# net.save_weights("NETV1/weight.292.h5")