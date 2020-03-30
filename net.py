import keras
from keras.layers import Dense, Conv2D, MaxPool2D, \
    Input, Flatten, Dot, Activation


def Net():
    state_shape, action_dim = [80, 80, 4], 2
    actions = Input([action_dim])
    state = Input(state_shape)

    x = Conv2D(32, kernel_size=8, strides=4, padding="same")(state)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(64, kernel_size=5, strides=2, padding="same")(x)
    x = Activation('relu')(x)

    x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
    x = Activation('relu')(x)

    x = MaxPool2D(pool_size=2, padding="same")(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)

    out1 = Dense(action_dim)(x)

    out2 = Dot(-1)([actions, out1])
    model = keras.Model([state, actions], out2)
    optimizer = keras.optimizers.Adam(1e-4)
    # optimizer = keras.optimizers.SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
    loss = keras.losses.mse
    model.compile(optimizer=optimizer, loss=loss)
    model.summary()
    return model


# model = Net()
# model.load_weights("bak/model.h5", skip_mismatch=True)

# print(model.get_weights()[1])
# model.load_weights("bak/weight.01.h5", skip_mismatch=True)
# print(model.get_weights()[1])
# model.save_weights("model/weight.01.h5")