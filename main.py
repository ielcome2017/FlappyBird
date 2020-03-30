
from data import GameMemory
# from agent import GameMemory
from net import Net
import matplotlib.pyplot as plt
import keras
import os
import numpy as np

EPOCHS = 400
STEPS_PER_EPOCH = 10000


class Draw(keras.callbacks.Callback):

    def __init__(self, *args, **kwargs):
        super(Draw, self).__init__(*args, **kwargs)
        self.x, self.y = [], []
        fig = plt.figure()
        self.ax = fig.add_subplot(1, 1, 1)
        self.line, = plt.plot([], [])

    def on_batch_end(self, batch, logs=None):
        loss = logs.get("loss")
        self.x.append(batch)
        self.y.append(loss)
        self.ax.set_xlim(self.x[0], self.x[-1])
        self.ax.set_ylim(0, max(self.y))
        if len(self.y) > 100:
            self.y.pop(0)
            self.x.pop(0)

        self.line.set_data(self.x, self.y)
        plt.pause(0.001)


def get_count(train_net):
    if len(os.listdir("model")) == 0:
        return train_net, 0
    counts = [int(file.split(".")[1]) for file in os.listdir("model")]
    count = max(counts)
    filename = "model/weight.%02d.h5" % count

    train_net.load_weights(filename)
    return train_net, count


def train():
    net = Net()
    # predict_net = Net(False, trainable=False)
    # get_count(predict_net)
    # func = keras.Model(predict_net.input[0], predict_net.layers[-3].output).predict
    # func = keras.backend.function(predict_net.input[0], predict_net.layers[-3].output)
    func = keras.backend.function(net.input[0], net.layers[-2].output)

    net, count = get_count(net)
    reader = GameMemory(func, count)
    data = reader.next_data()

    ckpt = [
        keras.callbacks.ModelCheckpoint(
            filepath="model/weight.{epoch:02d}.h5",
            monitor="loss",
            mode="min",
            verbose=1),
        # PredWeight(predict_net)
    ]

    # draw = Draw()
    # try:
    net.fit(data, epochs=EPOCHS, initial_epoch=reader.count, steps_per_epoch=STEPS_PER_EPOCH, callbacks=ckpt)
    # except Exception as e:
    #     print("\n================> train  stop <=================")


if __name__ == '__main__':
    train()