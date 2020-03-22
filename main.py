from data import Dataset
from agent import Net
import matplotlib.pyplot as plt
import keras

EPOCHS = 300
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




def train():

    net = Net(False)
    reader = Dataset(net)
    net.summary()

    epoch = EPOCHS - reader.count // STEPS_PER_EPOCH
    data = reader.next_data()

    count = reader.count // STEPS_PER_EPOCH
    model_name = "model/weight_%02d"%count + "_{epoch:02d}_.h5"
    ckpt = keras.callbacks.ModelCheckpoint(
        filepath=model_name,
        monitor="loss",
        mode="min",
        verbose=1,
    )

    # draw = Draw()
    try:
        net.fit(data, epochs=epoch, steps_per_epoch=STEPS_PER_EPOCH, callbacks=[ckpt])
    except StopIteration:
        print("\ntrain stop...")


if __name__ == '__main__':
    train()