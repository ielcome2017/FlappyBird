from agent import GameMemory
import keras
from net import NetV1, NetV2
import os

EPOCHS = 400
STEPS_PER_EPOCH = 10000
PATH = ["NETV1/", "NETV2/"]


def get_net(net_version):
    train_net, path = (NetV1(), "NETV1/") if net_version == 0 else (NetV2(), "NETV2/")
    call_function = [
        keras.callbacks.ModelCheckpoint(filepath=path + "weight.{epoch:02d}.h5")]

    if len(os.listdir(path)) == 0:
        return train_net, call_function, 0
    counts = [int(file.split(".")[1]) for file in os.listdir(path)]
    count = max(counts)
    filename = path + "weight.%02d.h5" % count
    train_net.load_weights(filename)

    return train_net, call_function, count


def train():
    net, call_function, count = get_net(net_version=0)
    func = keras.backend.function(net.input[0], net.layers[-2].output)
    agent = GameMemory(func, count, "train")
    net.fit(agent.next_data(), epochs=EPOCHS, initial_epoch=agent.count,
            steps_per_epoch=STEPS_PER_EPOCH, callbacks=call_function)


if __name__ == '__main__':
    train()
