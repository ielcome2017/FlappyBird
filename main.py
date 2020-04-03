from agent import GameMemory
# from data import GameMemory
import keras
from net import NetV1, NetV2
import os

EPOCHS = 400
STEPS_PER_EPOCH = 10000
PATH = ["NETV1/", "NETV2/"]


def get_net(net_version):
    if net_version == 0:
        train_net, path = NetV1(), "NETV1/"
    else:
        train_net, path = NetV2(), "NETV2/"
    print(path)
    if len(os.listdir(path)) == 0:
        return 0
    counts = [int(file.split(".")[1]) for file in os.listdir(path)]
    count = max(counts)

    filename = path + "weight.%02d.h5" % count
    train_net.load_weights(filename)
    call_function = [
        keras.callbacks.ModelCheckpoint(filepath=path + "weight.{epoch:02d}.h5")]
    return train_net, call_function, count


def train():
    net, call_function, count = get_net(net_version=1)
    func = keras.backend.function(net.input[0], net.layers[-2].output)
    agent = GameMemory(func, count, "train")

    net.fit(agent.next_data(), epochs=EPOCHS, initial_epoch=agent.count,
                steps_per_epoch=STEPS_PER_EPOCH, callbacks=call_function)


if __name__ == '__main__':
    train()