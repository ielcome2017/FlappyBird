import tensorflow.compat.v1 as tf
import h5py
import numpy as np
from net import NetV1
import os


def checkpoint_h5(checkpoint_filename):
    count = int(checkpoint_filename.split("-")[-1]) // 10000
    path = os.path.dirname(checkpoint_filename)
    h5_filename = "{}/weight.{}.h5".format(path, count)

    reader = tf.train.NewCheckpointReader(checkpoint_filename)
    tensor_variable_names = []
    for key in sorted(reader.get_variable_to_shape_map()):
        if "Adam" not in key:
            tensor_variable_names.append(key)

    net = NetV1()
    net.save_weights(h5_filename)

    writer = h5py.File(h5_filename, "r+")
    vars, var_names = [], []
    writer.visit(vars.append)
    for x in vars:
        if "kernel" in x or "bias" in x:
            var_names.append(x)
    var_names = np.array(var_names).reshape([-1, 2])[:, [1, 0]].reval()
    for i, x in enumerate(tensor_variable_names):
        t = reader.get_tensor(x)
        writer[var_names[i]][:] = t
    writer.close()


if __name__ == '__main__':
    checkpoint_h5("model/bird-dqn-2920000")