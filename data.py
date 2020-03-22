import os
import time
from collections import deque

from PIL import Image
from tensorflow import keras

from net import *
from play import *

GAMMA = 0.99
TIME_STEP = 4
STATE_SHAPE = [80, 80, TIME_STEP]
ACTION_DIM = 2
BATCH_SIZE = 32

MAX_LENGTH = 5000

OBSERVER = 128
EXPLORE = 3000000

PLAY_STEP = 1000

def convert(x):
    # x = cv2.cvtColor(cv2.resize(x, (80, 80)), cv2.COLOR_BGR2GRAY)
    # ret, x = cv2.threshold(x, 1, 255, cv2.THRESH_BINARY)
    x = Image.fromarray(x).resize(STATE_SHAPE[:2]).convert("L")
    x = np.array(x) < 1
    x = x.astype("int")
    return x.reshape([*STATE_SHAPE[:2], 1])


class Dataset():
    def __init__(self, train_net):

        file = os.listdir("model")
        if len(file) > 0:
            file = file[-1]
            filename = "model/" + file
            print(filename)
            train_net.load_weights(filename)
            count = (int(file.split("_")[1]) + int(file.split("_")[2]))* 10000
        else:
            count = 0
        self.count = count
        self.model = keras.Model(train_net.input[0], train_net.layers[-3].output)
        self.model.summary()

        self.func = keras.backend.function(train_net.input[0], train_net.layers[-3].output)

        # self.game = Game()

    def generator(self):
        init_epsilon, final_epsilon = 0.1, 0.001
        epsilon = init_epsilon

        game = Game()

        current_action = np.array([0, 1])
        image, reward, terminal = game.frame_step(current_action)
        image = convert(image)
        current_state = np.concatenate([image for i in range(TIME_STEP)], axis=2)

        position = 0
        epsilon -= max(self.count - OBSERVER, 0) * (init_epsilon - final_epsilon) / EXPLORE

        try:
            while True:
                # 随机动作
                if np.random.random() < epsilon:
                    action_ind = np.random.randint(0, ACTION_DIM)
                else:
                    state = np.expand_dims(current_state, axis=0)
                    action_ind = self.func(state).argmax(-1).astype("int")

                # 越到后期，随机越少
                epsilon -= (init_epsilon - final_epsilon) / EXPLORE if epsilon > final_epsilon and self.count > OBSERVER else 0

                # 开启辅助训练窗口
                # action_ind = 0 if position < PLAY_STEP else action_ind
                position += 1
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        pygame.quit()
                        break
                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_UP:
                            action_ind = 1

                # 获取动作
                current_action = np.zeros(2)
                current_action[action_ind] = 1
                image, reward, terminal = game.frame_step(current_action)
                image = convert(image)
                next_state = np.concatenate([image, current_state[:, :, :TIME_STEP-1]], axis=2)

                yield current_state, next_state,[*current_action, reward, terminal]
                current_state = next_state
                self.count += 1

        except pygame.error:
            print("\n-----game close----")

    def next_data(self):
        q = deque(maxlen=MAX_LENGTH)
        for data in self.generator():

            q.append(data)
            if not self.count > OBSERVER or len(q) < BATCH_SIZE:
                continue
            # 抽取数据训练
            num_sample = len(q) // 2 if len(q) > 64*2 else len(q)
            batch_ind = np.random.choice(np.arange(num_sample), BATCH_SIZE, replace=False)

            batch_current_state = np.stack([q[i][0] for i in batch_ind], axis=0)
            batch_next_state = np.stack([q[i][1] for i in batch_ind],  axis=0)
            batch_art = np.stack([q[i][2] for i in batch_ind], axis=0)

            batch_action = batch_art[:, 0:2]
            batch_reward = batch_art[:, -2]
            batch_terminal = batch_art[:, -1]

            out = self.func(batch_next_state).max(-1)
            batch_reward = np.clip((batch_reward - 0.105) * 200, a_max=100, a_min=-100)
            batch_y = batch_reward + GAMMA * out * (1 - batch_terminal)
            yield [batch_current_state, batch_action], batch_y


if __name__ == '__main__':
    stime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    net = Net()

    data = Dataset(net).next_data()

    net.fit(data, steps_per_epoch=10000, epochs=300)
