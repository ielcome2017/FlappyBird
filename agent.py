import os
import time
import sys
import cv2
import pygame
import numpy as np

from collections import deque

from tensorflow import keras

from net import Net
from play import Game

GAMMA = 0.99
TIME_STEP = 4
STATE_SHAPE = [80, 80, TIME_STEP]
ACTION_DIM = 2
BATCH_SIZE = 32

MAX_LENGTH = 10000

OBSERVER = 1000
EXPLORE = 3000000

PLAY_STEP = 1000

def convert(x):
    x = cv2.cvtColor(cv2.resize(x, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x = cv2.threshold(x, 1, 255, cv2.THRESH_BINARY)
    # x = Image.fromarray(x).resize(STATE_SHAPE[:2]).convert("L")
    # x = np.array(x) < 1
    # x = x.astype("int")
    return np.array(x) / 255


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
        image = convert(image).reshape(*STATE_SHAPE[:2], 1)
        current_state = np.concatenate([image for _ in range(TIME_STEP)], axis=2)

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
                image = convert(image)  # 80*80

                yield image, [*current_action, reward, terminal]
                image = image.reshape(*STATE_SHAPE[:2], 1)  #80*80*1
                current_state = np.concatenate([image, current_state[:, :, :TIME_STEP-1]], axis=2)  # 80*80*4

        except pygame.error:
            print("\n-----game close----")

    def next_data(self):

        que = deque()
        # self.count = int(open("count.txt", "r").read()) if os.path.exists("count.txt") else 0
        # if self.count < OBSERVER:
        #     self.count = 0

        for image, art in self.generator():
            que.append([image, art])
            print(self.count, file=open("count.txt", "w"))
            self.count += 1

            num_sample = len(que)
            if num_sample < OBSERVER:
                sys.stdout.write("\r count: %d/%d"%(num_sample, OBSERVER))
                sys.stdout.flush()
                continue

            # 抽取数据训练
            num_sample = min(num_sample, MAX_LENGTH) - 5
            batch_ind = np.random.choice(np.arange(num_sample-5), BATCH_SIZE, replace=False)

            batch_current_state = np.stack([
                np.stack([que[idx + offset][0] for offset in range(TIME_STEP)], axis=-1)
            ], axis=0)
            batch_next_state = np.stack([
                np.stack([que[idx + offset + 1][0] for offset in range(TIME_STEP)], axis=-1)
            ])
            for num, idx in enumerate(batch_ind):
                batch_current_state[num] = np.stack([que[idx + offset][0] for offset in range(TIME_STEP)], axis=-1)
                batch_next_state[num] = np.stack([que[idx + offset + 1][0] for offset in range(TIME_STEP)], axis=-1)

            batch_art = np.stack([
                que[idx][1] for idx in batch_ind
            ], axis=0)

            batch_action = batch_art[:, 0:2]
            batch_reward = batch_art[:, -2]
            batch_terminal = batch_art[:, -1]
            out = self.func(batch_next_state).max(-1)
            # batch_reward = np.clip((batch_reward - 0.105) * 200, a_max=100, a_min=-100)

            batch_y = batch_reward + GAMMA * out * (1 - batch_terminal)
            yield [batch_current_state, batch_action], batch_y


if __name__ == '__main__':
    stime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    net = Net(False)

    reader = Dataset(net)
    data = reader.next_data()

    for i in data:
        # print(i)
        pass

    # net.fit(data, steps_per_epoch=10000, epochs=300)
