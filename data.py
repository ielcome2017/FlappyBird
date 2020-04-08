import numpy as np
# import minpy.numpy as np
import threading
import sys
import cv2
import pygame
import random

from play import Game

IMAGE_SHAPE = (80, 80)


def convert(img):
    img = cv2.cvtColor(cv2.resize(img, IMAGE_SHAPE), cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    return np.array(img)


class Memory:
    def __init__(self):
        self.time_step = 4
        self.max_length = 50000
        self.head, self.next = self.time_step, 0
        self.memory = np.empty(self.max_length,
                               dtype=[("image", np.float, IMAGE_SHAPE), ("art", np.float, [4])])

    def memory_append(self, image, art):
        self.memory["image"][self.next % self.max_length] = image
        self.memory["art"][self.next % self.max_length] = art
        self.next = self.next + 1
        self.head += 1 if self.next > self.max_length else 0


class GameMemory(Memory):
    def __init__(self, func, count, flag="finetune"):
        self.count = count
        self.func = func
        self.flag = flag

        self.explore = 3000000
        self.observer = 10000

        self.image_shape = (80, 80)
        self.pre_step_epoch = 10000
        super().__init__()
        self.train = True
        th = threading.Thread(target=self.generator)
        th.start()

    def display(self):
        for _ in self.next_data():
            pass

    def generator(self):
        # 参数设置
        epsilon = 0.001 if self.flag in ["finetune", "test"] else 0.1
        init_epsilon, final_epsilon = 0.1, 0.001
        action_dim = 2
        # 初始化
        game = Game()
        action = np.array([1, 0])
        image, reward, terminal = game.frame_step(action)
        image = convert(image)
        for _ in range(4):
            self.memory_append(image, [*action, reward, terminal])
        epsilon -= (init_epsilon - final_epsilon) * self.count * self.pre_step_epoch / self.explore
        # 获取当前状态
        count = self.count * self.pre_step_epoch

        try:
            while True:
                # 获取动作
                epsilon = np.clip(epsilon, a_max=init_epsilon, a_min=final_epsilon)
                if random.random() < epsilon:
                    action_ind = np.random.randint(0, action_dim)
                else:
                    idx = (self.next - np.arange(1, self.time_step+1)) % self.max_length
                    state = self.memory["image"][idx]

                    state = np.transpose(state[np.newaxis, :, :], [0, 2, 3, 1])
                    action_ind = self.func(state).argmax(-1).astype("int")[0]

                epsilon -= (init_epsilon - final_epsilon) / self.explore
                count += 1

                action = game.get_event(action_ind)     # 游戏中事件触发
                image, reward, terminal = game.frame_step(action)
                image = convert(image)  # 80*8140

                self.memory_append(image, [*action, reward, terminal])

        except pygame.error:
            self.train = False
            print("\n================> game close <=================")

    def next_data(self, batch_size=32):
        gamma = 0.99
        while True:
            if not self.train:
                break
            num_sample = self.next - self.head
            if num_sample < self.observer:
                sys.stdout.write("\r num of sample is : %d/%d" % (num_sample, self.observer))
                sys.stdout.flush()
                continue
            batch_ind = np.random.choice(np.arange(self.head, self.next), [batch_size]) % self.max_length
            # 抽取数据训练
            image_ind = (batch_ind[:, np.newaxis] - np.arange(self.time_step)) % self.max_length
            # 当前步为预测动作产生的状态，要用上一个状态的reward与当前预测的y比较
            # 如当前批次索引为[1, 3, 5, 7...] 取第一个索引，那么当前状态为[0, 1, 2, 3], next_state为[1, 2, 3, 4]
            current_state = np.transpose(self.memory["image"][(image_ind - 1) % self.max_length], [0, 2, 3, 1])
            next_state = np.transpose(self.memory["image"][image_ind], [0, 2, 3, 1])
            art = self.memory["art"][batch_ind]
            action, reward, terminal = art[:, 0:2], art[:, -2], art[:, -1]
            out = self.func(next_state).max(-1)
            batch_y = reward + gamma * out * (1 - terminal)
            yield [current_state, action], batch_y
            # yield current_state, next_state, action, reward, terminal
