import numpy as np
import sys
import cv2
import pygame
import random

from game.play import Game

IMAGE_SHAPE = (80, 80)


def convert(img):
    img = cv2.cvtColor(cv2.resize(img, IMAGE_SHAPE), cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    return np.array(img) / 255.


class Memory:
    def __init__(self):
        self.length = 0
        self.max_length = 50000
        self.memory = np.empty(self.max_length,
                               dtype=[("image", np.float, IMAGE_SHAPE), ("art", np.float, [4]), ("idx", np.int, [1])])
        self.file_memory, self.file_length = "data/memory.h5", "data/length.txt"

    def memory_append(self, image, art, gap):
        idx = self.length % self.max_length
        self.memory["image"][idx] = image
        self.memory["art"][idx] = art
        self.memory["idx"][idx] = gap
        self.length += 1


class GameMemory(Memory):
    def __init__(self, func, count):
        self.count = count
        self.func = func

        self.explore = 3000000
        self.observer = 100 if count > 0 else 800
        self.time_step = 4
        self.image_shape = (80, 80)
        self.pre_step_epoch = 10000
        super().__init__()
        assert self.observer < self.max_length

    def generator(self):
        # 参数设置
        epsilon, init_epsilon, final_epsilon = 0.1, 0.1, 0.001
        action_dim = 2
        # 初始化
        game = Game()
        current_action = np.array([1, 0])
        image, reward, terminal = game.frame_step(current_action)
        image = convert(image)

        for i in range(self.time_step):
            yield image, [*current_action, reward, terminal], -1
        epsilon -= max(self.length % self.pre_step_epoch + self.count * self.pre_step_epoch - self.observer, 0) * \
                   (init_epsilon - final_epsilon) / self.explore

        # 获取当前状态
        state = np.stack([image for _ in range(4)], axis=-1)
        state = state[np.newaxis, :]

        slide = self.time_step
        try:
            while True:
                gap = self.length if slide >= self.time_step else -1
                # 获取动作
                if random.random() < epsilon:
                    action_ind = np.random.randint(0, action_dim)
                else:
                    value = self.func(state)
                    action_ind = value.argmax(-1).astype("int")[0]

                epsilon -= (init_epsilon - final_epsilon) / self.explore \
                    if epsilon > final_epsilon and self.length > self.observer else 0
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        pygame.quit()
                        break
                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_UP:
                            action_ind = 1
                current_action = np.zeros(2)
                current_action[action_ind] = 1
                image, reward, terminal = game.frame_step(current_action)
                image = convert(image)  # 80*80

                yield image, [*current_action, reward, terminal], gap
                slide = 0 if terminal else slide + 1
                state[:, :, :, 1:] = state[:, :, :, :3]
                state[:, :, :, 0] = image[np.newaxis, :, :]

        except pygame.error:
            print("\n================> game close <=================")

    def next_data(self, batch_size=32):
        gamma = 0.99
        # 根据已经获取的长度偏移

        for image, art, idx in self.generator():

            self.memory_append(image, art, idx)

            if self.length < self.max_length:
                num_sample = self.length
            else:
                num_sample = self.max_length
            if num_sample < self.observer:
                sys.stdout.write("\r num of sample is : %d/%d" % (num_sample, self.observer))
                sys.stdout.flush()
                continue

            # 抽取数据训练
            ind = self.memory["idx"][self.memory["idx"] > 0]
            batch_ind = np.random.choice(ind, batch_size) % self.max_length
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
