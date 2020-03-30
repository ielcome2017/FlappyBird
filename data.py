import numpy as np
# import minpy.numpy as np
import threading
import sys
import cv2
import pygame
import random
from collections import deque

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
        self.D = deque(maxlen=self.max_length)

    def memory_append(self, state, next_state, action, reward, terminal):
        if len(self.D) > self.max_length:
            self.D.popleft()
        self.D.append([state, next_state, action, reward, terminal])


class GameMemory(Memory):
    def __init__(self, func, count):
        self.count = count
        self.func = func

        self.explore = 3000000
        self.observer = 10000
        self.time_step = 4
        self.image_shape = (80, 80)
        self.pre_step_epoch = 10000
        super().__init__()

        assert self.observer < self.max_length

        self.train = True
        th = threading.Thread(target=self.generator)
        th.start()

    def generator(self):
        # 参数设置
        epsilon, init_epsilon, final_epsilon = 0.1, 0.1, 0.001
        action_dim = 2
        # 初始化
        game = Game()
        current_action = np.array([1, 0])
        image, reward, terminal = game.frame_step(current_action)
        image = convert(image)

        epsilon -= max(self.count * self.pre_step_epoch - self.observer, 0) * \
                   (init_epsilon - final_epsilon) / self.explore

        # 获取当前状态
        state = np.stack([image for _ in range(4)], axis=-1)
        try:
            while True:
                # 获取动作
                if random.random() < epsilon:
                    action_ind = np.random.randint(0, action_dim)
                else:
                    action_ind = self.func(state[np.newaxis, :]).argmax(-1).astype("int")[0]

                epsilon -= (init_epsilon - final_epsilon) / self.explore \
                    if epsilon > final_epsilon and self.length > self.observer else 0
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        pygame.quit()
                        break
                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_UP:
                            action_ind = 1
                action = np.zeros(2)
                action[action_ind] = 1
                image, reward, terminal = game.frame_step(action)
                image = convert(image)  # 80*80

                next_state = np.zeros_like(state.squeeze())
                if terminal > 0:
                    next_state[:, :, 1:] = state[:, :, :3]
                    next_state[:, :, 0] = image[np.newaxis, :, :]
                else:
                    next_state = np.stack([image for _ in range(4)], axis=-1)
                self.memory_append(state, next_state, action, reward, terminal)
                state = next_state
        except pygame.error:
            self.train = False
            print("\n================> game close <=================")

    def next_data(self, batch_size=32):
        gamma = 0.99
        # 根据已经获取的长度偏移
        while True:
            if not self.train:
                break
            if len(self.D) < self.observer:
                sys.stdout.write("\r num of sample is : %d/%d" % (len(self.D), self.observer))
                sys.stdout.flush()
                continue

            # 抽取数据训练
            batch = random.sample(self.D, batch_size)
            state, next_state, action, reward, terminal = zip(*batch)
            state = np.stack(state, axis=0)
            next_state = np.stack(state, axis=0)
            action = np.stack(action, axis=0)
            reward = np.stack(reward, axis=0)
            terminal = np.stack(terminal, axis=0)

            out = self.func(next_state).max(-1)
            batch_y = reward + gamma * out * (1 - terminal)
            yield [state, action], batch_y
