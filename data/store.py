import numpy as np
import threading
import os
import sys
import cv2
import pygame
import h5py
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
        self.file_memory, self.file_length = "data/test.h5", "data/test_length.txt"
        self.memory = self.start()

    def memory_append(self, image, art, gap):
        idx = self.length % self.max_length
        self.memory["image"][idx] = image
        self.memory["art"][idx] = art
        self.memory["idx"][idx] = gap
        self.length += 1

    def start(self):
        if os.path.exists(self.file_length):
            with open(self.file_length, "r") as f:
                self.length += int(f.read())
            return h5py.File(self.file_memory, "r+")
        memory = h5py.File(self.file_memory, "w")
        memory.create_dataset("image", shape=[self.max_length, 80, 80], maxshape=[None, 80, 80], dtype="f")
        memory.create_dataset("art", shape=[self.max_length, 4], maxshape=[None, 4], dtype="f")
        memory.create_dataset("idx", shape=[self.max_length, 1], maxshape=[None, 1], dtype="i")
        return memory

    def end(self):
        print(self.length, file=open(self.file_length, "w"))
        self.memory.close()


class GameMemory(Memory):
    def __init__(self, func, count):
        self.count = count
        self.func = func

        self.explore, self.observer = 3000000, 100
        self.time_step = 4
        self.image_shape = (80, 80)
        self.pre_step_epoch = 10000
        super().__init__()

        assert self.observer < self.max_length

        self.train = True
        th = threading.Thread(target=self.generator)
        th.start()
        self.pause()

    def pause(self):
        for _ in self.next_data():
            if self.length > self.observer:
                return

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
            self.memory_append(image, [*current_action, reward, terminal], -1)
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
                    if epsilon > final_epsilon and self.length > self.explore else 0
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

                self.memory_append(image, [*current_action, reward, terminal], gap)
                slide = 0 if terminal else slide + 1
                state[:, :, :, 1:] = state[:, :, :, :3]
                state[:, :, :, 0] = image[np.newaxis, :, :]

        except pygame.error:
            self.train = False
            self.end()
            print("\n================> game close <=================")

    def next_data(self, batch_size=64):
        gamma = 0.99
        # 根据已经获取的长度偏移
        while True:
            if not self.train:
                break
            if self.length < self.max_length:
                num_sample = self.length
            else:
                num_sample = self.max_length
            if num_sample < self.observer:
                sys.stdout.write("\r num of sample is : %d/%d" % (num_sample, self.observer))
                sys.stdout.flush()
                continue

            # 抽取数据训练
            idx = self.memory["idx"][:]
            ind = idx[idx > 0]

            batch_ind = np.random.choice(ind, batch_size, replace=False) % self.max_length
            batch_ind = np.sort(batch_ind)

            image_ind = (batch_ind[:, np.newaxis] - np.arange(self.time_step + 1)) % self.max_length

            data_ind = np.unique(image_ind)
            batch = self.memory["image"][data_ind]

            vmap = {elem: i for i, elem in enumerate(data_ind)}
            vfunc = np.vectorize(lambda x: vmap.get(x))

            image_ind = vfunc(image_ind)
            current_state = np.transpose(batch[image_ind[:, :-1]], [0, 2, 3, 1])
            next_state = np.transpose(batch[image_ind[:, 1:]], [0, 2, 3, 1])

            art = self.memory["art"][batch_ind]
            action, reward, terminal = art[:, 0:2], art[:, -2], art[:, -1]
            out = self.func(next_state).max(-1)
            batch_y = reward + gamma * out * (1 - terminal)

            yield [current_state, action], batch_y


if __name__ == '__main__':
    from net import Net
    import keras
    n = Net()
    func = keras.backend.function(n.input[0], n.layers[-2].output)
    count = 0
    g = GameMemory(func, count)

    for e in g.next_data():
        print(e)
        break
