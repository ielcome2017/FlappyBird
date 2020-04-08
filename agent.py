import numpy as np
import sys
import cv2
import random

from play import Game, GameOver

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
    def __init__(self, func, count, flag="explore"):
        self.count = count
        self.func = func
        self.flag = flag

        self.explore = 3000000
        self.observer = 10000

        self.image_shape = (80, 80)
        self.pre_step_epoch = 10000
        super().__init__()

    def show(self):
        for _ in self.next_data():
            yield _

    def next_data(self):
        # 参数设置
        epsilon = 0.001 if self.flag in ["explore", "display"] else 0.1
        init_epsilon, final_epsilon = 0.1, 0.001
        action_dim = 2
        # 初始化
        num = 40 if self.flag in ["explore", "train"] else 1
        game = Game(num)    # game为环境

        action = np.array([1, 0])
        image, reward, terminal = game.frame_step(action)

        image = convert(image)
        for _ in range(4):
            self.memory_append(image, [*action, reward, terminal])
        epsilon -= (init_epsilon - final_epsilon) / self.explore * self.count * self.pre_step_epoch
        epsilon = np.clip(epsilon, a_max=init_epsilon, a_min=final_epsilon)

        # 获取当前状态
        count = self.count * self.pre_step_epoch
        try:
            while True:
                # 获取动作
                if random.random() < epsilon:
                    action_ind = np.random.randint(0, action_dim)
                else:
                    idx = (self.next - np.arange(1, self.time_step+1)) % self.max_length
                    state = self.memory["image"][idx]

                    state = np.transpose(state[np.newaxis, :, :], [0, 2, 3, 1])
                    action_ind = self.func(state).argmax(-1).astype("int")[0]   # 智能体产生动作

                epsilon -= (init_epsilon - final_epsilon) / self.explore
                epsilon = np.clip(epsilon, a_max=init_epsilon, a_min=final_epsilon)
                count += 1

                action = game.get_event(action_ind)     # 游戏中事件触发

                image, reward, terminal = game.frame_step(action)   # 环境的激励
                image = convert(image)  # 80*80

                self.memory_append(image, [*action, reward, terminal])
                data = self.batch_data()
                if data is not None:
                    yield data

        except GameOver:
            print("\n{}> game close <{}".format("="*10, "="*10))

    def batch_data(self, batch_size=32):
        gamma = 0.99
        num_sample = self.next - self.head
        if num_sample < self.observer:
            sys.stdout.write("\r num of sample is : %d/%d" % (num_sample, self.observer))
            sys.stdout.flush()
            return None
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
        return [current_state, action], batch_y