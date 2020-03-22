import numpy as np


class Control:
    def __init__(self, player, scoreboard):
        self.player = player
        self.scoreboard = scoreboard
        self.max_observe = 4
        self.observes = []

    def attach(self, observe):
        if len(self.observes) >= self.max_observe:
            self.observes.pop(1)
        self.observes.append(observe)

    def notify(self, actions):
        self.player.update(actions)

        state = np.array([False, False, False])
        for observe in self.observes:
            state = np.logical_or(state, observe.update(self.player))
        self.scoreboard.update(int(state[-1]))
        # print(state)
        return state
