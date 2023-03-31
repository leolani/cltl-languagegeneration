import random

from cltl.reply_generation.api import ThoughtSelector


class RandomSelector(ThoughtSelector):
    def __init__(self, randomness=1.0, priority=()):
        self._randomness = randomness
        self._priority = {type: idx for idx, type in enumerate(priority)}

    def select(self, thoughts):
        if random.random() < self._randomness:
            return random.choice(list(thoughts))

        return next(filter(None, sorted(thoughts, key=self._get_order)))

    def _get_order(self, thought):
        if thought not in self._priority:
            return float('inf')

        return self._priority[thought]


if __name__ == '__main__':
    s = RandomSelector(randomness=0.5, priority=['a', 'b'])
    print(s.select(['b', 'a']))
