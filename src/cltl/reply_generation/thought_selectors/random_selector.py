import random

from cltl.reply_generation.api import ThoughtSelector


class RandomSelector(ThoughtSelector):

    def select(self, thoughts):
        return random.choice(list(thoughts))
