""" Filename:     rl_replier.py
    Author(s):    Thomas Bellucci
    Description:  Modified LenkaRepliers for use in the Chatbot. The replier
                  takes out the Thought selection step and only selects among
                  thoughts using reinforcement learning (RLReplier).
    Date created: Nov. 11th, 2021
"""

from cltl.combot.backend.utils.casefolding import (casefold_capsule)
from cltl.reply_generation.lenka_replier import LenkaReplier
from cltl.reply_generation.reinforcement_learning.rl import UCB
from cltl.reply_generation.utils.replier_utils import thoughts_from_brain


class RLReplier(LenkaReplier):
    def __init__(self, brain, savefile=None):
        """Creates a reinforcement learning-based replier to respond to questions
        and statements by the user. Statements are replied to by phrasing a
        thought; Selection of the thoughts are learnt by the UCB algorithm.

        params
        object brain: the brain of Leolani
        str savefile: file with stored utility values in JSON format

        returns: None
        """
        super(RLReplier, self).__init__()
        self.__thought_selector = UCB()
        self._log.debug(f"UCB RL Selector ready")

        self.__brain = brain
        self.__thought_selector.load(savefile)
        self.__last_thought = None
        self.__brain_states = []

    def reward_thought(self):
        """Rewards the last thought phrased by the replier by updating its
        utility estimate with the relative improvement of the brain as
        a result of the user response (i.e. a reward).

        returns: None
        """
        brain_state = 0

        if self.__brain:
            # Re-evaluate state of brain
            claims = float(self.__brain.count_statements())
            entities = len(self.__brain.get_labels_and_classes())
            brain_state = claims + entities

        self.__brain_states.append(brain_state)
        self._log.info(f"Brain state: {brain_state}")

        # Reward last thought with R = S_brain(t) - S_brain(t-1)
        if self.__last_thought:
            new_state = self.__brain_states[-1]
            old_state = self.__brain_states[-2]
            reward = new_state - old_state

            self.__thought_selector.update_utility(self.__last_thought, reward)
            self._log.info(f"{reward} reward due to {self.__last_thought}")

    def reply_to_statement(self, brain_response, entity_only=False, proactive=True, persist=False):
        """Selects a Thought from the brain response to verbalize.

        params
        dict brain_response: brain response from brain.update() converted to JSON

        returns: a string representing a verbalized thought
        """
        # Extract thoughts from brain response
        thoughts = thoughts_from_brain(brain_response)

        # Select thought
        self.__last_thought = self.__thought_selector.select(thoughts.keys())
        thought_type, thought_info = thoughts[self.__last_thought]
        self._log.info(f"Chosen thought type: {thought_type}")

        # Preprocess thought_info and utterance (triples)
        utterance = casefold_capsule(brain_response["statement"], format="natural")
        thought_info = {"thought": thought_info}
        thought_info = casefold_capsule(thought_info, format="natural")
        thought_info = thought_info["thought"]

        # Generate reply
        reply = self.phrase_correct_thought(utterance, thought_type, thought_info)

        return reply
