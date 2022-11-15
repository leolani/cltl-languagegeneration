""" Filename:     rl_replier.py
    Author(s):    Thomas Bellucci
    Description:  Modified LenkaRepliers for use in the Chatbot. The replier
                  takes out the Thought selection step and only selects among
                  thoughts using reinforcement learning (RLReplier).
    Date created: Nov. 11th, 2021
"""

from cltl.commons.casefolding import (casefold_capsule)

from cltl.reply_generation.lenka_replier import LenkaReplier
from cltl.reply_generation.phrasers.pattern_phraser import PatternPhraser
from cltl.reply_generation.thought_selectors.rl_selector import UCB
from cltl.reply_generation.utils.thought_utils import thoughts_from_brain


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
        self._thought_selector = UCB()
        self._log.debug(f"UCB RL Selector ready")

        self._phraser = PatternPhraser()
        self._log.debug(f"Pattern phraser ready")

        self._brain = brain
        self._thought_selector.load(savefile)
        self._last_thought = None
        self._brain_states = []

    @property
    def brain_states(self):
        return self._brain_states

    def _evaluate_brain_state(self):
        claims = float(self._brain.count_statements())
        entities = len(self._brain.get_labels_and_classes())
        brain_state = claims + entities

        return brain_state

    def reward_thought(self):
        """Rewards the last thought phrased by the replier by updating its
        utility estimate with the relative improvement of the brain as
        a result of the user response (i.e. a reward).

        returns: None
        """
        brain_state = 0

        if self._brain:
            # Re-evaluate state of brain
            brain_state = self._evaluate_brain_state()

        self._brain_states.append(brain_state)
        self._log.info(f"Brain state: {brain_state}")

        # Reward last thought with R = S_brain(t) - S_brain(t-1)
        if self._last_thought and self._brain_states[-1] and self._brain_states[-1]:
            new_state = self._brain_states[-1]
            old_state = self._brain_states[-2]
            reward = new_state - old_state

            self._thought_selector.update_utility(self._last_thought, reward)
            self._log.info(f"{reward} reward due to {self._last_thought}")

    def reply_to_statement(self, brain_response, persist=False, thought_options=None):
        """
        Phrase a thought based on the brain response
        Parameters
        ----------
        brain_response: output of the brain
        persist: Keep looping through thoughts until you find one to phrase
        thought_options: Set from before which types of thoughts to consider in the phrasing

        Returns
        -------

        """
        # Quick check if there is anything to do here
        if not brain_response['statement']['triple']:
            return None

        # What types of thoughts will we phrase?
        if not thought_options:
            thought_options = ['_entity_novelty', '_complement_gaps']
        self._log.debug(f'Thoughts options: {thought_options}')

        # Casefold
        utterance = casefold_capsule(brain_response['statement'], format='natural')
        thoughts = casefold_capsule(brain_response['thoughts'], format='natural')

        # Extract thoughts from brain response
        thoughts = thoughts_from_brain(utterance, thoughts, filter=thought_options)

        # Select thought
        self._last_thought = self._thought_selector.select(thoughts.keys())
        thought_type, thought_info = thoughts[self._last_thought]
        self._log.info(f"Chosen thought type: {thought_type}")

        # Preprocess thought_info and utterance (triples)
        utterance = casefold_capsule(brain_response["statement"], format="natural")
        thought_info = {"thought": thought_info}
        thought_info = casefold_capsule(thought_info, format="natural")
        thought_info = thought_info["thought"]

        # Generate reply
        reply = self._phraser.phrase_correct_thought(utterance, thought_type, thought_info)

        return reply

    def reply_to_mention(self, brain_response, persist=False, thought_options=None):
        """
        Phrase a thought based on the brain response
        Parameters
        ----------
        brain_response: output of the brain
        persist: Keep looping through thoughts until you find one to phrase
        thought_options: Set from before which types of thoughts to consider in the phrasing

        Returns
        -------

        """
        # Quick check if there is anything to do here
        if not brain_response['mention']['entity']:
            return None

        # What types of thoughts will we phrase?
        if not thought_options:
            thought_options = ['_entity_novelty', '_complement_gaps']
        self._log.debug(f'Thoughts options: {thought_options}')

        # Casefold
        utterance = casefold_capsule(brain_response['mention'], format='natural')
        thoughts = casefold_capsule(brain_response['thoughts'], format='natural')

        # Extract thoughts from brain response
        thoughts = thoughts_from_brain(utterance, thoughts, filter=thought_options)

        # Select thought
        self._last_thought = self._thought_selector.select(thoughts.keys())
        thought_type, thought_info = thoughts[self._last_thought]
        self._log.info(f"Chosen thought type: {thought_type}")

        # Preprocess thought_info and utterance (triples)
        utterance = casefold_capsule(brain_response["statement"], format="natural")
        thought_info = {"thought": thought_info}
        thought_info = casefold_capsule(thought_info, format="natural")
        thought_info = thought_info["thought"]

        # Generate reply
        reply = self._phraser.phrase_correct_thought(utterance, thought_type, thought_info)

        return reply
