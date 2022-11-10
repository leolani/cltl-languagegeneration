""" Filename:     nsp_replier.py
    Author(s):    Thomas Bellucci
    Description:  Modified LenkaRepliers for use in the Chatbot. The replier
                  takes out the Thought selection step and only selects among
                  thoughts using Next Sentence Prediction with BERT (NSPReplier).
    Date created: Nov. 11th, 2021
"""

from cltl.commons.casefolding import (casefold_capsule)

from cltl.reply_generation.lenka_replier import LenkaReplier
from cltl.reply_generation.phrasers.pattern_phraser import PatternPhraser
from cltl.reply_generation.thought_selectors.nsp_selector import NSP
from cltl.reply_generation.utils.thought_utils import thoughts_from_brain


class NSPReplier(LenkaReplier):
    def __init__(self, model_filepath):
        """Creates a replier to respond to questions and statements by the
        user. Statements are replied to by phrasing a thought. Selection
        is performed through Next Sentence Prediction (NSP).

        params
        str model_filepath:  file with a pretrained BERT NSP nsp_model

        returns: None
        """
        super(NSPReplier, self).__init__()
        self._thought_selector = NSP(model_filepath)
        self._log.debug(f"NSP Selector ready")

        self._phraser = PatternPhraser()
        self._log.debug(f"Pattern phraser ready")

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

        # Csefold
        utterance = casefold_capsule(brain_response['statement'], format='natural')
        thoughts = casefold_capsule(brain_response['thoughts'], format='natural')

        # Extract thoughts from brain response
        thoughts = thoughts_from_brain(utterance, thoughts, filter=thought_options)

        # Score phrasings of thoughts
        data = []
        for thought_type, thought_info in thoughts.values():
            # preprocess
            thought_info = {"thought": thought_info}
            thought_info = casefold_capsule(thought_info, format="natural")
            thought_info = thought_info["thought"]

            # Generate reply
            reply = self._phraser.phrase_correct_thought(utterance, thought_type, thought_info)

            # Score response w.r.t. context
            context = utterance["utterance"]
            score = self._thought_selector.score_response(context, reply)
            data.append((thought_type, reply, score))

        # Select thought
        best = self._thought_selector.select(data)
        self._log.info(f"Chosen thought type: {best[0]}")
        self._log.info(f"Response score: {best[2]}")

        return best[1]

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

        # Score phrasings of thoughts
        data = []
        for thought_type, thought_info in thoughts.values():
            # preprocess
            thought_info = {"thought": thought_info}
            thought_info = casefold_capsule(thought_info, format="natural")
            thought_info = thought_info["thought"]

            # Generate reply
            reply = self._phraser.phrase_correct_thought(utterance, thought_type, thought_info)

            # Score response w.r.t. context
            context = utterance["utterance"]
            score = self._thought_selector.score_response(context, reply)
            data.append((thought_type, reply, score))

        # Select thought
        best = self._thought_selector.select(data)
        self._log.info(f"Chosen thought type: {best[0]}")
        self._log.info(f"Response score: {best[2]}")

        return best[1]
