import random
from typing import Optional

from cltl.commons.language_data.sentences import TRUST, NO_TRUST

from cltl.reply_generation import logger


class ThoughtSelector(object):

    def select(self, options):
        raise NotImplementedError()


class Phraser(object):

    def phrase_correct_thought(self, utterance, thought_type, thought_info, fallback=False) -> Optional[str]:
        reply = None
        if thought_type == "_complement_conflict":
            reply = self._phrase_cardinality_conflicts(thought_info, utterance)

        elif thought_type == "_negation_conflicts":
            reply = self._phrase_negation_conflicts(thought_info, utterance)

        elif thought_type == "_statement_novelty":
            reply = self._phrase_statement_novelty(thought_info, utterance)

        elif thought_type == "_entity_novelty":
            reply = self._phrase_type_novelty(thought_info, utterance)

        elif thought_type == "_complement_gaps":
            reply = self._phrase_complement_gaps(thought_info, utterance)

        elif thought_type == "_subject_gaps":
            reply = self._phrase_subject_gaps(thought_info, utterance)

        elif thought_type == "_overlaps":
            reply = self._phrase_overlaps(thought_info, utterance)

        elif thought_type == "_trust":
            reply = self._phrase_trust(thought_info, utterance)

        if fallback and reply is None:  # Fallback strategy
            reply = self.phrase_fallback()

        # Formatting
        if reply:
            reply = reply.replace("-", " ").replace("  ", " ")

        return reply

    def _phrase_cardinality_conflicts(self, thought_info: dict, utterance: dict) -> Optional[str]:
        raise NotImplementedError()

    def _phrase_negation_conflicts(self, thought_info: dict, utterance: dict) -> Optional[str]:
        raise NotImplementedError()

    def _phrase_statement_novelty(self, thought_info: dict, utterance: dict) -> Optional[str]:
        raise NotImplementedError()

    def _phrase_type_novelty(self, thought_info: dict, utterance: dict) -> Optional[str]:
        raise NotImplementedError()

    def _phrase_complement_gaps(self, thought_info: dict, utterance: dict) -> Optional[str]:
        raise NotImplementedError()

    def _phrase_subject_gaps(self, thought_info: dict, utterance: dict) -> Optional[str]:
        raise NotImplementedError()

    def _phrase_overlaps(self, all_overlaps: dict, utterance: dict) -> Optional[str]:
        raise NotImplementedError()

    @staticmethod
    def _phrase_trust(trust: float, utterance: dict) -> Optional[str]:

        if not trust:
            return None

        elif float(trust) > 0.25:
            say = random.choice(TRUST)
        else:
            say = random.choice(NO_TRUST)

        return say

    @staticmethod
    def phrase_fallback() -> Optional[str]:
        """Phrases a fallback utterance when an error has occurred or no
        thoughts were generated.

        returns: phrase
        """
        return "I am out of words."


class BasicReplier(object):

    def __init__(self):
        # type: () -> None
        """
        Generate natural language based on structured data

        Parameters
        ----------
        """

        self._log = logger.getChild(self.__class__.__name__)
        self._log.info("Booted")
        self._thought_selector = ThoughtSelector()
        self._phraser = Phraser()

    @property
    def thought_selector(self):
        return self._thought_selector

    @property
    def phraser(self):
        return self._phraser

    def reply_to_question(self, brain_response):
        raise NotImplementedError()

    def reply_to_statement(self, brain_response, persist=False, thought_options=None):
        raise NotImplementedError()

    def reply_to_mention(self, brain_response, persist=False, thought_options=None):
        raise NotImplementedError()
