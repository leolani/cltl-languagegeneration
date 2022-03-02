""" Filename:     nsp_replier.py
    Author(s):    Thomas Bellucci
    Description:  Modified LenkaRepliers for use in the Chatbot. The replier
                  takes out the Thought selection step and only selects among
                  thoughts using Next Sentence Prediction with BERT (NSPReplier).
    Date created: Nov. 11th, 2021
"""

from cltl.combot.backend.utils.casefolding import (casefold_capsule)
from cltl.reply_generation.lenka_replier import LenkaReplier
from cltl.reply_generation.next_sentence_prediction.nsp import NSP
from cltl.reply_generation.utils.replier_utils import thoughts_from_brain


class NSPReplier(LenkaReplier):
    def __init__(self, model_filepath):
        """Creates a replier to respond to questions and statements by the
        user. Statements are replied to by phrasing a thought. Selection
        is performed through Next Sentence Prediction (NSP).

        params
        str model_filepath:  file with a pretrained BERT NSP model

        returns: None
        """
        super(NSPReplier, self).__init__()
        self._thought_selector = NSP(model_filepath)
        self._log.debug(f"NSP Selector ready")

    def reply_to_statement(self, brain_response, entity_only=False, proactive=True, persist=False):
        """Selects a Thought from the brain response to verbalize.

        params
        dict brain_response: brain response from brain.update() converted to JSON

        returns: a string representing a verbalized thought
        """
        # Preprocess
        utterance = casefold_capsule(brain_response["statement"], format="natural")

        # Extract thoughts from brain response
        thoughts = thoughts_from_brain(brain_response)

        # Score phrasings of thoughts
        data = []
        for thought_type, thought_info in thoughts.values():
            # preprocess
            thought_info = {"thought": thought_info}
            thought_info = casefold_capsule(thought_info, format="natural")
            thought_info = thought_info["thought"]

            # Generate reply
            reply = self.phrase_correct_thought(utterance, thought_type, thought_info)

            # Score response w.r.t. context
            context = utterance["utterance"]
            score = self._thought_selector.score_response(context, reply)
            data.append((thought_type, reply, score))

        # Select thought
        best = self._thought_selector.select(data)
        self._log.info(f"Chosen thought type: {best[0]}")
        self._log.info(f"Response score: {best[2]}")

        return best[1]
