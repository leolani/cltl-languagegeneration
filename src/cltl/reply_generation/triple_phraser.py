import copy
from typing import Optional, Tuple

from cltl.commons.casefolding import casefold_text
from cltl.reply_generation.simplenlg_phraser import SimplenlgPhraser
from cltl.thoughts.thought_selection.utils.thought_utils import separate_select_negation_conflicts
from dialogue_system.utils.global_variables import BASE_CAPSULE


def label_in_say(label, say):
    if say:
        return label.lower() in say.lower()
    else:
        return False


class TriplePhraser(SimplenlgPhraser):

    def __init__(self):
        # type: () -> None
        """
        Generate response as structured data (thought to capsule)

        Parameters
        ----------
        """

        super(SimplenlgPhraser, self).__init__()

    def phrase_correct_thought(self, utterance, thought_type, thought_info, fallback=False) -> Tuple[
        Optional[str], Optional[dict]]:
        reply = None
        response_template = None

        if thought_type == "_complement_conflict":
            reply, response_template = self._phrase_cardinality_conflicts(thought_info, utterance)

        elif thought_type == "_negation_conflicts":
            reply, response_template = self._phrase_negation_conflicts(thought_info, utterance)

        elif thought_type == "_statement_novelty":
            reply, response_template = self._phrase_statement_novelty(thought_info, utterance)

        elif thought_type == "_entity_novelty":
            reply, response_template = self._phrase_type_novelty(thought_info, utterance)

        elif thought_type == "_complement_gaps":
            reply, response_template = self._phrase_complement_gaps(thought_info, utterance)

        elif thought_type == "_subject_gaps":
            reply, response_template = self._phrase_subject_gaps(thought_info, utterance)

        elif thought_type == "_overlaps":
            reply, response_template = self._phrase_overlaps(thought_info, utterance)

        elif thought_type == "_trust":
            reply, response_template = self._phrase_trust(thought_info, utterance)

        if fallback and reply is None:  # Fallback strategy
            reply, response_template = self.phrase_fallback()

        # Formatting
        if reply:
            reply = casefold_text(reply, format='natural')

        return reply, response_template

    @staticmethod
    def _phrase_cardinality_conflicts(selected_thought: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = copy.deepcopy(BASE_CAPSULE), \
                            super(TriplePhraser, TriplePhraser)._phrase_cardinality_conflicts(selected_thought,
                                                                                              utterance)

        # There is no conflict, so nothing
        if not selected_thought or not selected_thought["thought_info"]:
            return say, capsule_user

        # There is a conflict, so we phrase it
        else:
            conflict = selected_thought["thought_info"]

            # Capsule with other conflicting triple, user should set the polarity and certainty
            capsule_user['subject'] = utterance['subject']
            capsule_user['predicate'] = utterance['predicate']
            capsule_user["object"] = {"label": conflict['_complement']['_label'],
                                      "type": conflict['_complement']['_types'],
                                      'uri': conflict['_complement']['_id']}

        return say, capsule_user

    @staticmethod
    def _phrase_negation_conflicts(selected_thought: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = copy.deepcopy(BASE_CAPSULE), \
                            super(TriplePhraser, TriplePhraser)._phrase_negation_conflicts(selected_thought, utterance)

        # There is no conflict, so no response
        if not selected_thought or not selected_thought["thought_info"]:
            return say, capsule_user

        # There is conflict entries
        else:
            conflicts = selected_thought["thought_info"]
            affirmative_conflict, negative_conflict = separate_select_negation_conflicts(conflicts)

            # There is a conflict, so we phrase it
            if affirmative_conflict and negative_conflict:
                # Capsule with original triple, user should set the polarity and certainty
                capsule_user['subject'] = utterance['subject']
                capsule_user['predicate'] = utterance['predicate']
                capsule_user['object'] = utterance['object']

        return say, capsule_user

    @staticmethod
    def _phrase_statement_novelty(selected_thought: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = copy.deepcopy(BASE_CAPSULE), \
                            super(TriplePhraser, TriplePhraser)._phrase_statement_novelty(selected_thought, utterance)

        # I do not know this before, so be happy to learn
        if not selected_thought or not selected_thought["thought_info"]:
            entity_role = selected_thought["extra_info"]

            if entity_role == '_subject':
                # Capsule with original subject & predicate, user should change object so we keep learning similar facts
                capsule_user['subject'] = utterance['subject']
                capsule_user['predicate'] = utterance['predicate']

            elif entity_role == '_complement':
                # Capsule with some original parts, user should change subject so we keep learning similar facts
                capsule_user['predicate'] = utterance['predicate']
                capsule_user['object'] = utterance['object']

        # I already knew this
        else:
            # Capsule with original triple, user should add perspective
            capsule_user['subject'] = utterance['subject']
            capsule_user['predicate'] = utterance['predicate']
            capsule_user['object'] = utterance['object']

        return say, capsule_user

    @staticmethod
    def _phrase_type_novelty(selected_thought: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = copy.deepcopy(BASE_CAPSULE), \
                            super(TriplePhraser, TriplePhraser)._phrase_type_novelty(selected_thought, utterance)

        entity_role = selected_thought["extra_info"]

        # There is no novelty information, so happy to learn
        if not selected_thought or not selected_thought["thought_info"] or selected_thought["thought_info"]["value"]:
            # Capsule with original triple subject or object, user should add predicate and other entity to keep learning
            if entity_role == '_subject':
                capsule_user['subject'] = utterance['subject']
            else:
                capsule_user['object'] = utterance['object']

        # I know this
        else:
            # Capsule with original triple predicate, user should add other entities to keep learning
            capsule_user['predicate'] = utterance['predicate']

            if entity_role == '_subject':
                capsule_user['subject'] = utterance['subject']
            else:
                capsule_user['object'] = utterance['object']

        return say, capsule_user

    @staticmethod
    def _phrase_subject_gaps(selected_thought: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = copy.deepcopy(BASE_CAPSULE), \
                            super(TriplePhraser, TriplePhraser)._phrase_subject_gaps(selected_thought, utterance)

        # There is no gaps, so no response
        if not selected_thought or not selected_thought["thought_info"]:
            return say, capsule_user

        # There is a gap
        else:
            entity_role = selected_thought["extra_info"]
            gap = selected_thought["thought_info"]

        if entity_role == '_subject':
            # Capsule with original triple subject + gap info, user should add object label
            capsule_user['subject'] = utterance['subject']
            capsule_user['predicate'] = {"label": gap["_predicate"]["_label"], "uri": gap["_predicate"]["_id"]}
            capsule_user['object'] = {"label": None, "type": gap["_target_entity_type"]["_types"], "uri": None}


        else:
            # Capsule with original triple object + gap info, user should add subject label
            capsule_user['subject'] = {"label": None, "type": gap['_target_entity_type']['_types'], 'uri': None}
            capsule_user['predicate'] = {"label": gap['_predicate']['_label'], 'uri': gap['_predicate']['_id']}
            capsule_user['object'] = utterance['subject']

        return say, capsule_user

    @staticmethod
    def _phrase_complement_gaps(selected_thought: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = copy.deepcopy(BASE_CAPSULE), \
                            super(TriplePhraser, TriplePhraser)._phrase_complement_gaps(selected_thought, utterance)

        # There is no gaps, so no response
        if not selected_thought or not selected_thought["thought_info"]:
            return say, capsule_user

        # There is a gap
        else:
            entity_role = selected_thought["extra_info"]
            gap = selected_thought["thought_info"]

        if entity_role == '_subject':
            # Capsule with original object as subject + gap info, user should add object label
            capsule_user['subject'] = utterance['object']
            capsule_user['predicate'] = {"label": gap['_predicate']['_label'], 'uri': gap['_predicate']['_id']}
            capsule_user['object'] = {"label": None, "type": gap['_target_entity_type']['_types'], 'uri': None}

        else:
            # Capsule with original triple object + gap info, user should add subject label
            capsule_user['subject'] = {"label": None, "type": gap['_target_entity_type']['_types'], 'uri': None}
            capsule_user['predicate'] = {"label": gap['_predicate']['_label'], 'uri': gap['_predicate']['_id']}
            capsule_user['object'] = utterance['object']

        return say, capsule_user

    @staticmethod
    def _phrase_overlaps(selected_thought: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = copy.deepcopy(BASE_CAPSULE), \
                            super(TriplePhraser, TriplePhraser)._phrase_overlaps(selected_thought, utterance)

        if not selected_thought or not selected_thought["thought_info"]:
            return say, capsule_user

        entity_role = selected_thought["extra_info"]
        overlap = selected_thought["thought_info"]

        if entity_role == '_subject':
            # Capsule with original triple subject + overlap info, user should add perspective
            capsule_user['subject'] = utterance['subject']
            capsule_user['predicate'] = utterance['predicate']
            capsule_user['object'] = {"label": overlap['_entity']['_label'],
                                      "type": overlap['_entity']['_types'],
                                      'uri': overlap['_entity']['_id']}


        elif entity_role == '_complement':
            # Capsule with original triple object + overlap info, user should add perspective
            capsule_user['subject'] = {"label": overlap['_entity']['_label'],
                                       "type": overlap['_entity']['_types'],
                                       'uri': overlap['_entity']['_id']}
            capsule_user['predicate'] = utterance['predicate']
            capsule_user['object'] = utterance['object']

        return say, capsule_user

    @staticmethod
    def _phrase_trust(trust: dict, utterance: dict) -> Tuple[Optional[str], Optional[dict]]:
        capsule_user, say = copy.deepcopy(BASE_CAPSULE), \
                            super(TriplePhraser, TriplePhraser)._phrase_trust(trust, utterance)

        # Capsule with original triple, user should set the polarity and certainty
        capsule_user['subject'] = utterance['subject']
        capsule_user['predicate'] = utterance['predicate']
        capsule_user['object'] = utterance['object']

        return say, capsule_user

    @staticmethod
    def phrase_fallback() -> Tuple[Optional[dict], Optional[dict]]:
        # Capsule with empty triple, redirect conversation
        capsule_user, say = copy.deepcopy(BASE_CAPSULE), \
                            super(TriplePhraser, TriplePhraser).phrase_fallback()

        return say, capsule_user

    def reply_to_statement(self, brain_response, persist=False):
        """
        Phrase a thought based on the brain response
        Parameters
        ----------
        brain_response: output of the brain
        persist: Call fallback

        Returns
        -------

                """
        # Quick check if there is anything to do here
        if not brain_response['statement']['triple']:
            return copy.deepcopy(BASE_CAPSULE), None

        # Generate reply
        (thought_type, thought_info) = list(brain_response['thoughts'].items())[0]
        reply = self.phrase_correct_thought(brain_response['statement'], thought_type, thought_info, fallback=persist)

        if persist and reply[0] is None and reply[1] is None:
            reply = self.phrase_fallback()

        return reply
