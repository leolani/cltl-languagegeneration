import random
from typing import Optional

from cltl.commons.casefolding import (casefold_capsule)
from cltl.commons.language_data.sentences import NO_ANSWER
from cltl.commons.language_data.sentences import TRUST, NO_TRUST
from cltl.commons.language_helpers import lexicon_lookup
from cltl.commons.triple_helpers import filtered_types_names

from cltl.reply_generation import logger
from cltl.reply_generation.utils.phraser_utils import replace_pronouns, assign_spo, deal_with_authors, fix_entity


class Phraser(object):

    def __init__(self):
        # type: () -> None
        """
        Generate natural language based on structured data

        Parameters
        ----------
        """

        self._log = logger.getChild(self.__class__.__name__)
        self._log.info("Booted")

    def phrase_triple(self, utterance):
        raise NotImplementedError()

    def phrase_correct_thought(self, utterance: dict, thought_type: str, thought_info: dict, fallback: bool = False) -> \
            Optional[str]:
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

    def _phrase_trust(self, selected_thought: dict, utterance: dict) -> Optional[str]:
        if not selected_thought or not selected_thought["thought_info"]:
            return None

        trust = selected_thought['thought_info']['value']
        if float(trust) > 0.25:
            say = random.choice(TRUST)
        else:
            say = random.choice(NO_TRUST)

        say = f"{utterance['author']['label']}, {say}"

        return say

    @staticmethod
    def phrase_fallback() -> Optional[str]:
        """Phrases a fallback utterance when an error has occurred or no
        thoughts were generated.

        returns: phrase
        """
        return "I am out of words."

    def reply_to_statement(self, brain_response, persist=False, casefold=True):
        """
        Phrase a thought based on the brain response
        Parameters
        ----------
        brain_response: output of the brain
        persist: Call fallback
        casefold: Whether to change the entity labels to natural language

        Returns
        -------

                """
        # Quick check if there is anything to do here
        if not brain_response['statement']['triple']:
            return None

        if casefold:
            # Casefold
            utterance = casefold_capsule(brain_response['statement'], format='natural')
            thoughts = casefold_capsule(brain_response['thoughts'], format='natural')
        else:
            utterance = brain_response['statement']
            thoughts = brain_response['thoughts']

        # Generate reply
        (thought_type, thought_info) = list(thoughts.items())[0]
        reply = self.phrase_correct_thought(utterance, thought_type, thought_info, fallback=not persist)

        if persist and reply is None:
            reply = self.phrase_fallback()

        return reply

    def reply_to_mention(self, brain_response, persist=False, thought_options=None):
        # Quick check if there is anything to do here
        if not brain_response['mention']['entity']:
            return None

        # Casefold
        utterance = casefold_capsule(brain_response['mention'], format='natural')
        thoughts = casefold_capsule(brain_response['thoughts'], format='natural')

        # Generate reply
        (thought_type, thought_info) = list(thoughts.items())[0]
        reply = self.phrase_correct_thought(utterance, thought_type, thought_info, fallback=not persist)

        if persist and reply is None:
            reply = self.phrase_fallback()

        return reply

    def _phrase_no_answer_to_question(self, utterance):
        self._log.info(f"Empty response")
        subject_types = filtered_types_names(utterance['subject']['type']) \
            if utterance['subject']['type'] is not None else ''
        object_types = filtered_types_names(utterance['object']['type']) \
            if utterance['object']['type'] is not None else ''

        if subject_types and object_types and utterance['predicate']['label']:
            say = "I know %s usually %s %s, but I do not know this case" % (
                random.choice(utterance['subject']['type']),
                str(utterance['predicate']['label']),
                random.choice(utterance['object']['type']))
            return say

        else:
            return random.choice(NO_ANSWER)

    def reply_to_question(self, brain_response):
        # Quick check if there is anything to do here
        if not brain_response['response']:
            return self._phrase_no_answer_to_question(brain_response['question'])

        # TODO revise (we conjugate the predicate by doing this)
        utterance = casefold_capsule(brain_response['question'], format='natural')

        # Each triple is hashed, so we can figure out when we are about the say things double
        handled_items = set()
        brain_response['response'].sort(key=lambda x: x['authorlabel']['value'])

        say = ''
        previous_author = ''
        previous_predicate = ''
        gram_person = ''
        gram_number = ''
        for item in brain_response['response']:
            # INITIALIZATION
            subject, predicate, object = assign_spo(utterance, item)

            author = replace_pronouns(utterance['author']['label'], author=item['authorlabel']['value'])
            subject = replace_pronouns(utterance['author']['label'], entity_label=subject, role='subject')
            object = replace_pronouns(utterance['author']['label'], entity_label=object, role='object')

            subject = fix_entity(subject, utterance['author']['label'])
            object = fix_entity(object, utterance['author']['label'])

            # Hash item such that duplicate entries have the same hash
            item_hash = '{}_{}_{}_{}'.format(subject, predicate, object, author)

            # If this hash is already in handled items -> skip this item and move to the next one
            if item_hash in handled_items:
                continue
            # Otherwise, add this item to the handled items (and handle item the usual way (with the code below))
            else:
                handled_items.add(item_hash)

            # Get grammatical properties
            subject_entry = lexicon_lookup(subject.lower())
            if subject_entry and 'person' in subject_entry:
                gram_person = subject_entry['person']
            if subject_entry and 'number' in subject_entry:
                gram_number = subject_entry['number']

            # Deal with author
            say, previous_author = deal_with_authors(author, previous_author, predicate, previous_predicate, say)

            if predicate.endswith('is'):

                say += object + ' is'
                if utterance['object']['label'].lower() == utterance['author']['label'].lower() or \
                        utterance['subject']['label'].lower() == utterance['author']['label'].lower():
                    say += ' your '
                elif utterance['object']['label'].lower() == 'leolani' or \
                        utterance['subject']['label'].lower() == 'leolani':
                    say += ' my '
                say += predicate[:-3]

                return say

            else:  # TODO fix_predicate_morphology
                be = {'first': 'am', 'second': 'are', 'third': 'is'}
                if predicate == 'be':  # or third person singular
                    if gram_number:
                        if gram_number == 'singular':
                            predicate = be[gram_person]
                        else:
                            predicate = 'are'
                    else:
                        # TODO: Is this a good default when 'number' is unknown?
                        predicate = 'is'
                elif gram_person == 'third' and '-' not in predicate:
                    predicate += 's'

                if item['certaintyValue']['value'] != 'CERTAIN':  # TODO extract correct certainty marker
                    predicate = 'maybe ' + predicate

                if item['polarityValue']['value'] != 'POSITIVE':
                    if ' ' in predicate:
                        predicate = predicate.split()[0] + ' not ' + predicate.split()[1]
                    else:
                        predicate = 'do not ' + predicate

                say += subject + ' ' + predicate + ' ' + object

            say += ' and '

        # Remove last ' and' and return
        say = say[:-5]
        return say.replace('-', ' ').replace('  ', ' ')


class Rephraser(object):

    def __init__(self):
        # type: () -> None
        """
        Rephrase formulation of natural language based on structured data

        Parameters
        ----------
        """

        self._log = logger.getChild(self.__class__.__name__)
        self._log.info("Booted")
        self._phraser = Phraser()

    @property
    def phraser(self):
        return self._phraser

    def llamalize_reply(self, text_response):
        raise NotImplementedError()
