import random

from cltl.commons.casefolding import casefold_capsule
from cltl.commons.language_data.sentences import NO_ANSWER
from cltl.commons.language_helpers import lexicon_lookup
from cltl.commons.triple_helpers import filtered_types_names

from cltl.reply_generation.api import BasicReplier
from cltl.reply_generation.phrasers.pattern_phraser import PatternPhraser
from cltl.reply_generation.thought_selectors.random_selector import RandomSelector
from cltl.reply_generation.utils.phraser_utils import replace_pronouns


class LenkaReplier(BasicReplier):

    def __init__(self):
        # type: () -> None
        """
        Generate natural language based on structured data

        Parameters
        ----------
        """

        super(LenkaReplier, self).__init__()
        self._thought_selector = RandomSelector()
        self._log.debug(f"Random Selector ready")

        self._phraser = PatternPhraser()
        self._log.debug(f"Pattern phraser ready")

    def reply_to_question(self, brain_response):
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
            subject, predicate, object = self._assign_spo(utterance, item)

            author = replace_pronouns(utterance['author']['label'], author=item['authorlabel']['value'])
            subject = replace_pronouns(utterance['author']['label'], entity_label=subject, role='subject')
            object = replace_pronouns(utterance['author']['label'], entity_label=object, role='object')

            subject = self._fix_entity(subject, utterance['author']['label'])
            object = self._fix_entity(object, utterance['author']['label'])

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
            say, previous_author = self._deal_with_authors(author, previous_author, predicate, previous_predicate, say)

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

    def reply_to_statement(self, brain_response, entity_only=False, proactive=True, persist=False):
        """
        Phrase a random thought
        Parameters
        ----------
        brain_response: output of the brain
        entity_only: Focus on thoughts related to entities (entity novelty, and gaps)
        proactive: Include gaps and overlaps for a more proactive agent (an agents that asks questions and bonds over
                known information)
        persist: Keep looping through thoughts until you find one to phrase

        Returns
        -------

        """
        # Quick check if there is anything to do here
        if 'statement' not in brain_response.keys() or brain_response['statement']['triple'] is None:
            return None

        # What types of thoughts will we phrase?
        utterance = brain_response['statement']
        if entity_only:
            options = ['_entity_novelty', '_subject_gaps', '_complement_gaps']
        else:
            options = ['_complement_conflict', '_negation_conflicts', '_statement_novelty', '_entity_novelty', '_trust']

        if proactive:
            options.extend(['_subject_gaps', '_complement_gaps', '_overlaps'])
        self._log.debug(f'Thoughts options: {options}')

        # Casefold
        utterance = casefold_capsule(utterance, format='natural')
        thoughts = brain_response['thoughts']
        thoughts = casefold_capsule(thoughts, format='natural')

        # Filter out None thoughts
        options = [option for option in options if thoughts[option] is not None]

        if not options:
            reply = self._phraser.phrase_fallback()
        else:
            # Select thought
            thought_type = self._thought_selector.select(options)
            self._log.info(f"Chosen thought type: {thought_type}")

            # Generate reply
            reply = self._phraser.phrase_correct_thought(utterance, thought_type, thoughts[thought_type])

            if persist and reply is None:
                reply = self.reply_to_statement(brain_response, proactive=proactive, persist=persist)

        return reply

    @staticmethod
    def _assign_spo(utterance, item):
        empty = ['', 'unknown', 'none']

        # INITIALIZATION
        predicate = utterance['predicate']['label']

        if utterance['subject']['label'] is None or utterance['subject']['label'].lower() in empty:
            subject = item['slabel']['value']
        else:
            subject = utterance['subject']['label']

        if utterance['object']['label'] is None or utterance['object']['label'].lower() in empty:
            object = item['olabel']['value']
        else:
            object = utterance['object']['label']

        return subject, predicate, object

    @staticmethod
    def _deal_with_authors(author, previous_author, predicate, previous_predicate, say):
        # Deal with author
        if not author:
            author = "someone"

        if author != previous_author:
            say += author + ' told me '
            previous_author = author
        else:
            if predicate != previous_predicate:
                say += ' that '

        return say, previous_author

    def _fix_entity(self, entity, speaker):
        new_ent = ''
        if '-' in entity:
            entity_tokens = entity.split('-')

            for word in entity_tokens:
                new_ent += replace_pronouns(speaker, entity_label=word, role='pos') + ' '

        else:
            new_ent += replace_pronouns(speaker, entity_label=entity)

        entity = new_ent
        return entity
