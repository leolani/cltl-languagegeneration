import random

from cltl.combot.backend.utils.casefolding import casefold_text, casefold_capsule
from cltl.reply_generation.api import BasicReplier
from cltl.reply_generation.data.sentences import NEW_KNOWLEDGE, EXISTING_KNOWLEDGE, CONFLICTING_KNOWLEDGE, \
    CURIOSITY, HAPPY, TRUST, NO_TRUST
from cltl.reply_generation.utils.helper_functions import lexicon_lookup


class LenkaReplier(BasicReplier):

    def __init__(self):
        # type: () -> None
        """
        Generate natural language based on structured data

        Parameters
        ----------
        """

        super(LenkaReplier, self).__init__()

    def reply_to_question(self, brain_response):
        say = ''
        previous_author = ''
        previous_predicate = ''
        gram_person = ''
        gram_number = ''

        utterance = brain_response['question']
        response = brain_response['response']

        # TODO revise (we conjugate the predicate by doing this)
        utterance = casefold_capsule(utterance, format='natural')

        if not response:
            if ' or '.join(utterance['subject']['type']) \
                    and ' or '.join(utterance['object']['type']) \
                    and utterance['predicate']['type']:
                say += "I know %s usually %s %s, but I do not know this case" % (
                    random.choice(utterance['subject']['type']),
                    str(utterance['predicate']['type']),
                    random.choice(utterance['object']['type']))
                return say

            else:
                return None

        # Each triple is hashed, so we can figure out when we are about the say things double
        handled_items = set()
        response.sort(key=lambda x: x['authorlabel']['value'])

        for item in response:

            # INITIALIZATION
            subject, predicate, object = self._assign_spo(utterance, item)

            author = self._replace_pronouns(utterance['author'], author=item['authorlabel']['value'])
            subject = self._replace_pronouns(utterance['author'], entity_label=subject, role='subject')
            object = self._replace_pronouns(utterance['author'], entity_label=object, role='object')

            subject = self._fix_entity(subject, utterance['author'])
            object = self._fix_entity(object, utterance['author'])

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
                if utterance['object']['label'].lower() == utterance['author'].lower() or \
                        utterance['subject']['label'].lower() == utterance['author'].lower():
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

        say = say[:-5]

        return say.replace('-', ' ').replace('  ', ' ')

    def reply_to_statement(self, brain_response, entity_only=True, proactive=True, persist=False):
        """
        Phrase a random thought
        Parameters
        ----------
        brain_response
        entity_only
        proactive
        persist

        Returns
        -------

        """
        if entity_only:
            options = ['cardinality_conflicts', 'negation_conflicts', 'statement_novelty', 'entity_novelty', 'trust']
        else:
            options = ['cardinality_conflicts', 'entity_novelty', 'trust']

        if proactive:
            options.extend(['subject_gaps', 'object_gaps', 'overlaps'])

        # Casefold and select approach randomly
        utterance = brain_response['statement']
        if utterance['triple'] is None:
            return None

        utterance = casefold_capsule(utterance, format='natural')
        thoughts = brain_response['thoughts']
        thoughts = casefold_capsule(thoughts, format='natural')
        approach = random.choice(options)
        say = None

        if approach == 'cardinality_conflicts':
            say = self._phrase_cardinality_conflicts(thoughts['_complement_conflict'], utterance)

        elif approach == 'negation_conflicts':
            say = self._phrase_negation_conflicts(thoughts['_negation_conflicts'], utterance)

        elif approach == 'statement_novelty':
            say = self._phrase_statement_novelty(thoughts['_statement_novelty'], utterance)

        elif approach == 'entity_novelty':
            say = self._phrase_type_novelty(thoughts['_entity_novelty'], utterance)

        elif approach == 'subject_gaps':
            say = self._phrase_subject_gaps(thoughts['_subject_gaps'], utterance)

        elif approach == 'object_gaps':
            say = self._phrase_complement_gaps(thoughts['_complement_gaps'], utterance)

        elif approach == 'overlaps':
            say = self._phrase_overlaps(thoughts['_overlaps'], utterance)

        if persist and say is None:
            say = self.reply_to_statement(brain_response, proactive=proactive, persist=persist)

        if say and '-' in say:
            say = say.replace('-', ' ').replace('  ', ' ')

        return say

    def phrase_all_conflicts(self, conflicts, speaker=None):
        # type: (list[dict], str) -> str

        say = 'I have %s conflicts in my brain.' % len(conflicts)
        conflict = random.choice(conflicts)

        # Conflict of subject
        if len(conflict['objects']) > 1:
            predicate = casefold_text(conflict['predicate'], format='natural')
            options = ['%s %s like %s told me' % (predicate, item['value'], item['author']) for item in
                       conflict['objects']]
            options = ' or '.join(options)
            subject = self._replace_pronouns(speaker, author=conflict['objects'][1]['author'],
                                             entity_label=conflict['subject'],
                                             role='subject')

            say = say + ' For example, I do not know if %s %s' % (subject, options)

        return say

    @staticmethod
    def _phrase_cardinality_conflicts(conflicts, utterance):
        # type: (list[dict], dict) -> str

        # There is no conflict, so nothing
        if not conflicts:
            say = None

        # There is a conflict, so we phrase it
        else:
            say = random.choice(CONFLICTING_KNOWLEDGE)
            conflict = random.choice(conflicts)
            x = 'you' if conflict['_provenance']['_author'] == utterance['author'] \
                else conflict['_provenance']['_author']
            y = 'you' if utterance['triple']['_subject']['_label'] == conflict['_provenance']['_author'] \
                else utterance['triple']['_subject']['_label']

            # Checked
            say += ' %s told me in %s that %s %s %s, but now you tell me that %s %s %s' \
                   % (x, conflict['_provenance']['_date'], y, utterance['triple']['_predicate']['_label'],
                      conflict['_complement']['_label'],
                      y, utterance['triple']['_predicate']['_label'], utterance['triple']['_complement']['_label'])

        return say

    @staticmethod
    def _phrase_negation_conflicts(conflicts, utterance):
        # type: (list[dict], dict) -> str

        say = None

        # There is conflict entries
        if conflicts and conflicts[0]:
            affirmative_conflict = [item for item in conflicts if item['_polarity_value'] == 'POSITIVE']
            negative_conflict = [item for item in conflicts if item['_polarity_value'] == 'NEGATIVE']

            # There is a conflict, so we phrase it
            if affirmative_conflict and negative_conflict:
                say = random.choice(CONFLICTING_KNOWLEDGE)

                affirmative_conflict = random.choice(affirmative_conflict)
                negative_conflict = random.choice(negative_conflict)

                say += ' %s told me in %s that %s %s %s, but in %s %s told me that %s did not %s %s' \
                       % (affirmative_conflict['_provenance']['_author'], affirmative_conflict['_provenance']['_date'],
                          utterance['triple']['_subject']['_label'], utterance['triple']['_predicate']['_label'],
                          utterance['triple']['_complement']['_label'],
                          negative_conflict['_provenance']['_date'], negative_conflict['_provenance']['_author'],
                          utterance['triple']['_subject']['_label'], utterance['triple']['_predicate']['_label'],
                          utterance['triple']['_complement']['_label'])

        return say

    @staticmethod
    def _phrase_statement_novelty(novelties, utterance):
        # type: (list[dict], dict) -> str

        # I do not know this before, so be happy to learn
        if not novelties:
            entity_role = random.choice(['subject', 'object'])

            say = random.choice(NEW_KNOWLEDGE)

            if entity_role == 'subject':
                if 'person' in ' or '.join(utterance['triple']['_complement']['_types']):
                    any_type = 'anybody'
                elif 'location' in ' or '.join(utterance['triple']['_complement']['_types']):
                    any_type = 'anywhere'
                else:
                    any_type = 'anything'

                # Checked
                say += ' I did not know %s that %s %s' % (any_type, utterance['triple']['_subject']['_label'],
                                                          utterance['triple']['_predicate']['_label'])

            elif entity_role == 'object':
                # Checked
                say += ' I did not know anybody who %s %s' % (utterance['triple']['_predicate']['_label'],
                                                              utterance['triple']['_complement']['_label'])

        # I already knew this
        else:
            say = random.choice(EXISTING_KNOWLEDGE)
            novelty = random.choice(novelties)

            # Checked
            say += ' %s told me about it in %s' % (novelty['_provenance']['_author'],
                                                   novelty['_provenance']['_date'])

        return say

    def _phrase_type_novelty(self, novelties, utterance):
        # type: (dict, dict) -> str

        entity_role = random.choice(['subject', 'object'])
        entity_label = utterance['triple']['_subject']['_label'] \
            if entity_role == 'subject' else utterance['triple']['_complement']['_label']
        novelty = novelties['_subject'] if entity_role == 'subject' else novelties['_complement']

        if novelty:
            entity_label = self._replace_pronouns(utterance['author'], entity_label=entity_label,
                                                  role=entity_role)
            say = random.choice(NEW_KNOWLEDGE)
            if entity_label != 'you':  # TODO or type person?
                # Checked
                say += ' I had never heard about %s before!' % self._replace_pronouns(utterance['author'],
                                                                                      entity_label=entity_label,
                                                                                      role='object')
            else:
                say += ' I am excited to get to know about %s!' % entity_label

        else:
            say = random.choice(EXISTING_KNOWLEDGE)
            if entity_label != 'you':
                # Checked
                say += ' I have heard about %s before' % self._replace_pronouns(utterance['author'],
                                                                                entity_label=entity_label,
                                                                                role='object')
            else:
                say += ' I love learning more and more about %s!' % entity_label

        return say

    @staticmethod
    def _phrase_subject_gaps(all_gaps, utterance):
        # type: (dict, dict) -> str

        entity_role = random.choice(['subject', 'object'])
        gaps = all_gaps['_subject'] if entity_role == 'subject' else all_gaps['_complement']
        say = None

        if entity_role == 'subject':
            say = random.choice(CURIOSITY)

            if not gaps:
                say += ' What types can %s %s' % (utterance['triple']['_subject']['_label'],
                                                  utterance['triple']['_predicate']['_label'])

            else:
                gap = random.choice(gaps)
                if 'is ' in gap['_predicate']['_label'] or ' is' in gap['_predicate']['_label']:
                    say += ' Is there a %s that %s %s?' % (' or'.join(gap['_entity']['_types']),
                                                           gap['_predicate']['_label'],
                                                           utterance['triple']['_subject']['_label'])
                elif ' of' in gap['_predicate']['_label']:
                    say += ' Is there a %s that %s is %s?' % (' or'.join(gap['_entity']['_types']),
                                                              utterance['triple']['_subject']['_label'],
                                                              gap['_predicate']['_label'])

                elif ' ' in gap['_predicate']['_label']:
                    say += ' Is there a %s that is %s %s?' % (' or'.join(gap['_entity']['_types']),
                                                              gap['_predicate']['_label'],
                                                              utterance['triple']['_subject']['_label'])
                else:
                    # Checked
                    say += ' Has %s %s %s?' % (utterance['triple']['_subject']['_label'],
                                               gap['_predicate']['_label'],
                                               ' or'.join(gap['_entity']['_types']))

        elif entity_role == 'object':
            say = random.choice(CURIOSITY)

            if not gaps:
                say += ' What kinds of things can %s a %s like %s' % (utterance['triple']['_predicate']['_label'],
                                                                      utterance['triple']['_complement']['_label'],
                                                                      utterance['triple']['_subject']['_label'])

            else:
                gap = random.choice(gaps)
                if '#' in ' or'.join(gap['_entity']['_types']):
                    say += ' What is %s %s?' % (utterance['triple']['_subject']['_label'],
                                                gap['_predicate']['_label'])
                elif ' ' in gap['_predicate']['_label']:
                    # Checked
                    say += ' Has %s ever %s %s?' % (' or'.join(gap['_entity']['_types']),
                                                    gap['_predicate']['_label'],
                                                    utterance['triple']['_subject']['_label'])

                else:
                    # Checked
                    say += ' Has %s ever %s a %s?' % (utterance['triple']['_subject']['_label'],
                                                      gap['_predicate']['_label'],
                                                      ' or'.join(gap['_entity']['_types']))

        return say

    @staticmethod
    def _phrase_complement_gaps(all_gaps, utterance):
        # type: (dict, dict) -> str

        # random choice between object or subject
        entity_role = random.choice(['subject', 'object'])
        gaps = all_gaps['_subject'] if entity_role == 'subject' else all_gaps['_complement']
        say = None

        if entity_role == 'subject':
            say = random.choice(CURIOSITY)

            if not gaps:
                # Checked
                say += ' What types can %s %s' % (utterance['triple']['_subject']['_label'],
                                                  utterance['triple']['_predicate']['_label'])

            else:
                gap = random.choice(gaps)  # TODO Lenka/Suzanna improve logic here
                if ' in' in gap['_predicate']['_label']:  # ' by' in gap['_predicate']['_label']
                    say += ' Is there a %s %s %s?' % (' or'.join(gap['_entity']['_types']),
                                                      gap['_predicate']['_label'],
                                                      utterance['triple']['_complement']['_label'])
                else:
                    say += ' Has %s %s by a %s?' % (utterance['triple']['_complement']['_label'],
                                                    gap['_predicate']['_label'],
                                                    ' or'.join(gap['_entity']['_types']))

        elif entity_role == 'object':
            say = random.choice(CURIOSITY)

            if not gaps:
                otypes = ' or'.join(utterance['triple']['_complement']['_types']) \
                    if ' or'.join(utterance['triple']['_complement']['_types']) != '' \
                    else 'things'
                stypes = ' or'.join(utterance['triple']['_subject']['_types']) \
                    if ' or '.join(utterance['triple']['_subject']['_types']) != '' \
                    else 'actors'
                say += ' What types of %s like %s do %s usually %s' % (otypes,
                                                                       utterance['triple']['_complement']['_label'],
                                                                       stypes,
                                                                       utterance['triple']['_predicate']['_label'])

            else:
                gap = random.choice(gaps)
                if '#' in ' or'.join(gap['_entity']['_types']):
                    say += ' What is %s %s?' % (utterance['triple']['_complement']['_label'],
                                                gap['_predicate']['_label'])
                elif ' by' in gap['_predicate']['_label']:
                    say += ' Has %s ever %s a %s?' % (utterance['triple']['_complement']['_label'],
                                                      gap['_predicate']['_label'],
                                                      ' or'.join(gap['_entity']['_types']))
                else:
                    say += ' Has a %s ever %s %s?' % (' or'.join(gap['_entity']['_types']),
                                                      gap['_predicate']['_label'],
                                                      utterance['triple']['_complement']['_label'])

        return say

    @staticmethod
    def _phrase_overlaps(all_overlaps, utterance):
        # type: (dict, dict) -> str

        entity_role = random.choice(['subject', 'object'])
        overlaps = all_overlaps['_subject'] if entity_role == 'subject' else all_overlaps['_complement']
        say = None

        if not overlaps:
            say = None

        elif len(overlaps) < 2 and entity_role == 'subject':
            say = random.choice(HAPPY)

            say += ' Did you know that %s also %s %s' % (utterance['triple']['_subject']['_label'],
                                                         utterance['triple']['_predicate']['_label'],
                                                         random.choice(overlaps)['_entity']['_label'])

        elif len(overlaps) < 2 and entity_role == 'object':
            say = random.choice(HAPPY)

            say += ' Did you know that %s also %s %s' % (random.choice(overlaps)['_entity']['_label'],
                                                         utterance['triple']['_predicate']['_label'],
                                                         utterance['triple']['_complement']['_label'])

        elif entity_role == 'subject':
            say = random.choice(HAPPY)
            sample = random.sample(overlaps, 2)

            entity_0 = sample[0]['_entity']['_label']
            entity_1 = sample[1]['_entity']['_label']

            say += ' Now I know %s items that %s %s, like %s and %s' % (len(overlaps),
                                                                        utterance['triple']['_subject']['_label'],
                                                                        utterance['triple']['_predicate']['_label'],
                                                                        entity_0, entity_1)

        elif entity_role == 'object':
            say = random.choice(HAPPY)
            sample = random.sample(overlaps, 2)
            types = ' or '.join(sample[0]['_entity']['_types']) if sample[0]['_entity']['_types'] else 'things'
            say += ' Now I know %s %s that %s %s, like %s and %s' % (len(overlaps), types,
                                                                     utterance['triple']['_predicate']['_label'],
                                                                     utterance['triple']['_complement']['_label'],
                                                                     sample[0]['_entity']['_label'],
                                                                     sample[1]['_entity']['_label'])

        return say

    @staticmethod
    def _phrase_trust(trust):
        # type: (float) -> str

        if trust > 0.75:
            say = random.choice(TRUST)
        else:
            say = random.choice(NO_TRUST)

        return say

    @staticmethod
    def _assign_spo(utterance, item):
        empty = ['', 'unknown', 'none']

        # INITIALIZATION
        predicate = utterance['predicate']['type']

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
                new_ent += self._replace_pronouns(speaker, entity_label=word, role='pos') + ' '

        else:
            new_ent += self._replace_pronouns(speaker, entity_label=entity)

        entity = new_ent
        return entity

    @staticmethod
    def _replace_pronouns(speaker, author=None, entity_label=None, role=None):
        if entity_label is None and author is None:
            return speaker

        if role == 'pos':
            # print('pos', speaker, entity_label)
            if speaker.lower() == entity_label.lower():
                pronoun = 'your'
            elif entity_label.lower() == 'leolani':
                pronoun = 'my'
            else:
                pronoun = entity_label  # third person pos.
            return pronoun

        # Fix author
        elif author is not None and author.lower() not in ['', 'unknown', 'none']:
            if speaker.lower() == author.lower():
                pronoun = 'you'
            elif author.lower() == 'leolani':
                pronoun = 'I'
            else:
                pronoun = author.title()

            return pronoun

        # Entity
        if entity_label is not None and entity_label.lower() not in ['', 'unknown', 'none']:
            if speaker.lower() in [entity_label.lower(), 'speaker'] or entity_label == 'Speaker':
                pronoun = 'you'
            elif entity_label.lower() == 'leolani':
                pronoun = 'I'
            else:
                pronoun = entity_label

            return pronoun
