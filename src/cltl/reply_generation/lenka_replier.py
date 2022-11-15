import random

from cltl.commons.casefolding import casefold_capsule
from cltl.commons.language_data.sentences import NO_ANSWER
from cltl.commons.language_helpers import lexicon_lookup
from cltl.commons.triple_helpers import filtered_types_names

from cltl.reply_generation.api import BasicReplier
from cltl.reply_generation.phrasers.pattern_phraser import PatternPhraser
from cltl.reply_generation.thought_selectors.random_selector import RandomSelector
from cltl.reply_generation.utils.phraser_utils import replace_pronouns, assign_spo, deal_with_authors, fix_entity


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
            thought_options = ['_complement_conflict', '_negation_conflicts', '_statement_novelty', '_entity_novelty',
                               '_subject_gaps', '_complement_gaps', '_overlaps', '_trust']
        self._log.debug(f'Thoughts options: {thought_options}')

        # Casefold
        utterance = casefold_capsule(brain_response['statement'], format='natural')
        thoughts = casefold_capsule(brain_response['thoughts'], format='natural')

        # Filter out None thoughts
        thought_options = [option for option in thought_options if thoughts[option] is not None]

        if not thought_options:
            reply = self._phraser.phrase_fallback()
        else:
            # Select thought
            thought_type = self._thought_selector.select(thought_options)
            self._log.info(f"Chosen thought type: {thought_type}")

            # Generate reply
            reply = self._phraser.phrase_correct_thought(utterance, thought_type, thoughts[thought_type])

            # Recursion if there is no answer
            # In theory we do not run into an infinite loop because there will always be a value for novelty
            if persist and reply is None:
                reply = self.reply_to_statement(brain_response, persist=persist, thought_options=thought_options)

        return reply

<<<<<<< HEAD
    @staticmethod
    def _phrase_cardinality_conflicts(conflicts, utterance):
        # type: (list[dict], dict) -> Optional[str]

        # There is no conflict, so no response
        if not conflicts:
            return None

        # There is a conflict, so we phrase it
        else:
            say = random.choice(CONFLICTING_KNOWLEDGE)
            conflict = random.choice(conflicts)
            x = 'you' if conflict['_provenance']['_author']['_label'] == utterance['author']['label'] \
                else conflict['_provenance']['_author']['_label']
            y = 'you' if utterance['triple']['_subject']['_label'] == conflict['_provenance']['_author']['_label'] \
                else utterance['triple']['_subject']['_label']

            # Checked
            say += ' %s told me on %s that %s %s %s, but now you tell me that %s %s %s' \
                   % (x, conflict['_provenance']['_date'], y, utterance['triple']['_predicate']['_label'],
                      conflict['_complement']['_label'],
                      y, utterance['triple']['_predicate']['_label'], utterance['triple']['_complement']['_label'])

            return say

    @staticmethod
    def _phrase_negation_conflicts(conflicts, utterance):
        # type: (list[dict], dict) -> Optional[str]

        # There is no conflict, so no response
        if not conflicts or len(conflicts) < 2:
            return None

        # There is conflict entries
        else:
            affirmative_conflict = [item for item in conflicts if item['_polarity_value'] == 'POSITIVE']
            negative_conflict = [item for item in conflicts if item['_polarity_value'] == 'NEGATIVE']

            # There is a conflict, so we phrase it
            if affirmative_conflict and negative_conflict:
                say = random.choice(CONFLICTING_KNOWLEDGE)

                affirmative_conflict = random.choice(affirmative_conflict)
                negative_conflict = random.choice(negative_conflict)

                say += ' %s told me in %s that %s %s %s, but in %s %s told me that %s did not %s %s' \
                       % (affirmative_conflict['_provenance']['_author']['_label'],
                          affirmative_conflict['_provenance']['_date'],
                          utterance['triple']['_subject']['_label'], utterance['triple']['_predicate']['_label'],
                          utterance['triple']['_complement']['_label'],
                          negative_conflict['_provenance']['_date'],
                          negative_conflict['_provenance']['_author']['_label'],
                          utterance['triple']['_subject']['_label'], utterance['triple']['_predicate']['_label'],
                          utterance['triple']['_complement']['_label'])

                return say

    @staticmethod
    def _phrase_statement_novelty(novelties, utterance):
        # type: (list[dict], dict) -> Optional[str]

        # I do not know this before, so be happy to learn
        if not novelties:
            entity_role = random.choice(['subject', 'object'])

            say = random.choice(NEW_KNOWLEDGE)

            if entity_role == 'subject':
                if 'person' in filtered_types_names(utterance['triple']['_complement']['_types']):
                    any_type = 'anybody'
                elif 'location' in filtered_types_names(utterance['triple']['_complement']['_types']):
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
            say += ' %s told me about it in %s' % (novelty['_provenance']['_author']['_label'],
                                                   novelty['_provenance']['_date'])

        return say

    def _phrase_type_novelty(self, novelties, utterance):
        # type: (dict, dict) -> Optional[str]

        # There is no novelty information, so no response
        if not novelties:
            return None

        entity_role = random.choice(['subject', 'object'])
        entity_label = utterance['triple']['_subject']['_label'] \
            if entity_role == 'subject' else utterance['triple']['_complement']['_label']
        novelty = novelties['_subject'] if entity_role == 'subject' else novelties['_complement']

        if novelty:
            entity_label = self._replace_pronouns(utterance['author']['label'], entity_label=entity_label,
                                                  role=entity_role)
            say = random.choice(NEW_KNOWLEDGE)
            if entity_label != 'you':  # TODO or type person?
                # Checked
                say += ' I had never heard about %s before!' % self._replace_pronouns(utterance['author']['label'],
                                                                                      entity_label=entity_label,
                                                                                      role='object')
            else:
                say += ' I am excited to get to know about %s!' % entity_label

        else:
            say = random.choice(EXISTING_KNOWLEDGE)
            if entity_label != 'you':
                # Checked
                say += ' I have heard about %s before' % self._replace_pronouns(utterance['author']['label'],
                                                                                entity_label=entity_label,
                                                                                role='object')
            else:
                say += ' I love learning more and more about %s!' % entity_label

        return say

    @staticmethod
    def _phrase_subject_gaps(all_gaps, utterance):
        # type: (dict, dict) -> Optional[str]

        # There is no gaps, so no response
        if not all_gaps:
            return None

        # random choice between object or subject
        entity_role = random.choice(['subject', 'object'])
        gaps = all_gaps['_subject'] if entity_role == 'subject' else all_gaps['_complement']
        say = None

        if entity_role == 'subject':
            say = random.choice(CURIOSITY)

            if not gaps:
                pass
            else:
                gap = random.choice(gaps)
                if 'is ' in gap['_predicate']['_label'] or ' is' in gap['_predicate']['_label']:
                    say += ' Is there a %s that %s %s?' % (filtered_types_names(gap['_entity']['_types']),
                                                           gap['_predicate']['_label'],
                                                           gap['_known_entity']['_label'])
                elif ' of' in gap['_predicate']['_label']:
                    say += ' Is there a %s that %s is %s?' % (filtered_types_names(gap['_entity']['_types']),
                                                              gap['_known_entity']['_label'],
                                                              gap['_predicate']['_label'])

                elif ' ' in gap['_predicate']['_label']:
                    say += ' Is there a %s that is %s %s?' % (filtered_types_names(gap['_entity']['_types']),
                                                              gap['_predicate']['_label'],
                                                              gap['_known_entity']['_label'])
                else:
                    # Checked
                    say += ' Has %s %s %s?' % (gap['_known_entity']['_label'],
                                               gap['_predicate']['_label'],
                                               filtered_types_names(gap['_entity']['_types']))

        elif entity_role == 'object':
            say = random.choice(CURIOSITY)

            if not gaps:
                pass
            else:
                gap = random.choice(gaps)
                if '#' in filtered_types_names(gap['_entity']['_types']):
                    say += ' What is %s %s?' % (gap['_known_entity']['_label'],
                                                gap['_predicate']['_label'])
                elif ' ' in gap['_predicate']['_label']:
                    # Checked
                    say += ' Has %s ever %s %s?' % (filtered_types_names(gap['_entity']['_types']),
                                                    gap['_predicate']['_label'],
                                                    gap['_known_entity']['_label'])

                else:
                    # Checked
                    say += ' Has %s ever %s a %s?' % (gap['_known_entity']['_label'],
                                                      gap['_predicate']['_label'],
                                                      filtered_types_names(gap['_entity']['_types']))

        return say

    @staticmethod
    def _phrase_complement_gaps(all_gaps, utterance):
        # type: (dict, dict) -> Optional[str]

        # There is no gaps, so no response
        if not all_gaps:
            return None

        # random choice between object or subject
        entity_role = random.choice(['subject', 'object'])
        gaps = all_gaps['_subject'] if entity_role == 'subject' else all_gaps['_complement']
        say = None

        if entity_role == 'subject':
            say = random.choice(CURIOSITY)

            if not gaps:
                pass
            else:
                gap = random.choice(gaps)  # TODO Lenka/Suzanna improve logic here
                if ' in' in gap['_predicate']['_label']:  # ' by' in gap['_predicate']['_label']
                    say += ' Is there a %s %s %s?' % (filtered_types_names(gap['_entity']['_types']),
                                                      gap['_predicate']['_label'],
                                                      gap['_known_entity']['_label'])
                else:
                    say += ' Has %s %s by a %s?' % (gap['_known_entity']['_label'],
                                                    gap['_predicate']['_label'],
                                                    filtered_types_names(gap['_entity']['_types']))

        elif entity_role == 'object':
            say = random.choice(CURIOSITY)

            if not gaps:
                pass
            else:
                gap = random.choice(gaps)
                if '#' in filtered_types_names(gap['_entity']['_types']):
                    say += ' What is %s %s?' % (gap['_known_entity']['_label'],
                                                gap['_predicate']['_label'])
                elif ' by' in gap['_predicate']['_label']:
                    say += ' Has %s ever %s a %s?' % (gap['_known_entity']['_label'],
                                                      gap['_predicate']['_label'],
                                                      filtered_types_names(gap['_entity']['_types']))
                else:
                    say += ' Has a %s ever %s %s?' % (filtered_types_names(gap['_entity']['_types']),
                                                      gap['_predicate']['_label'],
                                                      gap['_known_entity']['_label'])

        return say

    @staticmethod 
    def _phrase_overlaps(all_overlaps, utterance):
        # type: (dict, dict) -> Optional[str]

        if not all_overlaps:
            pass

        entity_role = random.choice(['subject', 'object'])
        overlaps = all_overlaps['_subject'] if entity_role == 'subject' else all_overlaps['_complement']
        say = None

        if not overlaps:
            pass

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
            # [Lea] add author name = you option
            say += ' Now I know %s items that %s %s, like %s and %s' % (len(overlaps),
                                                                        utterance['triple']['_subject']['_label'],
                                                                        utterance['triple']['_predicate']['_label'],
                                                                        entity_0, entity_1)

        elif entity_role == 'object':
            say = random.choice(HAPPY)
            sample = random.sample(overlaps, 2)
            types = filtered_types_names(sample[0]['_entity']['_types']) if sample[0]['_entity']['_types'] else 'things'
            say += ' Now I know %s %s that %s %s, like %s and %s' % (len(overlaps), types,
                                                                     utterance['triple']['_predicate']['_label'],
                                                                     utterance['triple']['_complement']['_label'],
                                                                     sample[0]['_entity']['_label'],
                                                                     sample[1]['_entity']['_label'])

        return say

    @staticmethod
    def _phrase_trust(trust):
        # type: (float) -> Optional[str]

        if not trust:
            return None

        elif float(trust) > 0.75:
            say = random.choice(TRUST)
        else:
            say = random.choice(NO_TRUST)

        return say

    def _phrase_fallback(self):
        """Phrases a fallback utterance when an error has occurred or no
        thoughts were generated.

        returns: phrase
=======
    def reply_to_mention(self, brain_response, persist=False, thought_options=None):
>>>>>>> 3a02d79840a6895d60fd03d61f3b5cb02a0b7d2b
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

        # Filter out None thoughts
        options = [option for option in thought_options if thoughts[option] is not None]

        if not thought_options:
            reply = self._phraser.phrase_fallback()
        else:
            # Select thought
            thought_type = self._thought_selector.select(options)
            self._log.info(f"Chosen thought type: {thought_type}")

            # Generate reply
            reply = self._phraser.phrase_correct_thought(utterance, thought_type, thoughts[thought_type])

            # Recursion if there is no answer.
            # In theory we do not run into an infinite loop because there will always be a value for novelty
            if persist and reply is None:
                reply = self.reply_to_mention(brain_response, persist=persist, thought_options=thought_options)

        return reply
