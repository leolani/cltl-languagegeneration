import random
from typing import Optional

from cltl.commons.language_data.sentences import NEW_KNOWLEDGE, EXISTING_KNOWLEDGE, CONFLICTING_KNOWLEDGE, \
    CURIOSITY, HAPPY
from cltl.commons.triple_helpers import filtered_types_names

from cltl.reply_generation.api import Phraser
from cltl.reply_generation.utils.phraser_utils import replace_pronouns
from cltl.reply_generation.utils.thought_utils import clean_overlaps


class PatternPhraser(Phraser):

    def __init__(self):
        # type: () -> None
        """
        Generate natural language based on structured data

        Parameters
        ----------
        """

        super(Phraser, self).__init__()

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
            say += ' %s told me in %s that %s %s %s, but now you tell me that %s %s %s' \
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
                # say += ' I did not know %s that %s %s' % (any_type, utterance['triple']['_subject']['_label'],
                #                                           utterance['triple']['_predicate']['_label'])

                say += ' Ik wist niet %s dat %s %s' % (any_type, utterance['triple']['_subject']['_label'],
                                                          utterance['triple']['_predicate']['_label'])

            elif entity_role == 'object':
                # Checked
                # say += ' I did not know anybody who %s %s' % (utterance['triple']['_predicate']['_label'],
                #                                               utterance['triple']['_complement']['_label'])

                say += ' Ik wist niet dat iemand %s %s' % (utterance['triple']['_predicate']['_label'],
                                                              utterance['triple']['_complement']['_label'])
        # I already knew this
        else:
            say = random.choice(EXISTING_KNOWLEDGE)
            novelty = random.choice(novelties)

            # Checked
            say += ' %s  heeft me dat verteld op %s' % (novelty['_provenance']['_author']['_label'],
                                                   novelty['_provenance']['_date'])
            # say += ' %s told me about it in %s' % (novelty['_provenance']['_author']['_label'],
            #                                        novelty['_provenance']['_date'])

        return say

    @staticmethod
    def _phrase_type_novelty(novelties, utterance):
        # type: (dict, dict) -> Optional[str]

        # There is no novelty information, so no response
        if not novelties:
            return None

        entity_role = random.choice(['subject', 'object'])
        novelty = novelties['_subject'] if entity_role == 'subject' else novelties['_complement']

        if 'entity' in utterance.keys():
            entity_label = utterance['entity']['_label']
            entity_label = replace_pronouns(utterance['source']['label'] if 'source' in utterance.keys() else 'author',
                                            entity_label=entity_label, role=entity_role)
        else:
            entity_label = utterance['triple']['_subject']['_label'] if entity_role == 'subject' \
                else utterance['triple']['_complement']['_label']
            entity_label = replace_pronouns(utterance['author']['label'], entity_label=entity_label, role=entity_role)

        if novelty:
            say = random.choice(NEW_KNOWLEDGE)
            say += ' I had never heard about %s before!' % entity_label

        else:
            say = random.choice(EXISTING_KNOWLEDGE)
            say += ' I have heard about %s before' % entity_label

        return say

    @staticmethod
    def _phrase_subject_gaps(all_gaps, utterance):
        # type: (dict) -> Optional[str]

        # There is no gaps, so no response
        if not all_gaps:
            return None

        # random choice between object or subject
        entity_role = random.choice(['subject', 'object'])
        gaps = all_gaps['_subject'] if entity_role == 'subject' else all_gaps['_complement']

        if not gaps:
            return None

        gap = random.choice(gaps)
        say = random.choice(CURIOSITY)

        if entity_role == 'subject':
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
        # type: (dict) -> Optional[str]

        # There is no gaps, so no response
        if not all_gaps:
            return None

        # random choice between object or subject
        entity_role = random.choice(['subject', 'object'])
        gaps = all_gaps['_subject'] if entity_role == 'subject' else all_gaps['_complement']

        if not gaps:
            return None

        gap = random.choice(gaps)
        say = random.choice(CURIOSITY)

        if entity_role == 'subject':
            if ' in' in gap['_predicate']['_label']:  # ' by' in gap['_predicate']['_label']
                say += ' Is there a %s %s %s?' % (filtered_types_names(gap['_entity']['_types']),
                                                  gap['_predicate']['_label'],
                                                  gap['_known_entity']['_label'])
            else:
                say += ' Has %s %s by a %s?' % (gap['_known_entity']['_label'],
                                                gap['_predicate']['_label'],
                                                filtered_types_names(gap['_entity']['_types']))

        elif entity_role == 'object':
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

        if not overlaps:
            return None

        overlaps = clean_overlaps(overlaps)
        say = random.choice(HAPPY)
        if len(overlaps) < 2 and entity_role == 'subject':
            say += ' Did you know that %s also %s %s' % (utterance['triple']['_subject']['_label'],
                                                         utterance['triple']['_predicate']['_label'],
                                                         random.choice(overlaps)['_entity']['_label'])

        elif len(overlaps) < 2 and entity_role == 'object':
            say += ' Did you know that %s also %s %s' % (random.choice(overlaps)['_entity']['_label'],
                                                         utterance['triple']['_predicate']['_label'],
                                                         utterance['triple']['_complement']['_label'])

        elif entity_role == 'subject':
            sample = random.sample(overlaps, 2)
            say += ' Now I know %s items that %s %s, like %s and %s' % (len(overlaps),
                                                                        utterance['triple']['_subject']['_label'],
                                                                        utterance['triple']['_predicate']['_label'],
                                                                        sample[0]['_entity']['_label'],
                                                                        sample[1]['_entity']['_label'])

        elif entity_role == 'object':
            sample = random.sample(overlaps, 2)
            types = filtered_types_names(sample[0]['_entity']['_types']) if sample[0]['_entity']['_types'] else 'things'
            say += ' Now I know %s %s that %s %s, like %s and %s' % (len(overlaps), types,
                                                                     utterance['triple']['_predicate']['_label'],
                                                                     utterance['triple']['_complement']['_label'],
                                                                     sample[0]['_entity']['_label'],
                                                                     sample[1]['_entity']['_label'])

        return say
