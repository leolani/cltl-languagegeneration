import random
from typing import Optional

from cltl.commons.language_data.sentences import NEW_KNOWLEDGE, EXISTING_KNOWLEDGE, CONFLICTING_KNOWLEDGE, \
    CURIOSITY, HAPPY
from cltl.commons.triple_helpers import filtered_types_names
from cltl.thoughts.thought_selection.utils.thought_utils import separate_select_negation_conflicts

from cltl.reply_generation.api import Phraser
from cltl.reply_generation.utils.phraser_utils import replace_pronouns, clean_overlaps


class PatternPhraser(Phraser):

    def __init__(self):
        # type: () -> None
        """
        Generate natural language based on structured data

        Parameters
        ----------
        """

        super(PatternPhraser, self).__init__()

    def phrase_triple(self, utterance):
        # type: (dict) -> Optional[str]
        say = f"{utterance['triple']['_subject']['_label']} " \
              f"{utterance['triple']['_predicate']['_label']} " \
              f"{utterance['triple']['_complement']['_label']}"

        return say

    def _phrase_cardinality_conflicts(self, selected_thought, utterance):
        # type: (dict, dict) -> Optional[str]

        # There is no conflict, so no response
        if not selected_thought or not selected_thought["thought_info"]:
            return None

        # There is a conflict, so we phrase it
        else:
            say = random.choice(CONFLICTING_KNOWLEDGE)
            conflict = selected_thought["thought_info"]

            x = 'you' if conflict['_provenance']['_author']['_label'] == utterance['author']['label'] \
                else conflict['_provenance']['_author']['_label']
            y = 'you' if utterance['triple']['_subject']['_label'] == conflict['_provenance']['_author']['_label'] \
                else utterance['triple']['_subject']['_label']

            # Checked
            say += ' %s told me in %s that %s %s %s, but now you tell me that %s %s %s' \
                   % (x, conflict['_provenance']['_date'],
                      y, utterance['triple']['_predicate']['_label'], conflict['_complement']['_label'],
                      y, utterance['triple']['_predicate']['_label'], utterance['triple']['_complement']['_label'])

            return say

    def _phrase_negation_conflicts(self, selected_thought, utterance):
        # type: (dict, dict) -> Optional[str]

        # There is no conflict, so no response
        if not selected_thought or not selected_thought["thought_info"]:
            return None

        # There is conflict entries
        else:
            conflicts = selected_thought["thought_info"]
            affirmative_conflict, negative_conflict = separate_select_negation_conflicts(conflicts)

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

    def _phrase_statement_novelty(self, selected_thought, utterance):
        # type: (dict, dict) -> Optional[str]

        # I do not know this before, so be happy to learn
        if not selected_thought or not selected_thought["thought_info"]:
            say = random.choice(NEW_KNOWLEDGE)
            entity_role = selected_thought["extra_info"]

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
            novelty = selected_thought["thought_info"]

            # Checked
            say += ' %s  heeft me dat verteld op %s' % (novelty['_provenance']['_author']['_label'],
                                                        novelty['_provenance']['_date'])
            # say += ' %s told me about it in %s' % (novelty['_provenance']['_author']['_label'],
            #                                        novelty['_provenance']['_date'])

        return say

    def _phrase_type_novelty(self, selected_thought, utterance):
        # type: (dict, dict) -> Optional[str]

        entity_role = selected_thought["extra_info"]
        novelty = selected_thought["thought_info"]

        # Only entity thought
        if 'entity' in utterance.keys():
            entity_label = utterance['entity']['_label']
            entity_label = replace_pronouns(utterance['source']['label'] if 'source' in utterance.keys() else 'author',
                                            entity_label=entity_label, role=entity_role)
        # Triple thought
        else:
            entity_label = utterance['triple']['_subject']['_label'] if entity_role == 'subject' \
                else utterance['triple']['_complement']['_label']
            entity_label = replace_pronouns(utterance['author']['label'], entity_label=entity_label, role=entity_role)

        # There is no novelty information, so happy to learn
        if novelty:
            say = random.choice(NEW_KNOWLEDGE)
            say += ' I had never heard about %s before!' % entity_label

        # I already knew this
        else:
            say = random.choice(EXISTING_KNOWLEDGE)
            say += ' I have heard about %s before' % entity_label

        return say

    def _phrase_subject_gaps(self, selected_thought, utterance):
        # type: (dict, dict) -> Optional[str]

        # There is no gaps, so no response
        if not selected_thought or not selected_thought["thought_info"]:
            return None

        # There is a gap
        entity_role = selected_thought["extra_info"]
        gap = selected_thought["thought_info"]

        say = random.choice(CURIOSITY)
        if entity_role == 'subject':
            if 'is ' in gap['_predicate']['_label'] or ' is' in gap['_predicate']['_label']:
                say += ' Is there a %s that %s %s?' % (filtered_types_names(gap['_target_entity_type']['_types']),
                                                       gap['_predicate']['_label'],
                                                       gap['_known_entity']['_label'])
            elif ' of' in gap['_predicate']['_label']:
                say += ' Is there a %s that %s is %s?' % (filtered_types_names(gap['_target_entity_type']['_types']),
                                                          gap['_known_entity']['_label'],
                                                          gap['_predicate']['_label'])
            elif ' ' in gap['_predicate']['_label']:
                say += ' Is there a %s that is %s %s?' % (filtered_types_names(gap['_target_entity_type']['_types']),
                                                          gap['_predicate']['_label'],
                                                          gap['_known_entity']['_label'])
            else:
                # Checked
                say += ' Has %s %s %s?' % (gap['_known_entity']['_label'],
                                           gap['_predicate']['_label'],
                                           filtered_types_names(gap['_target_entity_type']['_types']))

        elif entity_role == 'object':
            if '#' in filtered_types_names(gap['_target_entity_type']['_types']):
                say += ' What is %s %s?' % (gap['_known_entity']['_label'],
                                            gap['_predicate']['_label'])
            elif ' ' in gap['_predicate']['_label']:
                # Checked
                say += ' Has %s ever %s %s?' % (filtered_types_names(gap['_target_entity_type']['_types']),
                                                gap['_predicate']['_label'],
                                                gap['_known_target_entity_type']['_label'])
            else:
                # Checked
                say += ' Has %s ever %s a %s?' % (gap['_known_entity']['_label'],
                                                  gap['_predicate']['_label'],
                                                  filtered_types_names(gap['_target_entity_type']['_types']))

        return say

    def _phrase_complement_gaps(self, selected_thought, utterance):
        # type: (dict, dict) -> Optional[str]

        # There is no gaps, so no response
        if not selected_thought or not selected_thought["thought_info"]:
            return None

        # There is a gap
        entity_role = selected_thought["extra_info"]
        gap = selected_thought["thought_info"]

        say = random.choice(CURIOSITY)
        if entity_role == 'subject':
            if ' in' in gap['_predicate']['_label']:  # ' by' in gap['_predicate']['_label']
                say += ' Is there a %s %s %s?' % (filtered_types_names(gap['_target_entity_type']['_types']),
                                                  gap['_predicate']['_label'],
                                                  gap['_known_entity']['_label'])
            else:
                say += ' Has %s %s by a %s?' % (gap['_known_entity']['_label'],
                                                gap['_predicate']['_label'],
                                                filtered_types_names(gap['_target_entity_type']['_types']))

        elif entity_role == 'object':
            if '#' in filtered_types_names(gap['_target_entity_type']['_types']):
                say += ' What is %s %s?' % (gap['_known_entity']['_label'],
                                            gap['_predicate']['_label'])
            elif ' by' in gap['_predicate']['_label']:
                say += ' Has %s ever %s a %s?' % (gap['_known_entity']['_label'],
                                                  gap['_predicate']['_label'],
                                                  filtered_types_names(gap['_target_entity_type']['_types']))
            else:
                say += ' Has a %s ever %s %s?' % (filtered_types_names(gap['_target_entity_type']['_types']),
                                                  gap['_predicate']['_label'],
                                                  gap['_known_entity']['_label'])

        return say

    def _phrase_overlaps(self, selected_thought, utterance):
        # type: (dict, dict) -> Optional[str]

        if not selected_thought or not selected_thought["thought_info"]:
            return None

        entity_role = selected_thought["extra_info"]
        overlap = selected_thought["thought_info"]

        say = random.choice(HAPPY)
        if entity_role == 'subject':
            say += ' Did you know that %s also %s %s' % (utterance['triple']['_subject']['_label'],
                                                         utterance['triple']['_predicate']['_label'],
                                                         overlap['_entity']['_label'])

        elif entity_role == 'object':
            say += ' Did you know that %s also %s %s' % (overlap['_entity']['_label'],
                                                         utterance['triple']['_predicate']['_label'],
                                                         utterance['triple']['_complement']['_label'])

        return say
