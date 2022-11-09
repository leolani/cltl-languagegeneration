import random
from typing import Optional

from cltl.commons.language_data.sentences import NEW_KNOWLEDGE, EXISTING_KNOWLEDGE, CONFLICTING_KNOWLEDGE, \
    CURIOSITY, HAPPY, TRUST, NO_TRUST
from cltl.commons.triple_helpers import filtered_types_names

from cltl.reply_generation.api import Phraser
from cltl.reply_generation.utils.phraser_utils import replace_pronouns


class PatternPhraser(Phraser):

    def __init__(self):
        # type: () -> None
        """
        Generate natural language based on structured data

        Parameters
        ----------
        """

        super(Phraser, self).__init__()

    def phrase_correct_thought(self, utterance, thought_type, thought_info, fallback=True):
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
            reply = self._phrase_complement_gaps(thought_info)

        elif thought_type == "_subject_gaps":
            reply = self._phrase_subject_gaps(thought_info)

        elif thought_type == "_overlaps":
            reply = self._phrase_overlaps(thought_info, utterance)

        elif thought_type == "_trust":
            reply = self._phrase_trust(thought_info)

        if fallback and reply is None:  # Fallback strategy
            reply = self.phrase_fallback()

        # Formatting
        if reply:
            reply = reply.replace("-", " ").replace("  ", " ")

        return reply

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

    @staticmethod
    def _phrase_type_novelty(novelties, utterance):
        # type: (dict, dict) -> Optional[str]

        # There is no novelty information, so no response
        if not novelties:
            return None

        entity_role = random.choice(['subject', 'object'])
        entity_label = utterance['triple']['_subject']['_label'] \
            if entity_role == 'subject' else utterance['triple']['_complement']['_label']
        novelty = novelties['_subject'] if entity_role == 'subject' else novelties['_complement']

        if novelty:
            entity_label = replace_pronouns(utterance['author']['label'], entity_label=entity_label, role=entity_role)
            say = random.choice(NEW_KNOWLEDGE)
            if entity_label != 'you':  # TODO or type person?
                # Checked
                say += ' I had never heard about %s before!' % replace_pronouns(utterance['author']['label'],
                                                                                entity_label=entity_label,
                                                                                role='object')
            else:
                say += ' I am excited to get to know about %s!' % entity_label

        else:
            say = random.choice(EXISTING_KNOWLEDGE)
            if entity_label != 'you':
                # Checked
                say += ' I have heard about %s before' % replace_pronouns(utterance['author']['label'],
                                                                          entity_label=entity_label,
                                                                          role='object')
            else:
                say += ' I love learning more and more about %s!' % entity_label

        return say

    @staticmethod
    def _phrase_subject_gaps(all_gaps):
        # type: (dict) -> Optional[str]

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
    def _phrase_complement_gaps(all_gaps):
        # type: (dict) -> Optional[str]

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

    @staticmethod
    def phrase_fallback():
        """Phrases a fallback utterance when an error has occurred or no
        thoughts were generated.

        returns: phrase
        """
        # self._log.info(f"Empty response")
        return "I do not know what to say."
