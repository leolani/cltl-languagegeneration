import random
from typing import Optional

from cltl.commons.language_data.sentences import NEW_KNOWLEDGE, EXISTING_KNOWLEDGE, CONFLICTING_KNOWLEDGE, \
    CURIOSITY, HAPPY
from cltl.commons.triple_helpers import filtered_types_names
from simplenlg.framework import *
from simplenlg.realiser.english import *

from cltl.reply_generation.api import Phraser
from cltl.reply_generation.utils.phraser_utils import replace_pronouns, clean_overlaps

lexicon = Lexicon.getDefaultLexicon()
realiser = Realiser(lexicon)
nlgFactory = NLGFactory(lexicon)


def simple_nlg(subject, predicate, complement=None, modifier=None, negation=False, question=None, tense=None,
               perfect=False):
    p = nlgFactory.createClause()
    p.setSubject(subject)
    predicate = predicate.replace("-", " ")
    verb = nlgFactory.createVerbPhrase(predicate)
    p.setVerb(verb)
    if complement:
        p.setObject(complement)
    if modifier:
        verb.addModifier(modifier)
    p.setFeature(Feature.NEGATED, negation)
    if question:
        p.setFeature(Feature.INTERROGATIVE_TYPE, question)
    if tense:
        p.setFeature(Feature.TENSE, tense)
    p.setFeature(Feature.PERFECT, perfect)

    text = realiser.realiseSentence(p)

    return text


class SimplenlgPhraser(Phraser):

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
        # type: (dict, dict) -> Optional[str]

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
            say += ' %s told me on %s that %s But now you tell me that %s' \
                   % (x, conflict['_provenance']['_date'],
                      simple_nlg(y, utterance['triple']['_predicate']['_label'], conflict['_complement']['_label']),
                      simple_nlg(y, utterance['triple']['_predicate']['_label'],
                                 utterance['triple']['_complement']['_label']))

            return say

    @staticmethod
    def _phrase_negation_conflicts(conflicts, utterance):
        # type: (dict, dict) -> Optional[str]

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

                say += ' %s told me on %s that %s But on %s %s told me that %s' \
                       % (affirmative_conflict['_provenance']['_author']['_label'],
                          affirmative_conflict['_provenance']['_date'],
                          simple_nlg(utterance['triple']['_subject']['_label'],
                                     utterance['triple']['_predicate']['_label'],
                                     utterance['triple']['_complement']['_label']),
                          negative_conflict['_provenance']['_date'],
                          negative_conflict['_provenance']['_author']['_label'],
                          simple_nlg(utterance['triple']['_subject']['_label'],
                                     utterance['triple']['_predicate']['_label'],
                                     utterance['triple']['_complement']['_label'], negation=True))

                return say

    @staticmethod
    def _phrase_statement_novelty(novelties, utterance):
        # type: (dict, dict) -> Optional[str]
        novelties = novelties["provenance"]

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

                # [Lea] Not included in current tests
                say += ' I did not know %s that %s' % (any_type, simple_nlg(utterance['triple']['_subject']['_label'],
                                                                            utterance['triple']['_predicate'][
                                                                                '_label']))

            elif entity_role == 'object':
                # [Lea] Not included in current tests
                say += ' I did not know anybody %s' % (simple_nlg("who", utterance['triple']['_predicate']['_label'],
                                                                  utterance['triple']['_complement']['_label']))

        # I already knew this
        else:
            say = random.choice(EXISTING_KNOWLEDGE)
            novelty = random.choice(novelties)

            # [Lea] Checked
            say += ' %s told me about it on %s.' % (novelty['_provenance']['_author']['_label'],
                                                    novelty['_provenance']['_date'])

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
        # type: (dict, dict) -> Optional[str]

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
            if 'is-' in gap['_predicate']['_label'] or ' is' in gap['_predicate']['_label']:
                say += ' Is there a %s that %s %s?' % (filtered_types_names(gap['_target_entity_type']['_types']),
                                                       gap['_predicate']['_label'],
                                                       gap['_known_entity']['_label'])
            elif 'be-' in gap['_predicate']['_label']:
                # Be-question
                say += ' %s' % (simple_nlg(gap['_known_entity']['_label'],
                                           gap['_predicate']['_label'],
                                           filtered_types_names(gap['_target_entity_type']['_types']),
                                           question=InterrogativeType.YES_NO, tense=Tense.PRESENT, perfect=True))

            elif '-of' in gap['_predicate']['_label']:
                say += ' Is there a %s that %s is %s?' % (filtered_types_names(gap['_target_entity_type']['_types']),
                                                          gap['_known_entity']['_label'],
                                                          gap['_predicate']['_label'])

            elif ' ' in gap['_predicate']['_label']:
                say += ' Is there a %s that is %s %s?' % (filtered_types_names(gap['_target_entity_type']['_types']),
                                                          gap['_predicate']['_label'],
                                                          gap['_known_entity']['_label'])

            else:
                # Verb-question
                say += ' Has %s?' % (simple_nlg(gap['_known_entity']['_label'],
                                                gap['_predicate']['_label'],
                                                filtered_types_names(gap['_target_entity_type']['_types']),
                                                tense=Tense.PAST))

        elif entity_role == 'object':
            if '#' in filtered_types_names(gap['_target_entity_type']['_types']):
                say += ' What is %s %s?' % (gap['_known_entity']['_label'],
                                            gap['_predicate']['_label'])
            elif ' ' in gap['_predicate']['_label']:
                # Checked
                say += ' Has %s ever %s %s?' % (filtered_types_names(gap['_target_entity_type']['_types']),
                                                gap['_predicate']['_label'],
                                                gap['_known_entity']['_label'])
            elif 'be-' in gap['_predicate']['_label']:  # [Lea] TODO NEXT / CHECKPOINT
                # Checked
                say += ' %s ' % (simple_nlg(filtered_types_names(gap['_target_entity_type']['_types']),
                                            gap['_predicate']['_label'],
                                            gap['_known_entity']['_label'],
                                            # modifier="ever",
                                            question=InterrogativeType.YES_NO, tense=Tense.PRESENT, perfect=True))
            else:
                # Checked
                say += ' Has %s ever %s a %s?' % (gap['_known_entity']['_label'],
                                                  gap['_predicate']['_label'],
                                                  filtered_types_names(gap['_target_entity_type']['_types']))

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

        if not gaps:
            return None

        gap = random.choice(gaps)
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

    @staticmethod
    def _phrase_overlaps(all_overlaps, utterance):
        # type: (dict, dict) -> Optional[str]

        if not all_overlaps:
            return None

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
            say += ' Now I know %s items that %s' \
                   ' For example %s and %s.' % (len(overlaps),
                                                simple_nlg(utterance['triple']['_subject']['_label'],
                                                           utterance['triple']['_predicate']['_label']),
                                                sample[0]['_entity']['_label'],
                                                sample[1]['_entity']['_label'])

        elif entity_role == 'object':
            sample = random.sample(overlaps, 2)
            types = filtered_types_names(sample[0]['_entity']['_types']) if sample[0]['_entity']['_types'] else 'things'
            say += ' Now I know %s %s %s ' \
                   'For example %s and %s.' % (len(overlaps), types,
                                               simple_nlg("that", utterance['triple']['_predicate']['_label'],
                                                          utterance['triple']['_complement']['_label']),
                                               sample[0]['_entity']['_label'],
                                               sample[1]['_entity']['_label'])

        return say
