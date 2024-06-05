import random
from typing import Optional

from cltl.commons.language_data.sentences import NEW_KNOWLEDGE, EXISTING_KNOWLEDGE, CONFLICTING_KNOWLEDGE, \
    CURIOSITY, HAPPY, TRUST, NO_TRUST
from cltl.commons.triple_helpers import filtered_types_names
from cltl.thoughts.thought_selection.utils.thought_utils import separate_select_negation_conflicts
from simplenlg.framework import *
from simplenlg.realiser.english import *

from cltl.reply_generation.api import Phraser
from cltl.reply_generation.utils.phraser_utils import replace_pronouns, any_type

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


def phrase_ssubject_gap_question(gap):
    if 'is-' in gap['_predicate']['_label'] or ' is' in gap['_predicate']['_label']:
        say = ' Is there a %s that %s %s?' % (filtered_types_names(gap['_target_entity_type']['_types']),
                                              gap['_predicate']['_label'],
                                              gap['_known_entity']['_label'])
    elif 'be-' in gap['_predicate']['_label']:
        # Be-question
        say = ' %s' % (simple_nlg(gap['_known_entity']['_label'],
                                  gap['_predicate']['_label'],
                                  filtered_types_names(gap['_target_entity_type']['_types']),
                                  question=InterrogativeType.YES_NO, tense=Tense.PRESENT, perfect=True))

    elif '-of' in gap['_predicate']['_label']:
        say = ' Is there a %s that %s is %s?' % (filtered_types_names(gap['_target_entity_type']['_types']),
                                                 gap['_known_entity']['_label'],
                                                 gap['_predicate']['_label'])

    elif ' ' in gap['_predicate']['_label']:
        say = ' Is there a %s that is %s %s?' % (filtered_types_names(gap['_target_entity_type']['_types']),
                                                 gap['_predicate']['_label'],
                                                 gap['_known_entity']['_label'])

    else:
        # Verb-question
        say = ' Has %s?' % (simple_nlg(gap['_known_entity']['_label'],
                                       gap['_predicate']['_label'],
                                       filtered_types_names(gap['_target_entity_type']['_types']),
                                       tense=Tense.PAST))

    return say


def phrase_scomplement_gap_question(gap):
    if '#' in filtered_types_names(gap['_target_entity_type']['_types']):
        say = ' What is %s %s?' % (gap['_known_entity']['_label'],
                                   gap['_predicate']['_label'])
    elif ' ' in gap['_predicate']['_label']:
        # Checked
        say = ' Has %s ever %s %s?' % (filtered_types_names(gap['_target_entity_type']['_types']),
                                       gap['_predicate']['_label'],
                                       gap['_known_entity']['_label'])
    elif 'be-' in gap['_predicate']['_label']:  # [Lea] TODO NEXT / CHECKPOINT
        # Checked
        say = ' %s ' % (simple_nlg(filtered_types_names(gap['_target_entity_type']['_types']),
                                   gap['_predicate']['_label'],
                                   gap['_known_entity']['_label'],
                                   # modifier="ever",
                                   question=InterrogativeType.YES_NO, tense=Tense.PRESENT, perfect=True))
    else:
        # Checked
        say = ' Has %s ever %s a %s?' % (gap['_known_entity']['_label'],
                                         gap['_predicate']['_label'],
                                         filtered_types_names(gap['_target_entity_type']['_types']))

    return say


def phrase_csubject_gap_question(gap):
    if ' in' in gap['_predicate']['_label']:  # ' by' in gap['_predicate']['_label']
        say = ' Is there a %s %s %s?' % (filtered_types_names(gap['_target_entity_type']['_types']),
                                         gap['_predicate']['_label'],
                                         gap['_known_entity']['_label'])
    else:
        say = ' Has %s %s by a %s?' % (gap['_known_entity']['_label'],
                                       gap['_predicate']['_label'],
                                       filtered_types_names(gap['_target_entity_type']['_types']))
    return say


def phrase_ccomplement_gap_question(gap):
    if '#' in filtered_types_names(gap['_target_entity_type']['_types']):
        say = ' What is %s %s?' % (gap['_known_entity']['_label'],
                                   gap['_predicate']['_label'])
    elif ' by' in gap['_predicate']['_label']:
        say = ' Has %s ever %s a %s?' % (gap['_known_entity']['_label'],
                                         gap['_predicate']['_label'],
                                         filtered_types_names(gap['_target_entity_type']['_types']))
    else:
        say = ' Has a %s ever %s %s?' % (filtered_types_names(gap['_target_entity_type']['_types']),
                                         gap['_predicate']['_label'],
                                         gap['_known_entity']['_label'])
    return say


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
    def _phrase_cardinality_conflicts(selected_thought, utterance):
        # type: (dict, dict) -> Optional[str]

        # There is no conflict, so no response
        if not selected_thought or not selected_thought["thought_info"]:
            return None

        # There is a conflict, so we phrase it
        else:
            conflict = selected_thought["thought_info"]

            say = random.choice(CONFLICTING_KNOWLEDGE)
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
    def _phrase_negation_conflicts(selected_thought, utterance):
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
    def _phrase_statement_novelty(selected_thought, utterance):
        # type: (dict, dict) -> Optional[str]

        # I do not know this before, so be happy to learn
        if not selected_thought or not selected_thought["thought_info"]:
            say = random.choice(NEW_KNOWLEDGE)
            entity_role = selected_thought["extra_info"]

            if entity_role == '_subject':
                # [Lea] Not included in current tests
                typ = any_type(utterance)
                say += ' I did not know %s that %s' % (typ, simple_nlg(utterance['triple']['_subject']['_label'],
                                                                       utterance['triple']['_predicate']['_label']))

            elif entity_role == '_complement':
                # [Lea] Not included in current tests
                say += ' I did not know anybody %s' % (simple_nlg("who", utterance['triple']['_predicate']['_label'],
                                                                  utterance['triple']['_complement']['_label']))

        # I already knew this
        else:
            say = random.choice(EXISTING_KNOWLEDGE)
            novelty = selected_thought["thought_info"]

            # [Lea] Checked
            say += ' %s told me about it on %s.' % (novelty['_provenance']['_author']['_label'],
                                                    novelty['_provenance']['_date'])

        return say

    @staticmethod
    def _phrase_type_novelty(selected_thought, utterance):
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
            entity_label = novelty['entity']['_label']
            entity_label = replace_pronouns(utterance['author']['label'], entity_label=entity_label, role=entity_role)

        # There is no novelty information, so happy to learn
        if not selected_thought or not selected_thought["thought_info"] or selected_thought["thought_info"]["value"]:
            say = random.choice(NEW_KNOWLEDGE)
            say += ' I had never heard about %s before!' % entity_label

        # I already knew this
        else:
            say = random.choice(EXISTING_KNOWLEDGE)
            say += ' I have heard about %s before' % entity_label

        return say

    @staticmethod
    def _phrase_subject_gaps(selected_thought, utterance):
        # type: (dict, dict) -> Optional[str]

        # There is no gaps, so no response
        if not selected_thought or not selected_thought["thought_info"]:
            return None

        # There is a gap
        else:
            entity_role = selected_thought["extra_info"]
            gap = selected_thought["thought_info"]

            say = random.choice(CURIOSITY)
            if entity_role == '_subject':
                say += phrase_ssubject_gap_question(gap)

            elif entity_role == '_complement':
                say += phrase_scomplement_gap_question(gap)
            return say

    @staticmethod
    def _phrase_complement_gaps(selected_thought, utterance):
        # type: (dict, dict) -> Optional[str]

        # There is no gaps, so no response
        if not selected_thought or not selected_thought["thought_info"]:
            return None

        # There is a gap
        else:
            entity_role = selected_thought["extra_info"]
            gap = selected_thought["thought_info"]

            say = random.choice(CURIOSITY)
            if entity_role == '_subject':
                say += phrase_csubject_gap_question(gap)

            elif entity_role == '_complement':
                say += phrase_ccomplement_gap_question(gap)

        return say

    @staticmethod
    def _phrase_overlaps(selected_thought, utterance):
        # type: (dict, dict) -> Optional[str]
        if not selected_thought or not selected_thought["thought_info"]:
            return None

        entity_role = selected_thought["extra_info"]
        overlap = selected_thought["thought_info"]

        say = random.choice(HAPPY)
        if entity_role == '_subject':
            say += ' Did you know that %s also %s %s' % (utterance['triple']['_subject']['_label'],
                                                         utterance['triple']['_predicate']['_label'],
                                                         overlap['_entity']['_label'])

        elif entity_role == '_complement':
            say += ' Did you know that %s also %s %s' % (overlap['_entity']['_label'],
                                                         utterance['triple']['_predicate']['_label'],
                                                         utterance['triple']['_complement']['_label'])

        return say

    @staticmethod
    def _phrase_trust(selected_thought: dict, utterance: dict) -> Optional[str]:
        if not selected_thought or not selected_thought["thought_info"]:
            return None

        else:
            trust = selected_thought["thought_info"]['value']

            if float(trust) > 0.25:
                say = random.choice(TRUST)
            else:
                say = random.choice(NO_TRUST)

            say = f"{utterance['author']['label']}, {say}"

            return say
