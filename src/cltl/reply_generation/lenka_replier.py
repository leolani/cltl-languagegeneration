import random

from cltl.combot.backend.utils.casefolding import casefold_text
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

        # TODO revise by Lea (we conjugate the predicate by doing this)
        utterance.casefold(format='natural')

        if not response:
            if utterance.triple.subject.types and utterance.triple.complement.types and utterance.triple.predicate_name:
                say += "I know %s usually %s %s, but I do not know this case" % (
                    random.choice(utterance.triple.subject.types), str(utterance.triple.predicate_name),
                    random.choice(utterance.triple.complement.types))
                return say

            else:
                return None

        # Each triple is hashed, so we can figure out when we are about the say things double
        handled_items = set()
        response.sort(key=lambda x: x['authorlabel']['value'])

        for item in response:

            # INITIALIZATION
            subject, predicate, object = self._assign_spo(utterance, item)

            author = self._replace_pronouns(utterance.chat_speaker, author=item['authorlabel']['value'])
            subject = self._replace_pronouns(utterance.chat_speaker, entity_label=subject, role='subject')
            object = self._replace_pronouns(utterance.chat_speaker, entity_label=object, role='object')

            subject = self._fix_entity(subject, utterance.chat_speaker)
            object = self._fix_entity(object, utterance.chat_speaker)

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
                if utterance.triple.complement_name.lower() == utterance.chat_speaker.lower() or \
                        utterance.triple.subject_name.lower() == utterance.chat_speaker.lower():
                    say += ' your '
                elif utterance.triple.complement_name.lower() == 'leolani' or \
                        utterance.triple.subject_name.lower() == 'leolani':
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
                elif gram_person == 'third' and not '-' in predicate:
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

    def reply_to_statement(self, update, entity_only=False, proactive=True, persist=False):
        """
        Phrase a random thought
        Parameters
        ----------
        update
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
        utterance = update['statement']
        if utterance.triple is None:
            return None

        utterance.casefold(format='natural')
        thoughts = update['thoughts']
        thoughts.casefold(format='natural')
        approach = random.choice(options)
        say = None

        if approach == 'cardinality_conflicts':
            say = self._phrase_cardinality_conflicts(thoughts.complement_conflicts(), utterance)

        elif approach == 'negation_conflicts':
            say = self._phrase_negation_conflicts(thoughts.negation_conflicts(), utterance)

        elif approach == 'statement_novelty':
            say = self._phrase_statement_novelty(thoughts.statement_novelties(), utterance)

        elif approach == 'entity_novelty':
            say = self._phrase_type_novelty(thoughts.entity_novelty(), utterance)

        elif approach == 'subject_gaps':
            say = self._phrase_subject_gaps(thoughts.subject_gaps(), utterance)

        elif approach == 'object_gaps':
            say = self._phrase_complement_gaps(thoughts.complement_gaps(), utterance)

        elif approach == 'overlaps':
            say = self._phrase_overlaps(thoughts.overlaps(), utterance)

        if persist and say is None:
            say = self.reply_to_statement(update, proactive, persist)

        if say and '-' in say:
            say = say.replace('-', ' ').replace('  ', ' ')

        return say

    def phrase_all_conflicts(self, conflicts, speaker=None):
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

    def _phrase_cardinality_conflicts(self, conflicts, utterance):
        # type: (List[CardinalityConflict], Utterance) -> str

        # There is no conflict, so nothing
        if not conflicts:
            say = None

        # There is a conflict, so we phrase it
        else:
            say = random.choice(CONFLICTING_KNOWLEDGE)
            conflict = random.choice(conflicts)
            x = 'you' if conflict.author == utterance.chat_speaker else conflict.author
            y = 'you' if utterance.triple.subject_name == conflict.author else utterance.triple.subject_name

            # Checked
            say += ' %s told me in %s that %s %s %s, but now you tell me that %s %s %s' \
                   % (x, conflict.date.strftime("%B"), y, utterance.triple.predicate_name, conflict.complement_name,
                      y, utterance.triple.predicate_name, utterance.triple.complement_name)

        return say

    def _phrase_negation_conflicts(self, conflicts, utterance):
        # type: (List[NegationConflict], Utterance) -> str

        say = None

        # There is conflict entries
        if conflicts and conflicts[0]:
            affirmative_conflict = [item for item in conflicts if item.polarity_value == 'POSITIVE']
            negative_conflict = [item for item in conflicts if item.polarity_value == 'NEGATIVE']

            # There is a conflict, so we phrase it
            if affirmative_conflict and negative_conflict:
                say = random.choice(CONFLICTING_KNOWLEDGE)

                affirmative_conflict = random.choice(affirmative_conflict)
                negative_conflict = random.choice(negative_conflict)

                say += ' %s told me in %s that %s %s %s, but in %s %s told me that %s did not %s %s' \
                       % (affirmative_conflict.author, affirmative_conflict.date.strftime("%B"),
                          utterance.triple.subject_name, utterance.triple.predicate_name,
                          utterance.triple.complement_name,
                          negative_conflict.date.strftime("%B"), negative_conflict.author,
                          utterance.triple.subject_name, utterance.triple.predicate_name,
                          utterance.triple.complement_name)

        return say

    def _phrase_statement_novelty(self, novelties, utterance):
        # type: (List[StatementNovelty], Utterance) -> str

        # I do not know this before, so be happy to learn
        if not novelties:
            entity_role = random.choice(['subject', 'object'])

            say = random.choice(NEW_KNOWLEDGE)

            if entity_role == 'subject':
                if 'person' in utterance.triple.complement.types:
                    any_type = 'anybody'
                elif 'location' in utterance.triple.complement.types:
                    any_type = 'anywhere'
                else:
                    any_type = 'anything'

                # Checked
                say += ' I did not know %s that %s %s' % (any_type, utterance.triple.subject_name,
                                                          utterance.triple.predicate_name)

            elif entity_role == 'object':
                # Checked
                say += ' I did not know anybody who %s %s' % (utterance.triple.predicate_name,
                                                              utterance.triple.complement_name)

        # I already knew this
        else:
            say = random.choice(EXISTING_KNOWLEDGE)
            novelty = random.choice(novelties)

            # Checked
            say += ' %s told me about it in %s' % (novelty.author, novelty.date.strftime("%B"))

        return say

    def _phrase_type_novelty(self, novelties, utterance):
        # type: (EntityNovelty, Utterance) -> str

        entity_role = random.choice(['subject', 'object'])
        entity_label = utterance.triple.subject_name if entity_role == 'subject' else utterance.triple.complement_name
        novelty = novelties.subject if entity_role == 'subject' else novelties.complement

        if novelty:
            entity_label = self._replace_pronouns(utterance.chat_speaker, entity_label=entity_label, role=entity_role)
            say = random.choice(NEW_KNOWLEDGE)
            if entity_label != 'you':  # TODO or type person?
                # Checked
                say += ' I had never heard about %s before!' % self._replace_pronouns(utterance.chat_speaker,
                                                                                      entity_label=entity_label,
                                                                                      role='object')
            else:
                say += ' I am excited to get to know about %s!' % entity_label

        else:
            say = random.choice(EXISTING_KNOWLEDGE)
            if entity_label != 'you':
                # Checked
                say += ' I have heard about %s before' % self._replace_pronouns(utterance.chat_speaker,
                                                                                entity_label=entity_label,
                                                                                role='object')
            else:
                say += ' I love learning more and more about %s!' % entity_label

        return say

    def _phrase_subject_gaps(self, all_gaps, utterance):
        # type: (Gaps, Utterance) -> str

        entity_role = random.choice(['subject', 'object'])
        gaps = all_gaps.subject if entity_role == 'subject' else all_gaps.complement
        say = None

        if entity_role == 'subject':
            say = random.choice(CURIOSITY)

            if not gaps:
                say += ' What types can %s %s' % (utterance.triple.subject_name, utterance.triple.predicate_name)

            else:
                gap = random.choice(gaps)
                if 'is ' in gap.predicate_name or ' is' in gap.predicate_name:
                    say += ' Is there a %s that %s %s?' % (
                        gap.entity_range_name, gap.predicate_name, utterance.triple.subject_name)
                elif ' of' in gap.predicate_name:
                    say += ' Is there a %s that %s is %s?' % (
                        gap.entity_range_name, utterance.triple.subject_name, gap.predicate_name)

                elif ' ' in gap.predicate_name:
                    say += ' Is there a %s that is %s %s?' % (
                        gap.entity_range_name, gap.predicate_name, utterance.triple.subject_name)
                else:
                    # Checked
                    say += ' Has %s %s %s?' % (utterance.triple.subject_name, gap.predicate_name, gap.entity_range_name)

        elif entity_role == 'object':
            say = random.choice(CURIOSITY)

            if not gaps:
                say += ' What types can %s a %s like %s' % (utterance.triple.predicate_name,
                                                            utterance.triple.complement_name,
                                                            utterance.triple.complement_name)

            else:
                gap = random.choice(gaps)
                if '#' in gap.entity_range_name:
                    say += ' What is %s %s?' % (utterance.triple.subject_name, gap.predicate_name)
                elif ' ' in gap.predicate_name:
                    # Checked
                    say += ' Has %s ever %s %s?' % (
                        gap.entity_range_name, gap.predicate_name, utterance.triple.subject_name)

                else:
                    # Checked
                    say += ' Has %s ever %s a %s?' % (utterance.triple.subject_name, gap.predicate_name,
                                                      gap.entity_range_name)

        return say

    def _phrase_complement_gaps(self, all_gaps, utterance):
        # type: (Gaps, Utterance) -> str

        # random choice between object or subject
        entity_role = random.choice(['subject', 'object'])
        gaps = all_gaps.subject if entity_role == 'subject' else all_gaps.complement
        say = None

        if entity_role == 'subject':
            say = random.choice(CURIOSITY)

            if not gaps:
                # Checked
                say += ' What types can %s %s' % (utterance.triple.subject_name, utterance.triple.predicate_name)

            else:
                gap = random.choice(gaps)  # TODO Lenka/Suzanna improve logic here
                if ' in' in gap.predicate_name:  # ' by' in gap.predicate_name
                    say += ' Is there a %s %s %s?' % (
                        gap.entity_range_name, gap.predicate_name, utterance.triple.complement_name)
                else:
                    say += ' Has %s %s by a %s?' % (utterance.triple.complement_name,
                                                    gap.predicate_name,
                                                    gap.entity_range_name)

        elif entity_role == 'object':
            say = random.choice(CURIOSITY)

            if not gaps:
                otypes = utterance.triple.complement.types_names if utterance.triple.complement.types_names != '' \
                    else 'things'
                stypes = utterance.triple.subject.types_names if utterance.triple.subject.types_names != '' else 'actors'
                say += ' What types of %s like %s do %s usually %s' % (otypes, utterance.triple.complement_name, stypes,
                                                                       utterance.triple.predicate_name)

            else:
                gap = random.choice(gaps)
                if '#' in gap.entity_range_name:
                    say += ' What is %s %s?' % (utterance.triple.complement_name, gap.predicate_name)
                elif ' by' in gap.predicate_name:
                    say += ' Has %s ever %s a %s?' % (
                        utterance.triple.complement_name, gap.predicate_name, gap.entity_range_name)
                else:
                    say += ' Has a %s ever %s %s?' % (
                        gap.entity_range_name, gap.predicate_name, utterance.triple.complement_name)

        return say

    def _phrase_overlaps(self, all_overlaps, utterance):
        # type: (Overlaps, Utterance) -> str

        entity_role = random.choice(['subject', 'object'])
        overlaps = all_overlaps.subject if entity_role == 'subject' else all_overlaps.complement
        say = None

        if not overlaps:
            say = None

        elif len(overlaps) < 2 and entity_role == 'subject':
            say = random.choice(HAPPY)

            say += ' Did you know that %s also %s %s' % (utterance.triple.subject_name, utterance.triple.predicate_name,
                                                         random.choice(overlaps).entity_name)

        elif len(overlaps) < 2 and entity_role == 'object':
            say = random.choice(HAPPY)

            say += ' Did you know that %s also %s %s' % (random.choice(overlaps).entity_name,
                                                         utterance.triple.predicate_name,
                                                         utterance.triple.complement_name)

        elif entity_role == 'subject':
            say = random.choice(HAPPY)
            sample = random.sample(overlaps, 2)

            entity_0 = list(filter(str.isalpha, str(sample[0].entity_name)))
            entity_1 = list(filter(str.isalpha, str(sample[1].entity_name)))

            say += ' Now I know %s items that %s %s, like %s and %s' % (len(overlaps), utterance.triple.subject_name,
                                                                        utterance.triple.predicate_name,
                                                                        entity_0, entity_1)

        elif entity_role == 'object':
            say = random.choice(HAPPY)
            sample = random.sample(overlaps, 2)
            types = sample[0].entity_types[0] if sample[0].entity_types else 'things'
            say += ' Now I know %s %s that %s %s, like %s and %s' % (len(overlaps), types,
                                                                     utterance.triple.predicate_name,
                                                                     utterance.triple.complement_name,
                                                                     sample[0].entity_name, sample[1].entity_name)

        return say

    def _phrase_trust(self, trust):
        # type: (float) -> str

        if trust > 0.75:
            say = random.choice(TRUST)
        else:
            say = random.choice(NO_TRUST)

        return say

    def _assign_spo(self, utterance, item):
        # INITIALIZATION
        predicate = utterance.triple.predicate_name

        if utterance.triple.subject_name != '':
            subject = utterance.triple.subject_name
        else:
            subject = item['slabel']['value']

        if utterance.triple.complement_name != '':
            object = utterance.triple.complement_name
        elif 'olabel' in item:
            object = item['olabel']['value']
        else:
            object = ''

        return subject, predicate, object

    def _deal_with_authors(self, author, previous_author, predicate, previous_predicate, say):
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

    def _replace_pronouns(self, speaker, author=None, entity_label=None, role=None):
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
        elif author is not None:
            if speaker.lower() == author.lower() or author.lower() not in ['', 'unknown', 'none', 'person']:
                pronoun = 'you'
            elif author.lower() == 'leolani':
                pronoun = 'I'
            else:
                pronoun = author.title()

            return pronoun

        # Entity
        if entity_label is not None:
            if speaker.lower() in [entity_label.lower(), 'speaker'] or entity_label == 'Speaker':
                pronoun = 'you'
            elif entity_label.lower() == 'leolani':
                pronoun = 'I'
                '''
            elif entity_label.lower() in ['bram', 'piek']:
                pronoun = 'he' if role == 'subject' else 'him' if role == 'object'  else entity_label
            elif entity_label.lower() in ['selene', 'lenka', 'suzana']:
                pronoun = 'she' if role == 'subject' else 'her'
                '''
            else:
                pronoun = entity_label

            return pronoun
