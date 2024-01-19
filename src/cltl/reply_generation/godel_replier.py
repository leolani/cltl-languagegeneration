import random

from cltl.commons.casefolding import casefold_capsule
from cltl.commons.language_data.sentences import NO_ANSWER
from cltl.commons.language_helpers import lexicon_lookup
from cltl.commons.triple_helpers import filtered_types_names

from cltl.reply_generation.api import BasicReplier
from cltl.reply_generation.phrasers.pattern_phraser import PatternPhraser
from cltl.reply_generation.thought_selectors.random_selector import RandomSelector
from cltl.reply_generation.utils.phraser_utils import replace_pronouns, assign_spo, deal_with_authors, fix_entity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class GodelReplier(BasicReplier):
    def __init__(self, thought_selector = RandomSelector()):
        # type: (ThoughtSelector) -> None
        """
        Generate natural language based on structured data

        Parameters
        ----------
        thought_selector: ThoughtSelector
            Thought selector to pick thought type for the reply.
        """
        super(GodelReplier, self).__init__()
        self._thought_selector = thought_selector
        self._log.debug(f"Random Selector ready")
        self._tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
        self._model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
        self._phraser = PatternPhraser()
        self._log.debug(f"Pattern phraser ready")

   def _generate(self, instruction, knowledge, dialog):
       output = ""

       if knowledge != '':
            knowledge = '[KNOWLEDGE] ' + knowledge
       dialog = ' EOS '.join(dialog)
       query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
       input_ids = self._tokenizer(f"{query}", return_tensors="pt").input_ids
       outputs = self._model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
       output = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

       return output

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

    def reply_to_statement(self, brain_response, persist=False, thought_options=None, end_recursion=5):
        """
        Phrase a thought based on the brain response
        Parameters
        ----------
        brain_response: output of the brain
        persist: Keep looping through thoughts until you find one to phrase
        thought_options: Set from before which types of thoughts to consider in the phrasing
        end_recursion: Signal for last recursion call

        Returns
        -------

        """
        # Quick check if there is anything to do here
        if not 'triple'  in brain_response['statement']:
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


            # Replace this by callimng godel with the structured knowledge from the reponse:
            # dialog = three utterances from chat
            # knowledge = thought
            # If thought is statement:
            # instruction = f'Instruction: given a dialog context, you need to make a strong statement.'
            # else if thought is question:
            # instruction = f'Instruction: given a dialog context, you need to ask a question.'
            # reply = self._generate(instruction, knowledge, dialog)

            # Generate reply
            reply = self._phraser.phrase_correct_thought(utterance, thought_type, thoughts[thought_type],
                                                         fallback=end_recursion == 0)

            # Recursion if there is no answer
            # In theory we do not run into an infinite loop because there will always be a value for novelty
            if persist and reply is None:
                reply = self.reply_to_statement(brain_response, persist=persist, thought_options=thought_options,
                                                end_recursion=end_recursion - 1)

        return reply

    def reply_to_mention(self, brain_response, persist=False, thought_options=None):
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

            # Replace this by callimng godel with the structured knowledge from the reponse:
            # dialog = three utterances from chat
            # knowledge = thought
            # instruction = f'Instruction: given a dialog context, you need to make a strong statement.'
            # reply = self._generate(instruction, knowledge, dialog)

            # Generate reply
            reply = self._phraser.phrase_correct_thought(utterance, thought_type, thoughts[thought_type])

            # Recursion if there is no answer.
            # In theory we do not run into an infinite loop because there will always be a value for novelty
            if persist and reply is None:
                reply = self.reply_to_mention(brain_response, persist=persist, thought_options=thought_options)

        return reply
