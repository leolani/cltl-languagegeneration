import random
from typing import Optional

from cltl.commons.language_data.sentences import NEW_KNOWLEDGE, EXISTING_KNOWLEDGE, CONFLICTING_KNOWLEDGE, \
    CURIOSITY, HAPPY
from cltl.commons.triple_helpers import filtered_types_names
from langchain_ollama import ChatOllama

from cltl.reply_generation.api import Phraser
from cltl.reply_generation.utils.phraser_utils import clean_overlaps

LLAMA_MODEL = ChatOllama(model="llama3.2", temperature=0.1)


class LlamaPhraser(Phraser):

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
            conflict = random.choice(conflicts["provenance"])

            prompt_cardinality_conflict = {"role": "system",
                                           "content": f"You are an intelligent assistant. "
                                                      f"I will give you as input: two triples with a subject, a predicate and an object, and provenance information of who and when someone mentioned the second triple "
                                                      f"You need to paraphrase the input in plain English as a statement that acknowledges that the information is conflicting. "
                                                      f"Only reply with the short paraphrase of the input. "
                                                      f"When responding use the names from the triple and be specific. "
                                                      f"Do not give an explanation. "
                                                      f"Do not explain what the subject and object is. "
                                                      f"The response should be just the paraphrased text and nothing else."}

            triple_text = f"{utterance['triple']['_subject']['_label']} " \
                          f"{utterance['triple']['_predicate']['_label']} " \
                          f"{utterance['triple']['_complement']['_label']}"
            conflict_text = f"{utterance['triple']['_subject']['_label']} " \
                            f"{utterance['triple']['_predicate']['_label']} " \
                            f"{conflict['_complement']['_label']}"
            conflict_date = f"AUTHOR: {conflict['_provenance']['_author']['_label']}, " \
                            f"DATE: {conflict['_provenance']['_date']}"

            prompt = [prompt_cardinality_conflict, {"role": "user", "content": f"{triple_text} "
                                                                               f"{conflict_text}"
                                                                               f"{conflict_date}"}]
            response = LLAMA_MODEL.invoke(prompt)
            say += response.content

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
                prompt_negation_conflict = {"role": "system",
                                            "content": f"You are an intelligent assistant. "
                                                       f"I will give you as input: a triple with a subject, a predicate and an object, and provenance information of who and when someone confirmed and denied this triple "
                                                       f"You need to paraphrase the input in plain English as a statement that acknowledges that the information is conflicting. "
                                                       f"Only reply with the short paraphrase of the input. "
                                                       f"When responding use the names from the triple and be specific. "
                                                       f"Do not give an explanation. "
                                                       f"Do not explain what the subject and object is. "
                                                       f"The response should be just the paraphrased text and nothing else."}

                triple_text = f"{utterance['triple']['_subject']['_label']} " \
                              f"{utterance['triple']['_predicate']['_label']} " \
                              f"{utterance['triple']['_complement']['_label']}"
                affirmative_date = f"CONFIRM:" \
                                   f"AUTHOR: {affirmative_conflict['_provenance']['_author']['_label']}, " \
                                   f"DATE: {affirmative_conflict['_provenance']['_date']}"
                negative_date = f"DENY: " \
                                f"AUTHOR: {negative_conflict['_provenance']['_author']['_label']}, " \
                                f"DATE: {negative_conflict['_provenance']['_date']}"

                prompt = [prompt_negation_conflict, {"role": "user", "content": f"{triple_text} "
                                                                                f"{affirmative_date} "
                                                                                f"{negative_date}"}]
                response = LLAMA_MODEL.invoke(prompt)
                say += response.content

                return say

    @staticmethod
    def _phrase_statement_novelty(novelties, utterance):
        # type: (dict, dict) -> Optional[str]
        novelties = novelties["provenance"]

        # I do not know this before, so be happy to learn
        if not novelties:
            say = random.choice(NEW_KNOWLEDGE)
            prompt_novelty = {"role": "system",
                              "content": f"You are an intelligent assistant. "
                                         f"I will give you as input: a triple with a subject, a predicate and an object. "
                                         f"You need to paraphrase the input in plain English as a statement that excitement to learn about the triple. "
                                         f"Only reply with the short paraphrase of the input. "
                                         f"When responding use the names from the triple and be specific. "
                                         f"Do not give an explanation. "
                                         f"Do not explain what the subject and object is. "
                                         f"The response should be just the paraphrased text and nothing else."}
            triple_text = f"{utterance['triple']['_subject']['_label']} " \
                          f"{utterance['triple']['_predicate']['_label']} " \
                          f"{utterance['triple']['_complement']['_label']}"
            prompt = [prompt_novelty, {"role": "user", "content": triple_text}]
            response = LLAMA_MODEL.invoke(prompt)
            say += response.content

        # I already knew this
        else:  # TODO not working
            say = random.choice(EXISTING_KNOWLEDGE)
            novelty = random.choice(novelties)
            prompt_no_novelty = {"role": "system",
                                 "content": f"You are an intelligent assistant. "
                                            f"I will give you as input: a triple with a subject, a predicate and an object, and provenance information of who and when they said this triple "
                                            f"You need to paraphrase the input in plain English as a statement that acknowledges that the triple is known. "
                                            f"Only reply with the short paraphrase of the input. "
                                            f"When responding use the names from the triple and be specific. "
                                            f"Do not give an explanation. "
                                            f"Do not explain what the subject and object is. "
                                            f"The response should be just the paraphrased text and nothing else."}
            triple_text = f"{utterance['triple']['_subject']['_label']} " \
                          f"{utterance['triple']['_predicate']['_label']} " \
                          f"{utterance['triple']['_complement']['_label']}"
            triple_date = f"AUTHOR: {novelty['_provenance']['_author']['_label']}, " \
                          f"DATE: {novelty['_provenance']['_date']}"
            prompt = [prompt_no_novelty, {"role": "user", "content": f"{triple_text} "
                                                                     f"{triple_date}"}]
            response = LLAMA_MODEL.invoke(prompt)
            say += response.content

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
        else:
            entity_label = utterance['triple']['_subject']['_label'] if entity_role == 'subject' \
                else utterance['triple']['_complement']['_label']

        if not novelty:  # TODO: too dramatic
            say = random.choice(NEW_KNOWLEDGE)
            prompt_novelty = {"role": "system",
                              "content": f"You are an intelligent assistant. "
                                         f"I will give you as input: an entity name. "
                                         f"You need to paraphrase the input in plain English as statement that expresses excitement to learn about the entity. "
                                         f"Only reply with the short paraphrase of the input. "
                                         f"When responding use the names of the entity and be specific. "
                                         f"Do not give an explanation. "
                                         f"Do not explain what the entity is. "
                                         f"The response should be just the paraphrased text and nothing else."}
            prompt = [prompt_novelty, {"role": "user", "content": entity_label}]
            response = LLAMA_MODEL.invoke(prompt)
            say += response.content

        else:  # TODO: add provenance information of when it was learned and by who
            say = random.choice(EXISTING_KNOWLEDGE)
            prompt_no_novelty = {"role": "system",
                                 "content": f"You are an intelligent assistant. "
                                            f"I will give you as input: an entity name. "
                                            f"You need to paraphrase the input in plain English as statement that acknowledges that the entity is known. "
                                            f"Only reply with the short paraphrase of the input. "
                                            f"When responding use the names of the entity and be specific. "
                                            f"Do not give an explanation. "
                                            f"Do not explain what the entity is. "
                                            f"The response should be just the paraphrased text and nothing else."}
            prompt = [prompt_no_novelty, {"role": "user", "content": entity_label}]
            response = LLAMA_MODEL.invoke(prompt)
            say += response.content

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

        if entity_role == 'subject':  # TODO: not working
            prompt_subject_gap = {"role": "system",
                                  "content": f"You are an intelligent assistant. "
                                             f"I will give you as input: a triple with a subject, a predicate and an OBJECT TYPE. "
                                             f"You need to paraphrase the input in plain English as a question to find out the object. "
                                             f"Use who for the type person, where for the type location, when for the type time and what for everything else. "
                                             f"Only reply with the short paraphrase of the input. "
                                             f"When responding use the names from the triple and be specific. "
                                             f"Do not give an explanation. "
                                             f"Do not explain what the subject and object is. "
                                             f"The response should be just the paraphrased text and nothing else."}
            triple_text = f"{gap['_known_entity']['_label']} " \
                          f"{gap['_predicate']['_label']} " \
                          f"{filtered_types_names(gap['_target_entity_type']['_types']).upper()}"
            prompt = [prompt_subject_gap, {"role": "user", "content": triple_text}]
            response = LLAMA_MODEL.invoke(prompt)
            say += response.content


        elif entity_role == 'object':  # TODO sometimes working
            prompt_object_gap = {"role": "system",
                                 "content": f"You are an intelligent assistant. "
                                            f"I will give you as input: a triple with a SUBJECT TYPE, a predicate and an object."
                                            f"You need to paraphrase the input in plain English as a question to find out the subject. "
                                            f"Use who for the type person, where for the type location, when for the type time and what for everything else. "
                                            f"Only reply with the short paraphrase of the input. "
                                            f"When responding use the names from the triple and be specific. "
                                            f"Do not give an explanation. "
                                            f"Do not explain what the subject and object is. "
                                            f"The response should be just the paraphrased text and nothing else."}

            triple_text = f"{filtered_types_names(gap['_target_entity_type']['_types']).upper()} " \
                          f"{gap['_predicate']['_label']} " \
                          f"{gap['_known_entity']['_label']} "
            prompt = [prompt_object_gap, {"role": "user", "content": triple_text}]
            response = LLAMA_MODEL.invoke(prompt)
            say += response.content

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

        if entity_role == 'subject':  # TODO: not working
            prompt_subject_gap = {"role": "system",
                                  "content": f"You are an intelligent assistant. "
                                             f"I will give you as input: a triple with a subject, a predicate and an OBJECT TYPE. "
                                             f"You need to paraphrase the input in plain English as a question to find out the object. "
                                             f"Use who for the type person, where for the type location, when for the type time and what for everything else. "
                                             f"Only reply with the short paraphrase of the input. "
                                             f"When responding use the names from the triple and be specific. "
                                             f"Do not give an explanation. "
                                             f"Do not explain what the subject and object is. "
                                             f"The response should be just the paraphrased text and nothing else."}
            triple_text = f"{gap['_known_entity']['_label']} " \
                          f"{gap['_predicate']['_label']} " \
                          f"{filtered_types_names(gap['_target_entity_type']['_types']).upper()}"
            prompt = [prompt_subject_gap, {"role": "user", "content": triple_text}]
            response = LLAMA_MODEL.invoke(prompt)
            say += response.content


        elif entity_role == 'object':  # TODO sometimes working
            prompt_object_gap = {"role": "system",
                                 "content": f"You are an intelligent assistant. "
                                            f"I will give you as input: a triple with a SUBJECT TYPE, a predicate and an object."
                                            f"You need to paraphrase the input in plain English as a question to find out the subject. "
                                            f"Use who for the type person, where for the type location, when for the type time and what for everything else. "
                                            f"Only reply with the short paraphrase of the input. "
                                            f"When responding use the names from the triple and be specific. "
                                            f"Do not give an explanation. "
                                            f"Do not explain what the subject and object is. "
                                            f"The response should be just the paraphrased text and nothing else."}

            triple_text = f"{filtered_types_names(gap['_target_entity_type']['_types']).upper()} " \
                          f"{gap['_predicate']['_label']} " \
                          f"{gap['_known_entity']['_label']} "
            prompt = [prompt_object_gap, {"role": "user", "content": triple_text}]
            response = LLAMA_MODEL.invoke(prompt)
            say += response.content

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

        if len(overlaps) < 2:  # TODO: phrase as "did you know selene ALSO likes dancing"
            prompt_overlap = {"role": "system",
                              "content": f"You are an intelligent assistant. "
                                         f"I will give you as input: two triples with a subject, a predicate and an object."
                                         f"You need to paraphrase the input in plain English as a statement that expresses that the first triple is related to the second triple. "
                                         f"Only reply with the short paraphrase of the input. "
                                         f"When responding use the names from the triple and be specific. "
                                         f"Do not give an explanation. "
                                         f"Do not explain what the subject and object is. "
                                         f"The response should be just the paraphrased text and nothing else."}

            triple_text = f"{utterance['triple']['_subject']['_label']} " \
                          f"{utterance['triple']['_predicate']['_label']} " \
                          f"{utterance['triple']['_complement']['_label']} "
            overlap_text = f"{utterance['triple']['_subject']['_label']} " \
                           f"{utterance['triple']['_predicate']['_label']} " \
                           f"{overlaps[0]['_entity']['_label']} "
            prompt = [prompt_overlap, {"role": "user", "content": f"{triple_text} {overlap_text}"}]
            response = LLAMA_MODEL.invoke(prompt)
            say += response.content

        else:
            sample = random.sample(overlaps, 2)
            prompt_overlap = {"role": "system",
                              "content": f"You are an intelligent assistant. "
                                         f"I will give you as input: three triples with a subject, a predicate and an object, and a number."
                                         f"You need to paraphrase the input in plain English as a statement that expresses that the first triple is related to the other two triples and that there are NUMBER related triples in total. "
                                         f"Only reply with the short paraphrase of the input. "
                                         f"When responding use the names from the triple and be specific. "
                                         f"Do not give an explanation. "
                                         f"Do not explain what the subject and object is. "
                                         f"The response should be just the paraphrased text and nothing else."}

            triple_text = f"{utterance['triple']['_subject']['_label']} " \
                          f"{utterance['triple']['_predicate']['_label']} " \
                          f"{utterance['triple']['_complement']['_label']} "
            overlap_text = f"{utterance['triple']['_subject']['_label']} " \
                           f"{utterance['triple']['_predicate']['_label']} " \
                           f"{sample[0]['_entity']['_label']} " \
                           f"{utterance['triple']['_subject']['_label']} " \
                           f"{utterance['triple']['_predicate']['_label']} " \
                           f"{sample[1]['_entity']['_label']} "
            prompt = [prompt_overlap, {"role": "user", "content": f"{triple_text} {overlap_text}"}]
            response = LLAMA_MODEL.invoke(prompt)
            say += response.content

        return say
