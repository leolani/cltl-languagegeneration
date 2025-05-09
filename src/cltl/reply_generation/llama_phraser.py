import random
from typing import Optional

from cltl.commons.language_data.sentences import NEW_KNOWLEDGE, EXISTING_KNOWLEDGE, CONFLICTING_KNOWLEDGE, \
    CURIOSITY, HAPPY
from cltl.thoughts.thought_selection.utils.thought_utils import separate_select_negation_conflicts
from langchain_ollama import ChatOllama

from cltl.reply_generation.api import Phraser
from cltl.reply_generation.utils.phraser_utils import dash_replace, prepare_triple, prepare_perspective, \
    prepare_speaker, prepare_author_from_thought, prepare_gap, prepare_overlap


class LlamaPhraser(Phraser):

    def __init__(self):
        # type: () -> None
        """
        Generate natural language based on structured data

        Parameters
        ----------
        """

        super(LlamaPhraser, self).__init__()

        self.llama_model = ChatOllama(model="llama3.2", temperature=0.1)
        self.base_prompt = {"role": "system",
                            "content": f"You are in the middle of a conversation with a person. "
                                       f"They just gave you a piece of information and you must reply accordingly. "
                                       f"The information you will use for the response will be given as structured data that you need to phrase as natural language. "
                                       f"Specifically, you will be translating triples (subject - predicate - object) into plain English. "
                                       f"The response should be just the paraphrased text and nothing else. Do not add extra information. "
                                       f"Do not give an explanation of why you paraphrased it that way, or what the subject and object is. "
                                       f"When responding, be specific and use the names from the triple where possible, but do not add anything that is not given in the triples. "
                                       f"Remember to make it fluent and resolve any co-references or use pronouns where needed. "
                                       f"Only reply with the short paraphrase of the input. "}

    def phrase_triple(self, utterance):
        # type: (dict) -> Optional[str]
        triple_text = prepare_triple(utterance)
        speaker = prepare_speaker(utterance)
        perspective_text = prepare_perspective(utterance)
        prompt_triple = {"role": "system",
                         "content": f"You are a speaker in a conversation that wants to express some information and your perspective on it. "
                                    f"You will be given structured data as triples (subject - predicate - object) . "
                                    f"You will also be given perspectives including values like certainty polarity sentiment emotion. "
                                    f"The utterance should be express both the perspective and the triple information. Do not add extra information. "
                                    f"Do not give an explanation of why you phrased it that way. "
                                    f"Be fluent and natural for a casual dialogue. "
                                    f"Use pronouns where needed, for example if the triple information relates to you, the speaker, use 'I'. "
                                    f"YOUR NAME: <SPEAKER> "
                                    f"TRIPLE: <SUBJECT> <PREDICATE> <OBJECT> "
                                    f"PERSPECTIVE: <CERTAINTY> <POLARITY> <SENTIMENT> <EMOTION>"}

        prompt = [prompt_triple,
                  {"role": "user",
                   "content": f"YOUR NAME: {speaker} "
                              f"TRIPLE: {triple_text} "
                              f"PERSPECTIVE: {perspective_text} "}]
        response = self.llama_model.invoke(prompt)
        say = f" {response.content}"

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

            triple_text = prepare_triple(utterance)
            speaker = prepare_speaker(utterance)
            conflict_text = f"{dash_replace(utterance['triple']['_subject']['_label'])} " \
                            f"{dash_replace(utterance['triple']['_predicate']['_label'])} " \
                            f"{dash_replace(conflict['_complement']['_label'])}"
            conflict_author = prepare_author_from_thought(conflict)

            prompt_cardinality_conflict = {"role": "system",
                                           "content": f"CASE: The incoming information is conflicting with something someone else told you. "
                                                      f"RESPONSE FORMAT: Statement that acknowledges that the information is conflicting, and mentions the person that provided the conflicting information in the past. "
                                                      f"HAVING CONVERSATION WITH: <CURRENT_SPEAKER> "
                                                      f"INCOMING INFORMATION: <SUBJECT> <PREDICATE> <OBJECT1> "
                                                      f"PAST CONFLICTING INFORMATION: <SUBJECT> <PREDICATE> <OBJECT2> "
                                                      f"AUTHOR OF PAST CONFLICTING INFORMATION: <AUTHOR> "}

            prompt = [self.base_prompt, prompt_cardinality_conflict,
                      {"role": "user",
                       "content": f"HAVING CONVERSATION WITH: {speaker} "
                                  f"INCOMING INFORMATION: {triple_text} "
                                  f"PAST CONFLICTING INFORMATION: {conflict_text} "
                                  f"AUTHOR OF PAST CONFLICTING INFORMATION: {conflict_author}"}]
            response = self.llama_model.invoke(prompt)
            say += f" {response.content}"

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

                triple_text = prepare_triple(utterance)
                speaker = prepare_speaker(utterance)
                affirmative = prepare_author_from_thought(affirmative_conflict)
                negative = prepare_author_from_thought(negative_conflict)

                prompt_negation_conflict = {"role": "system",
                                            "content": f"CASE: The incoming information is denying something someone else told you. "
                                                       f"RESPONSE FORMAT: Statement that acknowledges that the information has both been confirmed and denied, and mentions the person that provided you with the conflicting information in the past. "
                                                       f"HAVING CONVERSATION WITH: <CURRENT_SPEAKER> "
                                                       f"INCOMING INFORMATION: <SUBJECT> <PREDICATE> <OBJECT> "
                                                       f"AUTHOR CONFIRMING: <AUTHOR1> "
                                                       f"AUTHOR DENYING: <AUTHOR2> "}

                prompt = [self.base_prompt, prompt_negation_conflict,
                          {"role": "user", "content": f"HAVING CONVERSATION WITH: {speaker} "
                                                      f"INCOMING INFORMATION: {triple_text} "
                                                      f"AUTHOR CONFIRMING: {affirmative} "
                                                      f"AUTHOR DENYING: {negative}"}]
                response = self.llama_model.invoke(prompt)
                f" {response.content}"

                return say

    def _phrase_statement_novelty(self, selected_thought, utterance):
        # type: (dict, dict) -> Optional[str]

        triple_text = prepare_triple(utterance)
        speaker = prepare_speaker(utterance)

        # I do not know this before, so be happy to learn
        if not selected_thought or not selected_thought["thought_info"]:
            say = random.choice(NEW_KNOWLEDGE)

            prompt_novelty = {"role": "system",
                              "content": f"CASE: The incoming information is new. "
                                         f"RESPONSE FORMAT: Statement that acknowledges that the information is was not known and shows excitement to learn something new. "
                                         f"HAVING CONVERSATION WITH: <CURRENT_SPEAKER> "
                                         f"INCOMING INFORMATION: <SUBJECT> <PREDICATE> <OBJECT> "}

            prompt = [self.base_prompt, prompt_novelty,
                      {"role": "user", "content": f"HAVING CONVERSATION WITH: {speaker} "
                                                  f"INCOMING INFORMATION: {triple_text} "}]
            response = self.llama_model.invoke(prompt)
            say += f" {response.content}"

        # I already knew this
        else:
            say = random.choice(EXISTING_KNOWLEDGE)
            novelty = selected_thought["thought_info"]
            past_author = prepare_author_from_thought(novelty)

            prompt_no_novelty = {"role": "system",
                                 "content": f"CASE: The incoming information was previously mentioned by someone. "
                                            f"RESPONSE FORMAT: Statement that acknowledges that the information is known and mentions the person that is the original source. "
                                            f"HAVING CONVERSATION WITH: <CURRENT_SPEAKER> "
                                            f"INCOMING INFORMATION: <SUBJECT> <PREDICATE> <OBJECT> "
                                            f"AUTHOR OF PAST MENTION OF INFORMATION: <ORIGINAL_AUTHOR> "}

            prompt = [self.base_prompt, prompt_no_novelty,
                      {"role": "user", "content": f"HAVING CONVERSATION WITH: {speaker} "
                                                  f"INCOMING INFORMATION: {triple_text} "
                                                  f"AUTHOR OF PAST MENTION OF INFORMATION: {past_author}"}]
            response = self.llama_model.invoke(prompt)
            say += f" {response.content}"

        return say

    def _phrase_type_novelty(self, selected_thought, utterance):
        # type: (dict, dict) -> Optional[str]

        # There is no novelty information, so no response
        entity_role = selected_thought["extra_info"]
        novelty = selected_thought["thought_info"]

        if 'entity' in utterance.keys():
            entity_label = dash_replace(utterance['entity']['_label'])
        else:
            speaker = prepare_speaker(utterance)
            entity_label = dash_replace(utterance['triple'][entity_role]['_label'])

        # There is no novelty information, so happy to learn
        if not selected_thought or not selected_thought["thought_info"] or selected_thought["thought_info"]["value"]:
            say = random.choice(NEW_KNOWLEDGE)

            prompt_novelty = {"role": "system",
                              "content": f"CASE: The entity mentioned is new. "
                                         f"RESPONSE FORMAT: Statement that acknowledges that the entity was not known and shows excitement to learn about it. "
                                         f"HAVING CONVERSATION WITH: <CURRENT_SPEAKER> "
                                         f"INCOMING ENTITY: <ENTITY> "}

            prompt = [self.base_prompt, prompt_novelty,
                      {"role": "user", "content": f"HAVING CONVERSATION WITH: {speaker} "
                                                  f"INCOMING ENTITY: {entity_label} "}]
            response = self.llama_model.invoke(prompt)
            say += f" {response.content}"

        else:
            say = random.choice(EXISTING_KNOWLEDGE)
            past_author = prepare_author_from_thought(novelty)

            prompt_no_novelty = {"role": "system",
                                 "content": f"CASE: The entity mentioned was previously mentioned by someone. "
                                            f"RESPONSE FORMAT: Statement that acknowledges that the entity is known and mentions the person that mentioned it before. "
                                            f"HAVING CONVERSATION WITH: <CURRENT_SPEAKER> "
                                            f"INCOMING ENTITY: <ENTITY> "
                                            f"AUTHOR OF PAST MENTION OF ENTITY: <ORIGINAL_AUTHOR> "}

            prompt = [self.base_prompt, prompt_no_novelty,
                      {"role": "user", "content": f"HAVING CONVERSATION WITH: {speaker} "
                                                  f"INCOMING ENTITY: {entity_label} "
                                                  f"AUTHOR OF PAST MENTION OF ENTITY: {past_author}"}]
            response = self.llama_model.invoke(prompt)
            say += f" {response.content}"

        return say

    def _phrase_subject_gaps(self, selected_thought, utterance):
        # type: (dict, dict) -> Optional[str]

        # There is no gaps, so no response
        if not selected_thought or not selected_thought["thought_info"]:
            return None

        # random choice between object or subject
        entity_role = selected_thought["extra_info"]
        gap = selected_thought["thought_info"]

        say = random.choice(CURIOSITY)
        speaker = prepare_speaker(utterance)
        triple_text = prepare_triple(utterance)
        triple_gap = prepare_gap(gap, role=entity_role)

        if entity_role == '_subject':
            prompt_gap = {"role": "system",
                          "content": f"CASE: The incoming information triggers a question about the subject mentioned. "
                                     f"RESPONSE FORMAT: Question to find out what the object of a triple might be, given the object type. Use who for the type person, where for the type location, when for the type time and what for everything else. "
                          # f"You can improve fluency in the dialogue by linking the question to the last incoming information. If not useful, just ignore it. "
                                     f"HAVING CONVERSATION WITH: <CURRENT_SPEAKER> "
                          # f"INCOMING INFORMATION: <SUBJECT> <PREDICATE> <OBJECT> "
                                     f"INFORMATION TO FILL IN: <SUBJECT> <PREDICATE2> <OBJECT_TYPE> "}

        elif entity_role == '_complement':
            prompt_gap = {"role": "system",
                          "content": f"CASE: The incoming information triggers a question about the subject mentioned. "
                                     f"RESPONSE FORMAT: Question to find out what the subject of a triple might be, given the subject type. Use who for the type person, where for the type location, when for the type time and what for everything else. "
                          # f"You can improve fluency in the dialogue by linking the question to the last incoming information. If not useful, just ignore it. "
                                     f"HAVING CONVERSATION WITH: <CURRENT_SPEAKER> "
                          # f"INCOMING INFORMATION: <SUBJECT> <PREDICATE> <OBJECT> "
                                     f"INFORMATION TO FILL IN: <SUBJECT_TYPE> <PREDICATE2> <SUBJECT> "}

        prompt = [self.base_prompt, prompt_gap,
                  {"role": "user", "content": f"HAVING CONVERSATION WITH: {speaker} "
                  # f"INCOMING INFORMATION: {triple_text} "
                                              f"INFORMATION TO FILL IN: {triple_gap}"}]
        response = self.llama_model.invoke(prompt)
        say += f" {response.content}"

        return say

    def _phrase_complement_gaps(self, selected_thought, utterance):
        # type: (dict, dict) -> Optional[str]

        # There is no gaps, so no response
        if not selected_thought or not selected_thought["thought_info"]:
            return None

        entity_role = selected_thought["extra_info"]
        gap = selected_thought["thought_info"]

        say = random.choice(CURIOSITY)
        speaker = prepare_speaker(utterance)
        # triple_text = prepare_triple(utterance)
        triple_gap = prepare_gap(gap, role=entity_role)

        if entity_role == '_subject':
            prompt_gap = {"role": "system",
                          "content": f"CASE: The incoming information triggers a question about the object mentioned. "
                                     f"RESPONSE FORMAT: Question to find out what the object of a triple might be, given the object type. Use who for the type person, where for the type location, when for the type time and what for everything else. "
                                     f"HAVING CONVERSATION WITH: <CURRENT_SPEAKER> "
                                     f"INFORMATION TO FILL IN: <SUBJECT> <PREDICATE2> <OBJECT_TYPE> "}

        elif entity_role == '_complement':
            prompt_gap = {"role": "system",
                          "content": f"CASE: The incoming information triggers a question about the object mentioned. "
                                     f"RESPONSE FORMAT: Question to find out what the subject of a triple might be, given the subject type. Use who for the type person, where for the type location, when for the type time and what for everything else. "
                                     f"HAVING CONVERSATION WITH: <CURRENT_SPEAKER> "
                                     f"INFORMATION TO FILL IN: <SUBJECT_TYPE> <PREDICATE2> <SUBJECT> "}

        prompt = [self.base_prompt, prompt_gap,
                  {"role": "user", "content": f"HAVING CONVERSATION WITH: {speaker} "
                  # f"INCOMING INFORMATION: {triple_text} "
                                              f"INFORMATION TO FILL IN: {triple_gap}"}]
        response = self.llama_model.invoke(prompt)
        say += f" {response.content}"

        return say

    def _phrase_overlaps(self, selected_thought, utterance):
        # type: (dict, dict) -> Optional[str]

        if not selected_thought or not selected_thought["thought_info"]:
            return None

        entity_role = selected_thought["extra_info"]
        overlap = selected_thought["thought_info"]

        say = random.choice(HAPPY)
        speaker = prepare_speaker(utterance)
        triple_text = prepare_triple(utterance)

        # TODO: phrase as "did you know selene ALSO likes dancing"
        prompt_overlap = {"role": "system",
                          "content": f"CASE: The incoming information overlaps with something already mentioned. "
                                     f"RESPONSE FORMAT: Statement to show that the incoming information is related to something known. "
                                     f"HAVING CONVERSATION WITH: <CURRENT_SPEAKER> "
                                     f"INCOMING INFORMATION: <SUBJECT> <PREDICATE> <OBJECT> "
                                     f"PAST KNOWN INFORMATION: <SUBJECT2> <PREDICATE> <OBJECT2> "}

        overlap_text = prepare_overlap(utterance, overlap=overlap, role=entity_role)
        prompt = [self.base_prompt, prompt_overlap,
                  {"role": "user",
                   "content": f"HAVING CONVERSATION WITH: {speaker} "
                              f"INCOMING INFORMATION:{triple_text} "
                              f"PAST KNOWN INFORMATION: {overlap_text} "}]
        response = self.llama_model.invoke(prompt)
        say += f" {response.content}"

        return say
