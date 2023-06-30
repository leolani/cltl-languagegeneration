""" Filename:     rl_replier.py
    Author(s):    Thomas Bellucci
    Description:  Modified BasicRepliers for use in the Chatbot. The replier
                  takes out the Thought selection step and only selects among
                  thoughts using reinforcement learning (RLReplier).
    Date created: Nov. 11th, 2021
"""

from cltl.commons.casefolding import (casefold_capsule)

from cltl.reply_generation.lenka_replier import LenkaReplier
from cltl.reply_generation.phrasers.simplenlg_phraser import SimplenlgPhraser
from cltl.reply_generation.thought_selectors.rl_selector import UCB
from cltl.reply_generation.utils.thought_utils import thoughts_from_brain

from rdflib import ConjunctiveGraph
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph


# from cltl.dialogue_evaluation.metrics.ontology_measures import get_avg_population
# from cltl.dialogue_evaluation.metrics.graph_measures import get_avg_degree, get_sparseness, get_shortest_path


class BrainEvaluator(object):
    def __init__(self, brain):
        """ Create an object to evaluate the state of the brain according to different graph metrics.
        The graph can be evaluated by a single given metric, or a full set of pre established metrics
        """
        self._brain = brain

    def brain_as_graph(self):
        # Take brain from previous episodes
        graph = ConjunctiveGraph()
        graph.parse(data=self._brain._connection.export_repository(), format='trig')

        return graph

    def brain_as_netx(self):
        # Take brain from previous episodes
        netx = rdflib_to_networkx_multidigraph(self.brain_as_graph())

        return netx

    def evaluate_brain_state(self, metric):
        brain_state = None

        # if metric == 'Average degree':
        #     brain_state = get_avg_degree(self.brain_as_netx())
        # elif metric == 'Sparseness':
        #     brain_state = get_sparseness(self.brain_as_netx())
        # elif metric == 'Shortest path':
        #     brain_state = get_shortest_path(self.brain_as_netx())

        if metric == 'Total triples':
            brain_state = self._brain.count_triples()
        # elif metric == 'Average population':
        #     brain_state = get_avg_population(self.brain_as_graph())

        elif metric == 'Ratio claims to triples':
            brain_state = self._brain.count_statements() / self._brain.count_triples()
        elif metric == 'Ratio perspectives to claims':
            brain_state = self._brain.count_perspectives() / self._brain.count_statements()
        elif metric == 'Ratio conflicts to claims':
            brain_state = len(self._brain.get_all_negation_conflicts()) / self._brain.count_statements()

        return brain_state

    def calculate_brain_statistics(self, brain_response):
        # Grab the thoughts
        thoughts = brain_response['thoughts']

        # Gather basic stats
        stats = {
            'turn': brain_response['statement']['turn'],

            'cardinality conflicts': len(thoughts['_complement_conflict']) if thoughts['_complement_conflict'] else 0,
            'negation conflicts': len(thoughts['_negation_conflicts']) if thoughts['_negation_conflicts'] else 0,
            'subject gaps': len(thoughts['_subject_gaps']) if thoughts['_subject_gaps'] else 0,
            'object gaps': len(thoughts['_complement_gaps']) if thoughts['_complement_gaps'] else 0,
            'statement novelty': len(thoughts['_statement_novelty']) if thoughts['_statement_novelty'] else 0,
            'subject novelty': thoughts['_entity_novelty']['_subject'],
            'object novelty': thoughts['_entity_novelty']['_complement'],
            'overlaps subject-predicate': len(thoughts['_overlaps']['_subject'])
            if thoughts['_overlaps']['_subject'] else 0,
            'overlaps predicate-object': len(thoughts['_overlaps']['_complement'])
            if thoughts['_overlaps']['_complement'] else 0,
            'trust': thoughts['_trust'],

            'Total triples': self._brain.count_triples(),
            # 'Total classes': len(self._brain.get_classes()),
            # 'Total predicates': len(self._brain.get_predicates()),
            'Total claims': self._brain.count_statements(),
            'Total perspectives': self._brain.count_perspectives(),
            'Total conflicts': len(self._brain.get_all_negation_conflicts()),
            'Total sources': self._brain.count_friends(),
        }

        # Compute composite stats
        stats['Ratio claims to triples'] = stats['Total claims'] / stats['Total triples']
        stats['Ratio perspectives to triples'] = stats['Total perspectives'] / stats['Total triples']
        stats['Ratio conflicts to triples'] = stats['Total conflicts'] / stats['Total triples']
        stats['Ratio perspectives to claims'] = stats['Total perspectives'] / stats['Total claims']
        stats['Ratio conflicts to claims'] = stats['Total conflicts'] / stats['Total claims']

        return stats


class RLReplier(LenkaReplier):
    def __init__(self, brain, savefile=None, reward="Total triples"):
        """Creates a reinforcement learning-based replier to respond to questions
        and statements by the user. Statements are replied to by phrasing a
        thought; Selection of the thoughts are learnt by the UCB algorithm.

        params
        object brain: the brain (triple store)
        str savefile: file with stored utility values in JSON format
        str reward: type of function to evaluate the brain state

        returns: None
        """
        super(RLReplier, self).__init__()
        self._thought_selector = UCB()
        self._thought_selector.load(savefile)
        self._log.debug(f"UCB RL Selector ready")

        self._phraser = SimplenlgPhraser()
        self._log.debug(f"SimpleNLG phraser ready")

        self._state_evaluator = BrainEvaluator(brain)
        self._log.debug(f"Brain state evaluator ready")

        self._reward = reward
        self._log.info(f"Reward: {self._reward}")

        self._last_thought = None
        self._state_history = []
        self._reward_history = [0]

    @property
    def state_history(self):
        return self._state_history

    @property
    def reward_history(self):
        return self._reward_history

    def reward_thought(self):
        """Rewards the last thought phrased by the replier by updating its
        utility estimate with the relative improvement of the brain as
        a result of the user response (i.e. a reward).

        returns: None
        """
        self._log.info(f"Calculate reward")
        brain_state = self._state_evaluator.evaluate_brain_state(self._reward)

        self._state_history.append(brain_state)
        self._log.info(f"Brain state: {brain_state}")

        # Reward last thought with R = S_brain(t) - S_brain(t-1)
        if self._last_thought and self._state_history[-1] and self._state_history[-2]:
            new_state = self._state_history[-1]
            old_state = self._state_history[-2]
            reward = new_state / old_state

            self._thought_selector.update_utility(self._last_thought, reward)
            self.reward_history.append(reward)
            self._log.info(f"{reward} reward due to {self._last_thought}")

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
            thought_options = ['_entity_novelty', '_complement_gaps']
        self._log.debug(f'Thoughts options: {thought_options}')

        # Casefold
        utterance = casefold_capsule(brain_response['statement'], format='natural')
        thoughts = casefold_capsule(brain_response['thoughts'], format='natural')

        # Extract thoughts from brain response
        thoughts = thoughts_from_brain(utterance, thoughts, filter=thought_options)

        # Select thought
        self._last_thought = self._thought_selector.select(thoughts.keys())
        thought_type, thought_info = thoughts[self._last_thought]
        self._log.info(f"Chosen thought type: {thought_type}")

        # Preprocess thought_info and utterance (triples)
        thought_info = {"thought": thought_info}
        thought_info = casefold_capsule(thought_info, format="natural")
        thought_info = thought_info["thought"]

        # Generate reply
        reply = self._phraser.phrase_correct_thought(utterance, thought_type, thought_info)

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

        # Extract thoughts from brain response
        thoughts = thoughts_from_brain(utterance, thoughts, filter=thought_options)

        # Select thought
        self._last_thought = self._thought_selector.select(thoughts.keys())
        thought_type, thought_info = thoughts[self._last_thought]
        self._log.info(f"Chosen thought type: {thought_type}")

        # Preprocess thought_info and utterance (triples)
        utterance = casefold_capsule(brain_response["statement"], format="natural")
        thought_info = {"thought": thought_info}
        thought_info = casefold_capsule(thought_info, format="natural")
        thought_info = thought_info["thought"]

        # Generate reply
        reply = self._phraser.phrase_correct_thought(utterance, thought_type, thought_info)

        return reply
