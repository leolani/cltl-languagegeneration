import logging
import random
from typing import List, Iterable, Tuple, Callable

from cltl.brain.utils.helper_functions import brain_response_to_json
from cltl.combot.event.emissor import TextSignalEvent, ScenarioStarted, ScenarioStopped
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.event.util import extract_scenario_id
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.time_util import timestamp_now
from cltl.combot.infra.topic_worker import TopicWorker
from cltl.commons.discrete import UtteranceType
from emissor.representation.scenario import TextSignal

from cltl.reply_generation.api import BasicReplier
from cltl.reply_generation.thought_selectors.nsp_selector import NSP

logger = logging.getLogger(__name__)


CONTENT_TYPE_SEPARATOR = ';'

class ReplyGenerationService:
    @classmethod
    def from_config(cls, replier_factory: Callable[[], List[BasicReplier]], event_bus: EventBus,
                    resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.reply_generation")

        thought_options = config.get("thought_options", multi=True) \
                if "thought_options" in config \
                else ['_complement_conflict', '_negation_conflicts', '_statement_novelty', '_entity_novelty',
                      '_subject_gaps', '_complement_gaps', '_overlaps', '_trust']
        utterance_types = [UtteranceType[t.upper()] for t in config.get("utterance_types", multi=True)] \
                if "utterance_types" in config \
                else [UtteranceType.QUESTION, UtteranceType.STATEMENT, UtteranceType.TEXT_MENTION]

        buffer_size = config.get_int("buffer_size") if "buffer_size" in config else 1

        return cls(config.get("topic_input"), config.get("topic_output"), config.get("topic_scenario"),
                   config.get("intentions", multi=True), config.get("topic_intention"),
                   replier_factory, utterance_types, thought_options, buffer_size, event_bus, resource_manager)

    def __init__(self, input_topic: str, output_topic: str, scenario_topic: str, intentions: Iterable[str], intention_topic: str,
                 replier_factory: Callable[[], List[BasicReplier]], utterance_types: List[UtteranceType], thought_options: List[str],
                 buffer_size: int, event_bus: EventBus, resource_manager: ResourceManager):
        self._utterance_types = utterance_types
        self._thought_options = thought_options

        self._buffer_size = buffer_size
        self._event_bus = event_bus
        self._resource_manager = resource_manager

        self._input_topic = input_topic
        self._output_topic = output_topic
        self._scenario_topic = scenario_topic
        self._intentions = intentions
        self._intention_topic = intention_topic
        self._topic_worker = None

        self._replier_factory = replier_factory
        self._repliers = dict()

    @property
    def app(self):
        return None

    def start(self, timeout=30):
        self._topic_worker = TopicWorker([self._input_topic, self._scenario_topic], self._event_bus, provides=[self._output_topic],
                                         resource_manager=self._resource_manager, processor=self._process,
                                         buffer_size=self._buffer_size,
                                         intentions=self._intentions, intention_topic = self._intention_topic,
                                         name=self.__class__.__name__)
        self._topic_worker.start().wait()

    def stop(self):
        if not self._topic_worker:
            pass

        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._topic_worker = None

    def _process(self, event: Event[List[dict]]):
        if event.metadata.topic == self._scenario_topic:
            self._update_repliers(event)
            return

        brain_responses = [brain_response_to_json(brain_response) for brain_response in event.payload]

        scenario_id = extract_scenario_id(event)
        repliers = self._repliers[scenario_id]

        response = self._best_response(brain_responses, repliers)
        if response:
            scenario_id = extract_scenario_id(event)
            extractor_event = self._create_payload(scenario_id, response)
            self._event_bus.publish(self._output_topic, Event.for_payload(extractor_event, source=event))
            logger.debug("Created reply: %s", extractor_event.signal.text)

    def _best_response(self, brain_responses, repliers: List[BasicReplier]):
        # Prioritize replies by utterance type first, then by replier, then choose random
        typed_responses = [(self._get_utterance_type(response), response) for response in brain_responses]
        typed_responses = filter(lambda x: x[0] in self._utterance_types, typed_responses)

        ordered_responses = [(utt_type, replier, response)
                             for utt_type, response in self._ordered_by_type(typed_responses)
                             for replier in repliers]

        if not ordered_responses:
            logger.debug("No responses for %s", brain_responses)
            return None

        replies = map(self._get_reply, *zip(*ordered_responses))

        return next(filter(None, replies), None)

    def _ordered_by_type(self, typed_responses: Tuple[UtteranceType, str]) -> Tuple[UtteranceType, str]:
        randomized = list(typed_responses)
        random.shuffle(randomized)

        return sorted(randomized, key=self._utterance_type_priority)

    def _utterance_type_priority(self, utterance_response):
        return self._utterance_types.index(utterance_response[0])

    def _get_reply(self, utterance_type, replier, response):
        if utterance_type == UtteranceType.STATEMENT:
            if type(replier._thought_selector) == NSP:
                return replier.reply_to_statement_in_context(brain_response=response, persist=True, thought_options=self._thought_options)
            else:
                return replier.reply_to_statement(brain_response=response, persist=True, thought_options=self._thought_options)
        if utterance_type == UtteranceType.QUESTION:
            return replier.reply_to_question(response)
        if utterance_type == UtteranceType.TEXT_MENTION:
            return replier.reply_to_mention(response, persist=True)

        return None

    def _get_utterance_type(self, brain_response):
        if 'statement' in brain_response:
            brain_input = brain_response['statement']
        elif 'question' in brain_response:
            brain_input = brain_response['question']
        elif 'mention' in brain_response:
            brain_input = brain_response['mention']
        else:
            return None

        response_type = brain_input['utterance_type']

        if isinstance(response_type, UtteranceType):
            return response_type

        try:
            return UtteranceType[response_type]
        except:
            return None

    def _create_payload(self, scenario_id, response):
        signal = TextSignal.for_scenario(scenario_id, timestamp_now(), timestamp_now(), None, response)

        return TextSignalEvent.for_agent(signal)

    def _update_repliers(self, event):
        if event.payload.type == ScenarioStarted.__name__:
            self._repliers[event.payload.scenario.id] = self._replier_factory()
            logger.debug("Started replier for scenario %s", event.payload.scenario.id)
        elif event.payload.type == ScenarioStopped.__name__:
            del self._repliers[event.payload.scenario.id]
            logger.debug("Cleaned up replier for scenario %s", event.payload.scenario.id)