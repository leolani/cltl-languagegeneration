import logging
from typing import List, Iterable

from cltl.brain.utils.helper_functions import brain_response_to_json
from cltl.combot.event.emissor import TextSignalEvent
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.time_util import timestamp_now
from cltl.combot.infra.topic_worker import TopicWorker
from cltl_service.emissordata.client import EmissorDataClient
from emissor.representation.scenario import TextSignal

from cltl.reply_generation.api import BasicReplier

logger = logging.getLogger(__name__)

CONTENT_TYPE_SEPARATOR = ';'


class ReplyGenerationService:
    @classmethod
    def from_config(cls, repliers: List[BasicReplier], emissor_data: EmissorDataClient, event_bus: EventBus, resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.reply_generation")

        return cls(config.get("topic_input"), config.get("topic_output"),
                   config.get("intentions", multi=True), config.get("topic_intention"),
                   repliers, emissor_data, event_bus, resource_manager)

    def __init__(self, input_topic: str, output_topic: str, intentions: Iterable[str], intention_topic: str,
                 repliers: List[BasicReplier], emissor_data: EmissorDataClient,
                 event_bus: EventBus, resource_manager: ResourceManager):
        self._repliers = repliers

        self._emissor_data = emissor_data
        self._event_bus = event_bus
        self._resource_manager = resource_manager

        self._input_topic = input_topic
        self._output_topic = output_topic
        self._intentions = intentions
        self._intention_topic = intention_topic

        self._topic_worker = None

    @property
    def app(self):
        return None

    def start(self, timeout=30):
        self._topic_worker = TopicWorker([self._input_topic], self._event_bus, provides=[self._output_topic],
                                         resource_manager=self._resource_manager, processor=self._process,
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
        reply_list = []
        for brain_response in event.payload:
            logger.debug("Brain response: (%s)", brain_response)
            try:
                brain_response = brain_response_to_json(brain_response)
                for replier in self._repliers:
                    reply = None
                    if 'statement' in brain_response:
                        reply = replier.reply_to_statement(brain_response, persist=True)
                    elif 'question' in brain_response:
                        reply = replier.reply_to_question(brain_response)
                    elif 'mention' in brain_response:
                        reply = replier.reply_to_statement(brain_response, entity_only=True, persist=True)
                    if reply:
                        reply_list.append(reply)
                        break
            except:
                logger.exception("Replier error on brain response %s", brain_response)

        response = '. '.join(set(reply_list))

        if response:
            extractor_event = self._create_payload(response)
            self._event_bus.publish(self._output_topic, Event.for_payload(extractor_event))

    def _create_payload(self, response):
        scenario_id = self._emissor_data.get_current_scenario_id()
        signal = TextSignal.for_scenario(scenario_id, timestamp_now(), timestamp_now(), None, response)

        return TextSignalEvent.for_agent(signal)
