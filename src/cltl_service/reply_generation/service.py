import logging
from typing import List

from cltl.brain.utils.helper_functions import brain_response_to_json
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.time_util import timestamp_now
from cltl.combot.infra.topic_worker import TopicWorker
from cltl.reply_generation.api import BasicReplier
from cltl_service.backend.schema import TextSignalEvent
from emissor.representation.scenario import TextSignal

logger = logging.getLogger(__name__)

CONTENT_TYPE_SEPARATOR = ';'


class ReplyGenerationService:
    @classmethod
    def from_config(cls, repliers: List[BasicReplier], event_bus: EventBus, resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.reply_generation")

        return cls(config.get("topic_input"), config.get("topic_output"), repliers, event_bus, resource_manager)

    def __init__(self, input_topic: str, output_topic: str, repliers: List[BasicReplier],
                 event_bus: EventBus, resource_manager: ResourceManager):
        self._repliers = repliers

        self._event_bus = event_bus
        self._resource_manager = resource_manager

        self._input_topic = input_topic
        self._output_topic = output_topic

        self._topic_worker = None

    @property
    def app(self):
        return None

    def start(self, timeout=30):
        self._topic_worker = TopicWorker([self._input_topic], self._event_bus, provides=[self._output_topic],
                                         resource_manager=self._resource_manager, processor=self._process)
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
                    reply = replier.reply_to_statement(brain_response)
                    if reply:
                        break

                reply_list.append(reply)
            except:
                logger.exception("Replier error")

        response = '. '.join(reply_list)

        if response:
            extractor_event = self._create_payload(response)
            self._event_bus.publish(self._output_topic, Event.for_payload(extractor_event))

    def _create_payload(self, response):
        signal = TextSignal.for_scenario(None, timestamp_now(), timestamp_now(), None, response)

        return TextSignalEvent.create(signal)