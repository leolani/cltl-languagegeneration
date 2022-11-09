from cltl.reply_generation import logger


class ThoughtSelector(object):

    def select(self, options):
        raise NotImplementedError()


class Phraser(object):

    def phrase_correct_thought(self, utterance, thought_type, thought_info, fallback=True):
        raise NotImplementedError()


class BasicReplier(object):

    def __init__(self):
        # type: () -> None
        """
        Generate natural language based on structured data

        Parameters
        ----------
        """

        self._log = logger.getChild(self.__class__.__name__)
        self._log.info("Booted")
        self._thought_selector = ThoughtSelector()
        self._phraser = Phraser()

    @property
    def thought_selector(self):
        return self._thought_selector

    @property
    def phraser(self):
        return self._phraser

    def reply_to_question(self, brain_response):
        raise NotImplementedError()

    def reply_to_statement(self, brain_response, entity_only=False, proactive=True, persist=False):
        raise NotImplementedError()
