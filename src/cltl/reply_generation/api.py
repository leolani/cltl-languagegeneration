from cltl.reply_generation import logger


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

    def reply_to_question(self, brain_response):
        raise NotImplementedError()

    def reply_to_statement(self, brain_response):
        raise NotImplementedError()
