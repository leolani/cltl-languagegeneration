import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('\r%(asctime)s - %(levelname)8s - %(name)40s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class BasicReplier(object):

    def __init__(self):
        # type: () -> None
        """
        Generate natural language based on structured data

        Parameters
        ----------
        """

        self._log = logger.getChild(self.__class__.__name__)
        self._log.debug("Booted")

    def reply_to_question(self, brain_response):
        raise NotImplementedError()

    def reply_to_statement(self, brain_response):
        raise NotImplementedError()
