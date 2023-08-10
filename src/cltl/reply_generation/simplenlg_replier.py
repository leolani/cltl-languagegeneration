from cltl.reply_generation.lenka_replier import LenkaReplier
from cltl.reply_generation.phrasers.simplenlg_phraser import SimplenlgPhraser


class SimpleNLGReplier(LenkaReplier):

    def __init__(self):
        # type: () -> None
        """
        Generate natural language based on structured data

        Parameters
        ----------
        """

        super(SimpleNLGReplier, self).__init__()

        self._phraser = SimplenlgPhraser()
        self._log.debug(f"SimpleNLG phraser ready")
