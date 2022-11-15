from cltl.reply_generation.lenka_replier import LenkaReplier
from cltl.reply_generation.phrasers.simplenlg_phraser import SimplenlgPhraser
from cltl.reply_generation.thought_selectors.random_selector import RandomSelector


class SimpleNLGReplier(LenkaReplier):

    def __init__(self):
        # type: () -> None
        """
        Generate natural language based on structured data

        Parameters
        ----------
        """

        super(SimpleNLGReplier, self).__init__()
        self._thought_selector = RandomSelector()
        self._log.debug(f"Random Selector ready")

        self._phraser = SimplenlgPhraser()
        self._log.debug(f"SimpleNLG phraser ready")
