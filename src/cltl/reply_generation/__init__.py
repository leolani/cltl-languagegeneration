"""
The Leolani Reply Generator Package contains the tools to generate replies during conversation, given structured data.
"""

import logging

# create logger
LOG_FORMAT = "%(asctime)s - %(levelname)8s - %(name)60s - %(message)s"
LOG_LEVEL = "INFO"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=getattr(logging, LOG_LEVEL),
                    format=LOG_FORMAT,
                    datefmt=LOG_DATEFMT)
logger = logging.getLogger(__name__)

from cltl.reply_generation.lenka_replier import LenkaReplier
