# cltl-languagegeneration

The natural language generation service (aka Leolani Reply generator) This service expects structured data and outputs
natural language.

### Description

## Getting started

### Prerequisites

This repository uses Python >= 3.7

Be sure to run in a virtual python environment (e.g. conda, venv, mkvirtualenv, etc.)

### Installation

1. In the root directory of this repo run

    ```bash
    pip install -e .
    python -c "import nltk; nltk.download('wordnet')"
    ```

### Usage

For using this repository as a package different project and on a different virtual environment, you may

- install a published version from PyPI:

    ```bash
    pip install cltl.reply_generation
    ```

- or, for the latest snapshot, run:

    ```bash
    pip install git+git://github.com/leolani/cltl-languagegeneration.git@main
    ```

Then you can import it in a python script as:

    import cltl.reply_generation

### Examples

Please take a look at the example scripts provided to get an idea on how to run and use this package. Each example has a
comment at the top of the script describing the behaviour of the script.

For these example scripts, you need

1. To change your current directory to ./examples/

1. Run some examples (e.g. python test_with_triples.py)
