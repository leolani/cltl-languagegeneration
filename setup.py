from setuptools import setup, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("VERSION", "r") as fh:
    version = fh.read().strip()

setup(
    name="cltl.reply_generation",
    description="The Leolani Language module for reply generation",
    version=version,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leolani/cltl-languagegeneration",
    license='MIT License',
    authors={
        "Bajc̆etic": ("Lenka Bajc̆etic", "l.bajccetic@vu.nl"),
        "Baez Santamaria": ("Selene Baez Santamaria", "s.baezsantamaria@vu.nl"),
        "Baier": ("Thomas Baier", "t.baier@vu.nl")
    },
    package_dir={'': 'src'},
    packages=find_namespace_packages(include=['cltl.*'], where='src'),
    package_data={'cltl.reply_generation': ['data/*']},
    python_requires='>=3.7',
    install_requires=[
        'nltk~=3.4.4',
        'tqdm==4.62.3'
    ],
    setup_requires=['flake8']
)
