Where's Yr Head At
==================

Where's Yr Head At (WYHA) is an implementation of a unlabeled dependency
parser using arc-hybrid transitions, a dynamic oracle, the averaged
perceptron classifier, and greedy search.

The arc-hybrid transition system was originally described by Kuhlmann
et al. 2011. The dynamic oracle framework was proposed by Goldberg &
Nivre 2013. The classifier features are based on those proposed by
Matthew Honnibal in a recent
[blog post](https://honnibal.wordpress.com/2013/12/18/a-simple-fast-algorithm-for-natural-language-dependency-parsing/), though I have attempted to
correct what I believe to be a few typos there.

WYHA is not designed to be particularly fast or to outperform state of
the art systems. Rather, the focus is on clean design of the sort useful
for teaching and research.

WYHA has been tested on CPython 3.4 and PyPy3 (2.4.0, corresponding to
Python 3.2). It requires three third-party packages: `nltk` and
`jsonpickle` from PyPI and my own `nlup` library, available from GitHub;
see `requirements.txt` for the versions used for testing.


Usage
-----

    python -m wheresyrheadat [-h] [-v | -V] (-r READ | -t TRAIN)
                             (-e EVALUATE | -p PARSE | -w WRITE)
                             [-E EPOCHS]

    A greedy arc-hybrid dependency parser

    optional arguments:

    -h, --help                          show this help message and exit
    -v, --verbose                       enable verbose output
    -V, --really-verbose                enable even more verbose output
    -r READ, --read READ                read in serialized model
    -t TRAIN, --train TRAIN             training data
    -e EVALUATE, --evaluate EVALUATE    evaluate on labeled data
    -p PARSE, --parse PARSE             parse tagged sentences
    -w WRITE, --write WRITE             write out serialized model
    -E EPOCHS, --epochs EPOCHS          # of epochs (default: 10)

For anything else, UTSL.

License
-------

MIT License (BSD-like); see source.

What's with the name?
---------------------

A transition-based dependency parser is fundamentally trying to answer
the question "where's your head at?" for every word. The name is also a
tribute to electronic music duo
[Basement Jaxx](http://www.basementjaxx.co.uk)'s 2001 hit single
["Where's Your Head At"](https://www.youtube.com/watch?v=5rAOyh7YmEc).

Bugs, comments?
---------------

Contact [Kyle Gorman](mailto:gormanky@ohsu.edu).

References
----------

M. Kuhlmann, C. Gómez-Rodríguez, and G. Satta. 2011. Dynamic programming
algorithms for transition-based dependency parsers. In _ACL_, 673-682.

Y. Goldberg and J. Nivre. 2013. Training deterministic parsers with
non-deterministic oracles. _Transactions of the Association for
Computational Linguistics_ 1: 403-414.
