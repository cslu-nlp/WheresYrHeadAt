# Copyright (C) 2015 Kyle Gorman
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import logging

from argparse import ArgumentParser

from nlup import depparsed_corpus, tagged_corpus, Accuracy

from .depparser import DependencyParser, EPOCHS


LOGGING_FMT = "%(message)s"


argparser = ArgumentParser(prog="python -m wheresyrheadat",
                           description="A greedy arc-hybrid dependency parser")
vrb_group = argparser.add_mutually_exclusive_group()
vrb_group.add_argument("-v", "--verbose", action="store_true",
                       help="enable verbose output")
vrb_group.add_argument("-V", "--really-verbose", action="store_true",
                       help="enable even more verbose output")
inp_group = argparser.add_mutually_exclusive_group(required=True)
inp_group.add_argument("-r", "--read", help="read in serialized model")
inp_group.add_argument("-t", "--train", help="training data")
out_group = argparser.add_mutually_exclusive_group(required=True)
out_group.add_argument("-e", "--evaluate",
                       help="evaluate on labeled data")
out_group.add_argument("-p", "--parse", help="parse tagged sentences")
out_group.add_argument("-w", "--write",
                       help="write out serialized model")
argparser.add_argument("-E", "--epochs", type=int, default=EPOCHS,
                       help="# of epochs (default: {})".format(EPOCHS))
args = argparser.parse_args()
# verbosity block
if args.really_verbose:
    logging.basicConfig(format=LOGGING_FMT, level="DEBUG")
elif args.verbose:
    logging.basicConfig(format=LOGGING_FMT, level="INFO")
else:
    logging.basicConfig(format=LOGGING_FMT)  # , level="WARNING")
# input block
parser = None
if args.read:
    logging.info("Reading pretrained parser '{}'.".format(args.read))
    parser = DependencyParser.load(args.read)
elif args.train:
    logging.info("Training model on '{}'.".format(args.train))
    parser = DependencyParser()
    parser.fit(depparsed_corpus(args.train), args.epochs)
# else unreachable
# output block
if args.write:
    logging.info("Writing trained parser to '{}'.".format(args.write))
    parser.dump(args.write)
elif args.parse:
    logging.info("Parsing unparsed data '{}'.".format(args.parse))
    for sentence in tagged_corpus(args.parse):
        print(parser.parse(sentence.tokens, sentence.tags).to_DPS())
        print()
elif args.evaluate:
    logging.info("Evaluating parsed data '{}'.".format(args.evaluate))
    cx = Accuracy()
    for gold in depparsed_corpus(args.evaluate):
        guess = parser.parse(gold.tokens, gold.tags).to_DPS()
        cx.batch_update(gold.heads, guess.heads)
    print("Accuracy: {:.4f} [{:.4f}, {:.4f}].".format(cx.accuracy,
                                                      *cx.confint))
# else unreachable
