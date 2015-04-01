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

from random import Random

from nlup import Accuracy, AveragedPerceptron, JSONable, Timer

from .move import Move
from .depparse import DependencyParse


EPOCHS = 10

MOVES = tuple(Move)


class DependencyParser(JSONable):

    """
    Arc-hybrid shift-reduce greedy dependency parser, described by 
    Goldberg & Nivre
    """

    def __init__(self, seed=None):
        self.random = Random(seed)
        self.classifier = AveragedPerceptron(seed=seed)
        self.classifier.register_classes(MOVES)

    def parse(self, tokens, tags):
        """
        Construct a `DependencyParse` using ordered containers of 
        `tokens` and `tags` and the current move classifier model
        """
        parse = DependencyParse.from_scratch(tokens, tags)
        while not parse.EOP:
            if not parse.stack:
                logging.debug("Performing mandatory SHIFT.")
                parse.apply_move(Move.shift)
                continue
            if not parse.queue:
                logging.debug("Performing mandatory RREDUCE.")
                parse.apply_move(Move.rreduce)
                continue
            # otherwise, use classifier to predict
            valid_moves = parse.valid_moves()
            phi = parse.features()
            scores = self.classifier.scores(phi)
            yhat = max(valid_moves, key=lambda move: scores[move])
            parse.apply_move(yhat)
        return parse

    def fit_one(self, gold, alpha=1):
        guess = DependencyParse.from_scratch(gold.get_tokens(),
                                             gold.get_tags())
        while not guess.EOP:
            if not guess.stack:
                logging.debug("Performing mandatory SHIFT.")
                guess.apply_move(Move.shift)
                continue
            if not guess.queue:
                logging.debug("Performing mandatory RREDUCE.")
                guess.apply_move(Move.rreduce)
                continue
            # otherwise, use classifier to predict
            valid_moves = guess.valid_moves()
            phi = guess.features()
            scores = self.classifier.scores(phi)
            yhat = max(valid_moves, key=lambda move: scores[move])
            # use gold parse to get true move
            gold_moves = DependencyParser.gold_moves(valid_moves,
                                                     guess, gold)
            if not gold_moves:
                logging.debug("Premature termination (no gold moves).")
                return guess
            y = max(gold_moves, key=lambda move: scores[move])
            # update weights if we made the wrong guess
            if y != yhat:
                self.classifier.update(y, yhat, phi, alpha)
            # apply predicted move
            guess.apply_move(yhat)
            # apply
            self.classifier.time += 1
        logging.debug("Final parse:\t{!r}".format(guess))
        return guess

    def fit(self, golds, epochs, alpha=1):
        golds = list(DependencyParse.from_DPS(gold) for gold in golds)
        for i in range(1, 1 + epochs):
            logging.info("Epoch {:>2}.".format(i))
            cx = Accuracy()
            with Timer():
                self.random.shuffle(golds)
                for gold in golds:
                    guess = self.fit_one(gold, alpha)
                    cx.batch_update(gold.heads, guess.heads)
            logging.info("Accuracy: {:.4f}.".format(cx.accuracy))
        self.classifier.finalize()

    @staticmethod
    def gold_moves(valid_moves, guess, gold):
        """
        The arc-hybrid dynamic oracle comes from Goldberg & Nivre 2013
        (TACL) where it is described in detail. The operations are:

        * Shift shifts the leftmost element off the queue and pushes it
          onto the stack.
        * L-reduce adds an arc stack[-1] <- queue[0] and pops the top
          of the stack.
        * R-reduce adds an arc stack[-2] -> stack[-1] and pops the top
          of the stack.
        """
        s0_head = gold.heads[guess.s0]
        q0_head = gold.heads[guess.q0]
        gold_moves = set(valid_moves)
        # if the top of the stack is the head of the front of the queue,
        # we need to shift
        if Move.shift in gold_moves and q0_head == guess.s0:
            return {Move.shift}
        if Move.lreduce in gold_moves:
            # if the top of the queue is the head of the stack,
            # we need to left-reduce
            if s0_head == guess.q0:
                return {Move.lreduce}
            # but if the second-highest element in the stack is the head
            # of top of the stack, we must not left-reduce
            if len(guess.stack) >= 2 and \
                   s0_head == gold.heads[guess.stack[-2]]:
                gold_moves.remove(Move.lreduce)
        if Move.shift in gold_moves:
            # if there are any dependencies between the front of the queue
            # and the stack (other than the mandatory left-reduce one
            # we already considered), we must not shift
            if any(q0_head == i for i in guess.stack[:-1]) or \
               any(gold.heads[i] == guess.q0 for i in guess.stack):
                gold_moves.remove(Move.shift)
        # if there are any dependencies between the top of the stack
        # and the queue, popping the stack will lose them, so we must
        # not reduce
        if any(s0_head == i for i in guess.queue[1:]) or \
           any(gold.heads[i] == guess.s0 for i in guess.queue):
            gold_moves.discard(Move.lreduce)
            gold_moves.discard(Move.rreduce)
        return gold_moves
