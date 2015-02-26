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

"""
This module contains `DependencyParse`, the representation of a (possibly
partial) dependency parse.
"""

from .move import Move

from nlup import tupleify, DependencyParsedSentence


NYL = "*NYL*"
ROOT_LABEL = "*ROOT*"

CLIP = 8  # clip numerical counts at this value


class DependencyParse(object):

    moves = list(Move)

    def __init__(self, tokens, tags, heads, labels):
        L = len(tokens)
        assert L == len(tags) == len(heads) == len(labels)
        self.tokens = (ROOT_LABEL,) + tuple(tokens)
        self.tags = (ROOT_LABEL,) + tuple(tags)
        self.heads = [-L] + list(heads)
        self.labels = [NYL] + list(labels)
        self.stack = [0]
        self.q0 = 1  # index of front of queue
        self.ldeps = tuple([] for _ in range(len(self)))
        self.rdeps = tuple([] for _ in range(len(self)))

    @property
    def queue(self):
        """
        The queue here is implicitly defined by `self.q0` and the
        length of the instance, but sometimes we want the whole thing
        """
        return range(self.q0, len(self))

    @property
    def s0(self):
        """
        The stack is explicitly defined by `self.stack`, but sometimes
        we just want the top of the stack
        """
        return self.stack[-1]

    # convert between formats

    @classmethod
    def from_scratch(cls, tokens, tags):
        L = len(tokens)
        heads = [-L for _ in range(L)]
        labels = [NYL for _ in range(L)]
        retval = cls(tokens, tags, heads, labels)
        return retval

    @classmethod
    def from_DPS(cls, dps):
        """
        DependencyParsedSentence -> DependencyParse
        """
        return cls(dps.tokens, dps.tags, dps.heads, dps.labels)

    def to_DPS(self):
        """
        DependencyParse -> DependencyParsedSentence, for pretty-printing
        """
        return DependencyParsedSentence(self.get_tokens(),
                                        self.get_tags(),
                                        self.get_heads(),
                                        self.get_labels())

    def __repr__(self):
        return "{}(tokens={!r}, tags={!r}, stack={!r}, queue={!r}, heads={!r}, labels={!r})".format(self.__class__.__name__, self.tokens, self.tags,
                                                                                                    self.stack, self.queue, self.heads, self.labels)

    def __len__(self):
        return len(self.tokens)

    # getters for non-parsing purposes, ignoring the root dummy symbol

    def get_tokens(self):
        return self.tokens[1:]

    def get_tags(self):
        return self.tags[1:]

    def get_heads(self):
        return tuple(self.heads[1:])

    def get_labels(self):
        return tuple(self.labels[1:])

    # utilities

    @property
    def EOP(self):
        """
        True iff the parse is complete (i.e., the only symbol on the stack
        is ROOT, and the queue is empty)
        """
        return self.depth == 1 and self.q0 >= len(self)

    @property
    def depth(self):
        return len(self.stack)

    def add(self, head, dep, label=NYL):
        """
        Add an arc `head` -> `dep` (with an optional `label`) to the 
        current parse
        """
        self.heads[dep] = head
        if dep < head:
            self.ldeps[head].append(dep)
        else:
            self.rdeps[head].append(dep)
        self.labels[dep] = label

    def apply_move(self, move, label=NYL):
        """
        Apply the given `move` to the current parse configuration; if 
        `label` is specified and the move is a reduce operation, the arc 
        takes on a label
        """
        if move == Move.shift:
            # in this case, label is irrelevant
            self.stack.append(self.q0)
            self.q0 += 1
        else:
            if move == Move.rreduce:
                # in the standard (and hybrid) system:
                # ... wi, wj ] [ ... -> ... wi ] [ ... , adding wi -> wj
                self.add(self.stack[-2], self.stack.pop(), label)
            elif move == Move.lreduce:
                # in the eager (and hybrid) system:
                # ... wi ] [ wj ... -> ... ] [ wj ... , adding wj -> wi
                self.add(self.queue[0], self.stack.pop(), label)
            else:
                raise ValueError("Unknown move '{!r}'.".format(move))

    @tupleify
    def valid_moves(self):
        """
        Get valid moves for the current parse configuration
        """
        if self.queue:
            yield Move.shift
            if self.depth >= 1:
                yield Move.lreduce
        if self.depth >= 2:
            yield Move.rreduce

    # feature extraction

    @tupleify
    def features(self, clip=CLIP):
        """
        Extract features from the current parse configuration
        """
        utokens = [token.upper() for token in self.tokens]
        yield "*bias*"
        depth = "depth={}".format(min(self.depth, clip))
        yield depth
        # string distance between top of stack and top of queue
        gap = "gap={}".format(min(self.q0 - self.s0, clip))
        yield gap
        # stack and queue, tokens and tags
        slen = min(self.depth, 3)
        qlen = min(len(self.queue) - 1, 3)
        # we index the stack backwards so that `s_0` is the TOP
        tstack = self.stack[-slen:][::-1]
        tqueue = self.queue[:qlen]
        sw = ["s_{}:w='{}'".format(i, utoken) for (i, utoken) in
              enumerate(utokens[i] for i in tstack)]
        st = ["s_{}:t='{}'".format(i, tag) for (i, tag) in 
              enumerate(self.tags[i] for i in tstack)]
        qw = ["q_{}:w='{}'".format(i, utoken) for (i, utoken) in
              enumerate(utokens[i] for i in tqueue)]
        qt = ["q_{}:t='{}'".format(i, tag) for (i, tag) in 
              enumerate(self.tags[i] for i in tqueue)]
        # leftmost children of top of stack, tokens and tags
        ls0 = self.ldeps[self.s0]
        lsv = "|s_0:ldeps|={}".format(min(len(ls0), clip))
        yield lsv
        ls0 = ls0[:2]
        lsw = ["s_0:ldep_{}:w='{}'".format(i, utoken) for (i, utoken) in
               enumerate(utokens[i] for i in ls0)]
        lst = ["s_0:ldep_{}:t='{}'".format(i, tag) for (i, tag) in 
               enumerate(self.tags[i] for i in ls0)]
        # rightmost children of top of stack
        rs0 = self.rdeps[self.s0]
        rsv = "|s_0:rdeps|={}".format(min(len(rs0), clip))
        yield rsv
        rs0 = rs0[-2:]
        rsw = ["s_0:rdep_{}:w='{}'".format(i, utoken) for (i, utoken) in
               enumerate(utokens[i] for i in rs0)]
        rst = ["s_0:rdep_{}:t='{}'".format(i, tag) for (i, tag) in 
               enumerate(self.tags[i] for i in rs0)]
        # leftmost children of top of queue
        lq0 = self.ldeps[self.q0]
        lqv = "|q_0:ldeps|={}".format(min(len(lq0), clip))
        yield lqv
        lq0 = lq0[:2]
        lqw = ["q_0:ldep_{}:w='{}'".format(i, utoken) for (i, utoken) in
               enumerate(utokens[i] for i in lq0)]
        lqt = ["q_0:ldep_{}:t='{}'".format(i, tag) for (i, tag) in
               enumerate(self.tags[i] for i in lq0)]
        # NB: by definition, q0 does not yet have any rdeps
        # output all lists of word and tag features thus far
        for feat in sw + st + qw + qt + lsw + lst + rsw + rst + lqw + lqt:
            yield feat
        # word/tag unigrams from the queue
        for (w, t) in zip(qw, qt):
            yield "{}/{}".format(w, t)
        # word/tag unigrams from the stack
        for (w, t) in zip(sw, st):
            yield "{}/{}".format(w, t)
        # valence and gap bigrams and trigrams
        yield "{},{}".format(sw[-1], lsv)
        yield "{},{}".format(st[-1], lsv)
        yield "{},{}".format(sw[-1], rsv)
        yield "{},{}".format(st[-1], rsv)
        yield "{},{}".format(sw[-1], gap)
        yield "{},{}".format(st[-1], gap)
        if qw: # and thus, qt
            # bigrams
            yield "{},{}".format(sw[-1], qw[0])
            yield "{}/{},{}".format(qw[0], qt[0], sw[-1])
            yield "{}/{},{}".format(qw[0], qt[0], st[-1])
            yield "{}/{},{}".format(sw[-1], st[-1], qw[0])
            yield "{}/{},{}".format(sw[-1], st[-1], qt[0])
            yield "{}/{},{}/{}".format(sw[-1], st[-1], qw[0], qt[0])
            yield "{},{}".format(st[-1], qt[0])
            if len(qt) > 1:
                yield "{},{}".format(qt[0], qt[1])
            # tag trigrams
            if len(st) >= 2:
                yield "{},{},{}".format(st[-1], st[-2], qt[0])
            if len(qt) >= 2:
                yield "{},{},{}".format(st[-1], qt[0], qt[1])
                if len(qt) == 3:
                    yield "{},{},{}".format(*qt)
            if lst:
                yield "{},{},{}".format(st[-1], lst[0], qt[0])
            if rst:
                yield "{},{},{}".format(st[-1], rst[0], qt[0])
            if lqt:
                yield "{},{},{}".format(st[-1], qt[0], lqt[0])
                if len(lqt) == 2:
                    yield "{},{},{}".format(qt[0], lqt[0], lqt[1])
            # valence and gap bigrams and trigrams
            yield "{},{}".format(qw[0], lqv)
            yield "{},{}".format(qt[0], lqv)
            yield "{},{}".format(qw[0], gap)
            yield "{},{}".format(qt[0], gap)
            yield "{},{},{}".format(sw[-1], qw[0], gap)
            yield "{},{},{}".format(st[-1], qt[0], gap)
        if len(lst) == 2:
            yield "{},{},{}".format(st[-1], lst[0], lst[1])
        if len(rst) == 2:
            yield "{},{},{}".format(st[-1], rst[0], rst[1])
        if len(st) == 3:
            yield "{},{},{}".format(*reversed(st))
