import collections.abc
import gzip
import numpy as np


class TreebankNode(object):
    pass


class InternalTreebankNode(TreebankNode):
    def __init__(self, label, children):
        assert isinstance(label, str)
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, TreebankNode) for child in children)
        assert children

        self.left = children[0].left
        self.right = children[-1].right

        self.children = tuple(children)

    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self, index=0, nocache=False):
        tree = self
        sublabels = [self.label]

        while len(tree.children) == 1 and isinstance(
                tree.children[0], InternalTreebankNode):
            tree = tree.children[0]
            sublabels.append(tree.label)

        children = []
        for child in tree.children:
            children.append(child.convert(index=index))
            index = children[-1].right

        return InternalParseNode(tuple(sublabels), children, nocache=nocache)


class LeafTreebankNode(TreebankNode):
    def __init__(self, word_idx, tag, word, lbound, rbound, llabel, rlabel):
        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

        self.left = word_idx
        self.right = word_idx + 1

        self.lbound = lbound
        self.rbound = rbound

        self.llabel = llabel
        self.rlabel = rlabel

    def linearize(self):
        return "({} {})".format(self.tag, self.word)

    def leaves(self):
        yield self

    def convert(self, index=0):
        return LeafParseNode(index, self.tag, self.word, self.lbound, self.rbound, self.llabel, self.rlabel)


class ParseNode(object):
    pass


class InternalParseNode(ParseNode):
    def __init__(self, label, children, nocache=False):
        assert isinstance(label, tuple)
        assert all(isinstance(sublabel, str) for sublabel in label)
        assert label
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, ParseNode) for child in children)
        assert children
        assert len(children) > 1 or isinstance(children[0], LeafParseNode)
        assert all(
            left.right == right.left
            for left, right in zip(children, children[1:]))
        self.children = tuple(children)

        self.left = children[0].left
        self.right = children[-1].right

        self.nocache = nocache

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label[-1], children)
        for sublabel in reversed(self.label[:-1]):
            tree = InternalTreebankNode(sublabel, [tree])
        return tree

    def enclosing(self, left, right):
        assert self.left <= left < right <= self.right
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing(left, right)
        return self

    def oracle_label(self, left, right):
        enclosing = self.enclosing(left, right)
        if enclosing.left == left and enclosing.right == right:
            return enclosing.label
        return ()

    def oracle_splits(self, left, right):
        return [
            child.left
            for child in self.enclosing(left, right).children
            if left < child.left < right
        ]


class LeafParseNode(ParseNode):
    def __init__(self, index, tag, word, lbound, rbound, llabel, rlabel):
        assert isinstance(index, int)
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, str)
        self.tag = tag

        self.lbound = lbound
        self.rbound = rbound

        self.llabel = llabel
        self.rlabel = rlabel

        assert isinstance(word, str)
        self.word = word

    def leaves(self):
        yield self

    def chil_enclosing(self, left, right):
        assert self.left <= left < right <= self.right
        return self

    def convert(self):
        return LeafTreebankNode(self.left, self.tag, self.word, self.lbound, self.rbound, self.llabel, self.rlabel)


def load_trees(path, strip_top=True):
    with open(path) as infile:
        treebank = infile.read()

    tokens = treebank.replace("(", " ( ").replace(")", " ) ").split()
    cun_sent = 0
    word_idx = 0

    def helper(index, flag_sent):
        nonlocal cun_sent
        nonlocal word_idx
        trees = []

        while index < len(tokens) and tokens[index] == "(":
            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]

            if label != '-NONE-' and '-' in label:
                label = label.split('-')[0]

            index += 1

            if tokens[index] == "(":
                children, index = helper(index, flag_sent=0)
                if len(children) > 0:
                    tr = InternalTreebankNode(label, children)
                    trees.append(tr)
            else:
                word = tokens[index]
                index += 1
                if label != '-NONE-':
                    trees.append(LeafTreebankNode(
                        word_idx, label, word, lbound=0, rbound=0, llabel=(), rlabel=()))
                    word_idx += 1

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

            if flag_sent == 1:
                cun_sent += 1
                word_idx = 0

        return trees, index

    trees, index = helper(0, flag_sent=1)
    assert index == len(tokens)
    assert len(trees) == cun_sent

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label in ("TOP", "ROOT"):
                assert len(tree.children) == 1
                trees[i] = tree.children[0]

    def process_NONE(tree):
        if isinstance(tree, LeafTreebankNode):
            label = tree.tag
            if label == '-NONE-':
                return None
            else:
                return tree

        tr = []
        label = tree.label
        if label == '-NONE-':
            return None
        for node in tree.children:
            new_node = process_NONE(node)
            if new_node is not None:
                tr.append(new_node)
        if tr == []:
            return None
        else:
            return InternalTreebankNode(label, tr)

    def get_lbound(tree, is_right_child=False):
        if not isinstance(tree, LeafTreebankNode) and len(tree.children) == 1:
            return get_lbound(tree.children[0], is_right_child)

        lbounds = []
        l = tree.left
        r = tree.right
        if not is_right_child:
            lbounds = [l]

        if isinstance(tree, LeafTreebankNode):
            return lbounds

        tmp = []
        for c in tree.children[:-1]:
            tmp = tmp + get_lbound(c, False)
        lbounds = tmp + get_lbound(tree.children[-1], True) + lbounds
        return lbounds

    def get_rbound(lbound, left, right):
        lbound = [-1] + lbound
        rbound = []
        stack_i = [0] * (2 * len(lbound) + 5)
        stack_j = [0] * (2 * len(lbound) + 5)
        stack_f = [0] * (2 * len(lbound) + 5)
        stack_idx = 1
        idx = 0
        stack_i[1] = 0
        stack_j[1] = len(lbound) - 1
        stack_f[1] = 1

        while stack_idx > 0:
            i = stack_i[stack_idx]
            j = stack_j[stack_idx]
            f = stack_f[stack_idx]
            if f == 1:
                rbound.append(j)
            stack_idx -= 1
            idx += 1
            if i + 1 < j:
                for k in range(j - 1, i, -1):
                    if lbound[k] <= i:
                        stack_idx += 1
                        stack_i[stack_idx] = k
                        stack_j[stack_idx] = j
                        stack_f[stack_idx] = 1
                        stack_idx += 1
                        stack_i[stack_idx] = i
                        stack_j[stack_idx] = k
                        stack_f[stack_idx] = 0
                        break
        return rbound

    def process_bound(root_tree, tree, lbounds, rbounds):
        if isinstance(tree, LeafTreebankNode):
            return LeafTreebankNode(tree.left, tree.tag, tree.word, lbounds[tree.left], rbounds[tree.left], root_tree.oracle_label(lbounds[tree.left], tree.right), root_tree.oracle_label(tree.left, rbounds[tree.left]))

        tr = []
        label = tree.label
        for node in tree.children:
            new_node = process_bound(root_tree, node, lbounds, rbounds)
            if new_node is not None:
                tr.append(new_node)
        if tr == []:
            return None
        else:
            return InternalTreebankNode(label, tr)

    new_trees = []
    for i, tree in enumerate(trees):
        new_tree = process_NONE(tree)
        lbounds = get_lbound(new_tree)
        rbounds = get_rbound(lbounds, 0, len(lbounds))
        new_tree = process_bound(
            new_tree.convert(), new_tree, lbounds, rbounds)
        new_trees.append(new_tree)

    return new_trees
