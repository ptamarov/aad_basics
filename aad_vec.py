"""Path-vectorised version of aad.py: every number holds a numpy array of
values, one per Monte Carlo path, so a simulation records one tape for all
paths instead of one tape per path. Partials are arrays (or scalars) and the
backward sweep is a multiply-add over all paths at once. Adjoints stay per
path, so leaves end with one gradient per path.

The interface follows aad.py: tape.new_leaf, tape.seed, tape.backprop, and
number arithmetic, exp, log, sqrt, max. Constants may be floats or arrays."""

from __future__ import annotations

import numpy as np


class node:
    __slots__ = ("partials", "parents", "adjoint")

    def __init__(self, partials, parents):
        self.partials = partials
        self.parents = parents
        self.adjoint = None  # set on first contribution

    def propagate(self):
        a = self.adjoint
        if a is None:
            return
        for d, p in zip(self.partials, self.parents):
            # d * a is a fresh array, so the parent owns it and can add in place
            if p.adjoint is None:
                p.adjoint = d * a
            else:
                p.adjoint += d * a


class tape:
    __slots__ = ("nodes", "num_paths")

    def __init__(self, num_paths: int):
        self.nodes = []
        self.num_paths = num_paths

    def new_leaf(self, value) -> number:
        return number(value, node((), ()), self)

    def record(self, value, partials, parents) -> number:
        n = node(partials, parents)
        self.nodes.append(n)
        return number(value, n, self)

    def seed(self) -> None:
        self.nodes[-1].adjoint = np.ones(self.num_paths)

    def backprop(self) -> None:
        for n in reversed(self.nodes):
            n.propagate()


class number:
    __slots__ = ("value", "node", "tape")

    def __init__(self, value, nd: node, t: tape):
        self.value = value
        self.node = nd
        self.tape = t

    def __add__(self, other):
        if isinstance(other, number):
            return self.tape.record(
                self.value + other.value, (1.0, 1.0), (self.node, other.node)
            )
        return self.tape.record(self.value + other, (1.0,), (self.node,))

    def __sub__(self, other):
        if isinstance(other, number):
            return self.tape.record(
                self.value - other.value, (1.0, -1.0), (self.node, other.node)
            )
        return self.tape.record(self.value - other, (1.0,), (self.node,))

    def __mul__(self, other):
        if isinstance(other, number):
            return self.tape.record(
                self.value * other.value,
                (other.value, self.value),
                (self.node, other.node),
            )
        return self.tape.record(self.value * other, (other,), (self.node,))

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, number):
            q = self.value / other.value
            return self.tape.record(
                q, (1.0 / other.value, -q / other.value), (self.node, other.node)
            )
        return self.tape.record(self.value / other, (1.0 / other,), (self.node,))

    def __neg__(self):
        return self.tape.record(-self.value, (-1.0,), (self.node,))

    def exp(self):
        e = np.exp(self.value)
        return self.tape.record(e, (e,), (self.node,))

    def log(self):
        return self.tape.record(np.log(self.value), (1.0 / self.value,), (self.node,))

    def sqrt(self):
        s = np.sqrt(self.value)
        return self.tape.record(s, (0.5 / s,), (self.node,))

    def max(self, other: float):
        """max(self, other) for a constant other; the partial is 1 where self
        wins, as in aad.py (ties go to the constant)."""
        win = self.value > other
        return self.tape.record(
            np.where(win, self.value, other), (win.astype(float),), (self.node,)
        )
