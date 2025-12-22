from __future__ import annotations
from dataclasses import dataclass
from math import log, exp, sqrt, sin, cos, erf, pi


@dataclass
class fltptr:
    v: float


@dataclass
class nodeptr:
    n: node


@dataclass
class node:
    arity: int
    adjoint: fltptr
    partials: list[float]
    adjoints: list[fltptr]

    def propagate(self):
        if self.arity == 0 or self.adjoint.v == 0.00:
            return
        for i in range(self.arity):
            self.adjoints[i].v += self.partials[i] * self.adjoint.v


@dataclass
class tape:
    nodes: list[node]

    def seed(self) -> None:
        self.nodes[-1].adjoint.v = 1.00

    def backprop(self) -> None:
        for n in reversed(self.nodes):
            n.propagate()

    def clear(self) -> None:
        self.nodes = []


TAPE = tape(nodes=[])


def from_nodes(value: float, nodes: list[nodeptr]) -> number:
    n = node(len(nodes), fltptr(0.00), [], [])

    if len(nodes) > 0:
        n.adjoints = [fltptr(0.00) for _ in range(len(nodes))]
        for i in range(len(nodes)):
            n.adjoints[i] = nodes[i].n.adjoint
        n.partials = [0.00 for _ in range(len(nodes))]

    TAPE.nodes.append(n)
    nptr = nodeptr(n)
    num = number(value, nptr)
    return num


def new_leaf(value: float) -> number:
    np = nodeptr(node(0, fltptr(0.00), [], []))
    return number(value, np)


@dataclass
class number:
    value: float
    node: nodeptr

    def __add__(self, other: number | float | int) -> number:
        if type(other) == number:
            return self.add(other)
        elif type(other) == int or type(other) == float:
            return self.add_scalar(other)
        assert False

    def __sub__(self, other: number | float | int) -> number:
        if type(other) == number:
            return self.sub(other)
        elif type(other) == int or type(other) == float:
            return self.add_scalar(-other)
        assert False

    def __mul__(self, other: number | float | int) -> number:
        if type(other) == number:
            return self.mul(other)
        elif type(other) == int or type(other) == float:
            return self.mul_scalar(other)
        assert False

    def __truediv__(self, other: number | float | int) -> number:
        if type(other) == number:
            return self.div(other)
        elif type(other) == int or type(other) == float:
            return self.div_scalar(other)
        assert False

    def __rmul__(self, other: float | int) -> number:
        return self * other

    def __neg__(self) -> number:
        num = from_nodes(-(self.value), [self.node])
        num.node.n.partials[0] = -1.00
        return num

    def __inv__(self) -> number:
        num = from_nodes(1 / (self.value), [self.node])
        num.node.n.partials[0] = -1 / self.value**2
        return num

    def add(self, other: number) -> number:
        num = from_nodes(self.value + other.value, [self.node, other.node])
        num.node.n.partials[0] = 1.00
        num.node.n.partials[1] = 1.00
        return num

    def sub(self, other: number) -> number:
        num = from_nodes(self.value - other.value, [self.node, other.node])
        num.node.n.partials[0] = 1.00
        num.node.n.partials[1] = -1.00
        return num

    def div(self, other: number) -> number:
        num = from_nodes(self.value / other.value, [self.node, other.node])
        num.node.n.partials[0] = 1.00 / other.value
        num.node.n.partials[1] = -self.value / (other.value**2)
        return num

    def mul(self, other: number) -> number:
        num = from_nodes(self.value * other.value, [self.node, other.node])
        num.node.n.partials[0] = other.value
        num.node.n.partials[1] = self.value
        return num

    def exp(self) -> number:
        num = from_nodes(exp(self.value), [self.node])
        num.node.n.partials[0] = exp(self.value)
        return num

    def log(self) -> number:
        num = from_nodes(log(self.value), [self.node])
        num.node.n.partials[0] = 1.00 / self.value
        return num

    def sin(self) -> number:
        num = from_nodes(sin(self.value), [self.node])
        num.node.n.partials[0] = cos(self.value)
        return num

    def cos(self) -> number:
        num = from_nodes(cos(self.value), [self.node])
        num.node.n.partials[0] = -sin(self.value)
        return num

    def sqrt(self) -> number:
        num = from_nodes(sqrt(self.value), [self.node])
        num.node.n.partials[0] = 0.50 / sqrt(self.value)
        return num

    def square(self) -> number:
        num = from_nodes(self.value**2, [self.node])
        num.node.n.partials[0] = 2.00 * self.value
        return num

    def add_scalar(self, other: float | int) -> number:
        num = from_nodes(self.value + other, [self.node])
        num.node.n.partials[0] = 1.00
        return num

    def div_scalar(self, other: float | int) -> number:
        num = from_nodes(self.value / other, [self.node])
        num.node.n.partials[0] = 1.00 / other
        return num

    def mul_scalar(self, other: float | int) -> number:
        num = from_nodes(self.value * other, [self.node])
        num.node.n.partials[0] = other
        return num

    def norm_cdf(self) -> number:
        num = from_nodes(normal_cdf(self.value), [self.node])
        num.node.n.partials[0] = normal_pdf(self.value)
        return num

    def norm_pdf(self) -> number:
        num = from_nodes(normal_pdf(self.value), [self.node])
        num.node.n.partials[0] = -self.value * normal_pdf(self.value)
        return num

    def backprop(self) -> None:
        self.node.n.propagate()

    def max(self, other: number | float | int) -> number:
        if type(other) == number:
            num = from_nodes(max(self.value, other.value), [self.node, other.node])
            if self.value > other.value:
                num.node.n.partials[0] = 1.00
                num.node.n.partials[1] = 0.00
            else:
                num.node.n.partials[0] = 0.00
                num.node.n.partials[1] = 1.00
            return num

        elif type(other) == int or type(other) == float:
            num = from_nodes(max(self.value, other), [self.node])
            if self.value > other:
                num.node.n.partials[0] = 1.00
            else:
                num.node.n.partials[0] = 0.00
            return num
        raise TypeError(
            f"cannot compute max(number, other) for type(other)={type(other)}"
        )


def normal_cdf(v: float) -> float:
    return (1 + erf(v / sqrt(2))) / 2.00


def normal_pdf(v: float) -> float:
    return 1 / sqrt(2 * pi) * exp(-(v**2) / 2)
