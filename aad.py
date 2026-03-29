from __future__ import annotations
from dataclasses import dataclass
from math import log, exp, sqrt, sin, cos, erf, pi


@dataclass
class node:
    arity: int
    adjoint: float
    partials: list[float]
    parents: list[node]
    tape: tape

    def propagate(self):
        if self.arity == 0 or self.adjoint == 0.00:
            return
        for i in range(self.arity):
            self.parents[i].adjoint += self.partials[i] * self.adjoint


@dataclass
class tape:
    nodes: list[node]

    def seed(self) -> None:
        self.nodes[-1].adjoint = 1.00

    def backprop(self) -> None:
        for n in reversed(self.nodes):
            n.propagate()

    def clear(self) -> None:
        self.nodes = []

    def from_nodes(self, value: float, nodes: list[node]) -> number:
        n = node(len(nodes), 0.00, [], [], self)

        if len(nodes) > 0:
            n.parents = nodes
            n.partials = [0.00 for _ in range(len(nodes))]

        self.nodes.append(n)
        num = number(value, n)
        return num

    def new_leaf(self, value: float) -> number:
        np = node(0, 0.00, [], [], self)
        return number(value, np)


@dataclass
class number:
    value: float
    node: node

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
        num = self.node.tape.from_nodes(-(self.value), [self.node])
        num.node.partials[0] = -1.00
        return num

    def __inv__(self) -> number:
        num = self.node.tape.from_nodes(1 / (self.value), [self.node])
        num.node.partials[0] = -1 / self.value**2
        return num

    def add(self, other: number) -> number:
        num = self.node.tape.from_nodes(
            self.value + other.value, [self.node, other.node]
        )
        num.node.partials[0] = 1.00
        num.node.partials[1] = 1.00
        return num

    def sub(self, other: number) -> number:
        num = self.node.tape.from_nodes(
            self.value - other.value, [self.node, other.node]
        )
        num.node.partials[0] = 1.00
        num.node.partials[1] = -1.00
        return num

    def div(self, other: number) -> number:
        num = self.node.tape.from_nodes(
            self.value / other.value, [self.node, other.node]
        )
        num.node.partials[0] = 1.00 / other.value
        num.node.partials[1] = -self.value / (other.value**2)
        return num

    def mul(self, other: number) -> number:
        num = self.node.tape.from_nodes(
            self.value * other.value, [self.node, other.node]
        )
        num.node.partials[0] = other.value
        num.node.partials[1] = self.value
        return num

    def exp(self) -> number:
        num = self.node.tape.from_nodes(exp(self.value), [self.node])
        num.node.partials[0] = exp(self.value)
        return num

    def log(self) -> number:
        num = self.node.tape.from_nodes(log(self.value), [self.node])
        num.node.partials[0] = 1.00 / self.value
        return num

    def sin(self) -> number:
        num = self.node.tape.from_nodes(sin(self.value), [self.node])
        num.node.partials[0] = cos(self.value)
        return num

    def cos(self) -> number:
        num = self.node.tape.from_nodes(cos(self.value), [self.node])
        num.node.partials[0] = -sin(self.value)
        return num

    def sqrt(self) -> number:
        num = self.node.tape.from_nodes(sqrt(self.value), [self.node])
        num.node.partials[0] = 0.50 / sqrt(self.value)
        return num

    def square(self) -> number:
        num = self.node.tape.from_nodes(self.value**2, [self.node])
        num.node.partials[0] = 2.00 * self.value
        return num

    def add_scalar(self, other: float | int) -> number:
        num = self.node.tape.from_nodes(self.value + other, [self.node])
        num.node.partials[0] = 1.00
        return num

    def div_scalar(self, other: float | int) -> number:
        num = self.node.tape.from_nodes(self.value / other, [self.node])
        num.node.partials[0] = 1.00 / other
        return num

    def mul_scalar(self, other: float | int) -> number:
        num = self.node.tape.from_nodes(self.value * other, [self.node])
        num.node.partials[0] = other
        return num

    def norm_cdf(self) -> number:
        num = self.node.tape.from_nodes(normal_cdf(self.value), [self.node])
        num.node.partials[0] = normal_pdf(self.value)
        return num

    def norm_pdf(self) -> number:
        num = self.node.tape.from_nodes(normal_pdf(self.value), [self.node])
        num.node.partials[0] = -self.value * normal_pdf(self.value)
        return num

    def backprop(self) -> None:
        self.node.propagate()

    def max(self, other: number | float | int) -> number:
        if type(other) == number:
            num = self.node.tape.from_nodes(
                max(self.value, other.value), [self.node, other.node]
            )
            if self.value > other.value:
                num.node.partials[0] = 1.00
                num.node.partials[1] = 0.00
            else:
                num.node.partials[0] = 0.00
                num.node.partials[1] = 1.00
            return num

        elif type(other) == int or type(other) == float:
            num = self.node.tape.from_nodes(max(self.value, other), [self.node])
            if self.value > other:
                num.node.partials[0] = 1.00
            else:
                num.node.partials[0] = 0.00
            return num
        raise TypeError(
            f"cannot compute max(number, other) for type(other)={type(other)}"
        )


def normal_cdf(v: float) -> float:
    return (1 + erf(v / sqrt(2))) / 2.00


def normal_pdf(v: float) -> float:
    return 1 / sqrt(2 * pi) * exp(-(v**2) / 2)
