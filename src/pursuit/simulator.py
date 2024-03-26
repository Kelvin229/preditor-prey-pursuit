"""
Defines the core simulation logic for the predator-prey environment. It includes classes for managing the simulation
context and executing the behaviors of both predators and prey within a defined space.

Classes:
- Context: Manages the simulation environment, including prey positions and the historical steps of the
simulation.
- PredatorSimulator: Simulates the behavior of a predator within the environment,
tracking its position, direction, and the actions it takes.

Key Functions within PredatorSimulator:
- left, right, forward: Basic movement actions available to the predator.
- _has_prey_ahead, if_prey_ahead, if_prey_left, if_prey_right, if_prey_behind: Decision-making functions evaluating the
presence of prey (prey) relative to the predator's position.
- run: Executes a given strategy within the simulation context, assessing its effectiveness.

This module is central to running simulations and evaluating the evolutionary outcomes of genetic programming strategies
in predator-prey dynamics.
"""


import random


class Context:
    def __init__(
        self,
        nrows: int,
        ncols: int,
        preys: list[tuple[int, int]],
        *,
        seed=None,
    ) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.original_preys = list(preys)
        self.preys = list(self.original_preys)
        self.eaten_preys = []
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.steps = []

    def _predator_nearby(self, predatorloc: tuple[int, int], preyloc: tuple[int, int]):
        manhdist = abs(predatorloc[0] - preyloc[0]) + abs(predatorloc[1] - preyloc[1])
        return manhdist <= 1

    def update(self, predator: "PredatorSimulator"):
        for i in range(len(self.preys)):
            if (
                self._predator_nearby(predator.pos, self.preys[i])
                and self.rng.random() <= 0.5
            ):
                dir = self.rng.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
                newloc = (
                    (self.preys[i][0] + dir[0]) % self.nrows,
                    (self.preys[i][1] + dir[1]) % self.ncols,
                )
                if newloc not in self.preys:
                    # prevent multiple preys overlapping in the same cell
                    self.preys[i] = newloc
        self.steps.append(
            dict(
                predator=dict(pos=predator.pos, dir=(predator.dir.real, predator.dir.imag)),
                preys=list(self.preys),
                eaten=list(self.eaten_preys),
            )
        )

    def eat(self, loc: tuple[int, int]):
        self.preys.remove(loc)
        self.eaten_preys.append(loc)

    def reset(self):
        self.preys = list(self.original_preys)
        self.eaten_preys = []
        self.rng = random.Random(self.seed)
        self.steps = []


class PredatorSimulator:
    def __init__(
        self,
        nrows: int,
        ncols: int,
        startpos: tuple[int, int],
        startdir: complex,
        preys: list[tuple[int, int]],
        *,
        max_moves: int,
    ) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.startpos = startpos
        self.startdir = startdir
        self.max_moves = max_moves
        self.pos = startpos
        self.dir = startdir
        self.moves = []
        self.preys = preys

    def left(self, *, ctx: "Context"):
        if len(self.moves) >= self.max_moves:
            return
        self.dir *= -1j
        self.moves.append("L")
        ctx.update(self)

    def right(self, *, ctx: "Context"):
        if len(self.moves) >= self.max_moves:
            return
        self.dir *= 1j
        self.moves.append("R")
        ctx.update(self)

    def forward(self, *, ctx: "Context"):
        if len(self.moves) >= self.max_moves:
            return
        self.pos = (
            (self.pos[0] + int(self.dir.imag)) % self.nrows,
            (self.pos[1] + int(self.dir.real)) % self.ncols,
        )
        if self.pos in ctx.preys:
            ctx.eat(self.pos)
        self.moves.append("F")
        ctx.update(self)

    def _has_prey_ahead(self, ctx: "Context", turn_dir: complex = 1) -> bool:
        d = self.dir * turn_dir
        next_pos = (
            (self.pos[0] + int(d.imag)) % self.nrows,
            (self.pos[1] + int(d.real)) % self.ncols,
        )
        return next_pos in ctx.preys

    def if_prey_ahead(self, out1, out2, *, ctx: "Context"):
        if self._has_prey_ahead(ctx):
            out1(ctx=ctx)
        else:
            out2(ctx=ctx)

    def if_prey_left(self, out1, out2, *, ctx: "Context"):
        if self._has_prey_ahead(ctx, turn_dir=-1j):
            out1(ctx=ctx)
        else:
            out2(ctx=ctx)

    def if_prey_right(self, out1, out2, *, ctx: "Context"):
        if self._has_prey_ahead(ctx, turn_dir=1j):
            out1(ctx=ctx)
        else:
            out2(ctx=ctx)

    def if_prey_behind(self, out1, out2, *, ctx: "Context"):
        if self._has_prey_ahead(ctx, turn_dir=-1 + 0j):
            out1(ctx=ctx)
        else:
            out2(ctx=ctx)

    def _reset(self):
        self.pos = self.startpos
        self.dir = self.startdir
        self.moves = []

    def run(self, routine, ctx: "Context"):
        ctx.reset()
        self._reset()
        while len(ctx.preys) > 0 and len(self.moves) < self.max_moves:
            routine(ctx=ctx)
        n_eaten = len(ctx.eaten_preys)
        moves = self.moves
        steps = ctx.steps
        self._reset()
        ctx.reset()
        return n_eaten, moves, steps
