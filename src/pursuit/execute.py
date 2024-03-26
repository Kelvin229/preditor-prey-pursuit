"""
This script is responsible for executing a genetic programming simulation run, utilizing a provided solution tree to
guide the behavior of agents within a simulated environment. It demonstrates how individual solutions can be tested and
visualized in the context of predator-prey dynamics.

Functions:
- progn(*outs, ctx): Sequentially executes given actions within a simulation context.
- create_primitive_set(predatorsim): Initializes a set of primitives (functions and terminals) for the genetic programming
environment.
- parse_str(text: str): Parses and prepares the simulation environment from a text representation.
- parse_file(filepath: str): Reads and prepares the simulation environment from a file.
- main(): Orchestrates the simulation execution, including solution interpretation, simulation, and results output.

This script takes an input file with a solution tree, executes the corresponding actions in the simulation, and outputs
the simulation trace.
"""


import sys
import argparse
from functools import partial
import itertools
import json

from deap import gp, creator, base
from .simulator import PredatorSimulator, Context

parser = argparse.ArgumentParser(prog="Run")
parser.add_argument(
    "-i",
    "--input",
    dest="inputfile",
    type=argparse.FileType("r"),
    default=sys.stdin,
    required=False,
    help="filepath to a solution tree",
)
parser.add_argument(
    "-o",
    "--output",
    dest="outputfile",
    type=argparse.FileType("w"),
    default=sys.stdout,
    required=False,
    help="file which to dump the predator trace",
)
parser.add_argument(
    "--max-moves",
    dest="max_moves",
    type=int,
    default=600,
    required=False,
    help="maximum number of moves allowed",
)


def progn(*outs, ctx):
    for out in outs:
        out(ctx=ctx)


def create_primitive_set(predatorsim: PredatorSimulator) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("MAIN", arity=0)
    pset.addPrimitive(
        lambda *args: partial(predatorsim.if_prey_ahead, *args), 2, name="ifpreyAhead"
    )
    pset.addPrimitive(
        lambda *args: partial(predatorsim.if_prey_left, *args), 2, name="ifpreyLeft"
    )
    pset.addPrimitive(
        lambda *args: partial(predatorsim.if_prey_right, *args), 2, name="ifpreyRight"
    )
    # pset.addPrimitive(
    #     lambda *args: partial(predatorsim.if_prey_behind, *args), 2, name="ifpreyBehind"
    # )
    pset.addPrimitive(lambda *args: partial(progn, *args), 2, name="prog2")
    pset.addPrimitive(lambda *args: partial(progn, *args), 3, name="prog3")
    pset.addTerminal(predatorsim.forward, name="forward")
    pset.addTerminal(predatorsim.left, name="left")
    pset.addTerminal(predatorsim.right, name="right")
    return pset


def parse_str(text: str):
    predator_icons = {
        "^": 0 - 1j,  # up
        ">": 1 + 0j,  # right
        "v": 0 + 1j,  # down
        "<": -1 + 0j,  # left
    }
    prey_icon = "#"

    assert (
        sum(map(text.count, predator_icons.keys())) == 1
    ), f"exactly one of {list(predator_icons.keys())} must be present in the environment"
    grid = [list(line) for line in map(str.strip, text.split("\n")) if line != ""]
    nrows = len(grid)
    ncols = len(grid[0]) if nrows > 0 else 0
    start_pos = (0, 0)
    start_dir = predator_icons[">"]
    for row, col in itertools.product(range(nrows), range(ncols)):
        if grid[row][col] in predator_icons.keys():
            start_pos = (row, col)
            start_dir = predator_icons[grid[row][col]]
            break
    preys = [
        (row, col)
        for row in range(nrows)
        for col in range(ncols)
        if grid[row][col] == prey_icon
    ]
    return dict(
        nrows=nrows,
        ncols=ncols,
        startpos=start_pos,
        startdir=start_dir,
        preys=preys,
    )


def parse_file(filepath: str):
    with open(filepath, "r") as file:
        return parse_str(file.read())


def main() -> int:
    args = parser.parse_args()
    config = parse_file("examples/spredatorafe.txt")
    ctx = Context(
        ncols=config["ncols"],
        nrows=config["nrows"],
        preys=config["preys"],
    )
    predatorsim = PredatorSimulator(
        ncols=config["ncols"],
        nrows=config["nrows"],
        startpos=config["startpos"],
        startdir=config["startdir"],
        preys=config["preys"],
        max_moves=args.max_moves,
    )

    pset = create_primitive_set(predatorsim)
    creator.create("Fitness", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness, steps=list)

    routine = gp.compile(args.inputfile.read(), pset=pset)
    _, _, steps = predatorsim.run(routine, ctx)
    json.dump(steps, fp=args.outputfile)

    return 0


if __name__ == "__main__":
    sys.exit(main())
