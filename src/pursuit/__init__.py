"""
This module serves as the entry point for the genetic programming simulation aimed at evolving predator behaviors in
different environmental contexts. It leverages the DEAP library to create, evaluate, and evolve a population of
solutions over generations.

Functions:
- parse_str(text: str): Parses environment configuration from a string.
- parse_file(filepath: str | os.PathLike): Reads and parses environment configuration from a file.
- progn(*outs, ctx): Executes a sequence of actions in the given context.
- eval_artificial_predator(individual, pset, predatorsim, ctx, out=None): Evaluates the fitness of a single individual.
- create_primitive_set(predatorsim): Creates a DEAP PrimitiveSet tailored for the predator simulation.
- eaSimpleElitism(population, toolbox, cxpb, mutpb, nelites, ngen, stats, halloffame, verbose, logfile): An
elitism-based evolutionary algorithm.
- main(): The main execution function for setting up and running the genetic programming simulation.

The script also defines command-line argument parsing for configuration and control of the simulation run.
"""


import argparse
import itertools
import os
import random
import sys
from functools import partial

import numpy as np
from deap import algorithms, base, creator, gp, tools

from .simulator import PredatorSimulator, Context


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


def parse_file(filepath: str | os.PathLike):
    with open(filepath, "r") as file:
        return parse_str(file.read())


def progn(*outs, ctx):
    for out in outs:
        out(ctx=ctx)


def eval_artificial_predator(
    individual,
    pset: gp.PrimitiveSet,
    predatorsim: PredatorSimulator,
    ctx: Context,
    out: list[str] | None = None,
):
    routine = gp.compile(individual, pset=pset)
    n_preys = len(predatorsim.preys)
    n_eaten, moves, steps = predatorsim.run(routine, ctx)
    individual.steps = steps
    if out is not None:
        out[:] = moves
    return (n_eaten / n_preys,)


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


def eaSimpleElitism(
    population,
    toolbox,
    cxpb,
    mutpb,
    nelites,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
    logfile=sys.stdout,
):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream, file=logfile)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - nelites)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # include elites from previous generation
        elites = sorted(population, key=lambda ind: ind.fitness.values, reverse=True)[
            :nelites
        ]
        offspring = offspring + elites
        random.shuffle(offspring)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream, file=logfile)

    return population, logbook


parser = argparse.ArgumentParser(prog="Pursuit")
parser.add_argument(
    "-o",
    "--output",
    dest="outputfile",
    type=argparse.FileType("w"),
    default=sys.stdout,
    required=False,
    help="file which to write the best solution tree",
)
parser.add_argument(
    "--logfile",
    dest="logfile",
    type=argparse.FileType("w"),
    default=sys.stderr,
    required=False,
    help="file which to write the training logs",
)
parser.add_argument(
    "-c",
    "--crossover-rate",
    dest="crossover_rate",
    type=float,
    required=False,
    default=0.9,
)
parser.add_argument(
    "-m",
    "--mutation-rate",
    dest="mutation_rate",
    type=float,
    required=False,
    default=0.1,
)
parser.add_argument(
    "-g",
    "--generations",
    dest="n_generations",
    type=int,
    required=False,
    default=50,
)
parser.add_argument(
    "-e", "--elites", dest="n_elites", type=int, required=False, default=0
)
parser.add_argument(
    "-p", "--popsize", dest="popsize", type=int, required=False, default=100
)
parser.add_argument(
    "--max-moves", dest="max_moves", type=int, required=False, default=600
)
parser.add_argument(
    "--tournsize",
    dest="tournsize",
    type=int,
    required=False,
    default=None,
    help="the number of individuals to compete in tournament [default: none - uses roulette wheel instead]",
)
parser.add_argument(
    "-s", "--seed", dest="random_seed", type=int, required=False, default=123
)


def main() -> int:
    args = parser.parse_args()
    random.seed(args.random_seed)

    config = parse_file("examples/spredatorafe.txt")
    ctx = Context(
        ncols=config["ncols"],
        nrows=config["nrows"],
        preys=config["preys"],
        seed=args.random_seed,
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
    toolbox = base.Toolbox()
    toolbox.register("expr_init", gp.genFull, pset=pset, min_=3, max_=7)
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, toolbox.expr_init
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_artificial_predator, pset=pset, predatorsim=predatorsim, ctx=ctx)
    if args.tournsize is not None:
        toolbox.register("select", tools.selTournament, tournsize=args.tournsize)
    else:
        toolbox.register("select", tools.selRoulette)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=3, max_=7)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(lambda ind: ind.height, 17))
    toolbox.decorate("mutate", gp.staticLimit(lambda ind: ind.height, 17))

    pop = toolbox.population(n=args.popsize)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("max", np.max)

    _, _logbook = eaSimpleElitism(
        pop,
        toolbox,
        args.crossover_rate,
        args.mutation_rate,
        args.n_elites,
        args.n_generations,
        stats,
        hof,
        verbose=True,
        logfile=args.logfile,
    )

    print(f"Best Individual: fitness={hof[0].fitness.values}, height={hof[0].height}")
    best = hof[0]
    with args.outputfile as f:
        f.write(str(best))

    return 0


if __name__ == "__main__":
    sys.exit(main())
