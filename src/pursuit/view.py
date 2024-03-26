"""
A utility script designed to visualize the results of simulation runs. It can generate and display graphical
representations of the simulation environment, including the positions and movements of agents.

Functions:
- parse_str(text: str): Parses a string representation of the simulation environment.
- parse_file(filepath: str | os.PathLike): Reads and parses the simulation environment from a file.
- visualize(state, nrows, ncols): Generates a text-based visualization of the simulation state.
- vizinit(config): Generates an initial visualization of the simulation setup.
- vizfinal(steps, nrows, ncols): Generates a final visualization of the simulation outcome based on the steps taken.
- main(): Entry point for executing the visualization process based on input files.

This script is intended for post-simulation analysis, helping to visualize the actions and strategies of agents
throughout the simulation.
"""


import argparse
import itertools
import json
from operator import iconcat
import os
import sys

parser = argparse.ArgumentParser(prog="View")
parser.add_argument(
    "-i",
    "--inputfile",
    dest="inputfile",
    type=argparse.FileType("r"),
    default=sys.stdin,
    required=False,
)
parser.add_argument(
    "-m",
    "--mapfile",
    dest="mapfile",
    type=str,
    default="examples/spredatorafe.txt",
    required=False,
)

predator_icons = {
    "^": 0 - 1j,  # up
    ">": 1 + 0j,  # right
    "v": 0 + 1j,  # down
    "<": -1 + 0j,  # left
}
inv_predator_icons = {
    0 - 1j: "^",  # up
    1 + 0j: ">",  # right
    0 + 1j: "v",  # down
    -1 + 0j: "<",  # left
}

prey_icon = "#"


def parse_str(text: str):
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


def visualize(state, nrows, ncols):
    pos = state["pos"]
    icon = state["icon"]
    preys = state["preys"]
    eaten = state["eaten"]
    grid = [["." for _ in range(ncols)] for _ in range(nrows)]
    for x, y in preys:
        grid[x][y] = "#"
    for x, y in eaten:
        grid[x][y] = "?"
    grid[pos[0]][pos[1]] = icon
    return "\n".join("".join(row) for row in grid)


def vizinit(config):
    pos = config["startpos"]
    dir = config["startdir"]
    preys = config["preys"]
    grid = [["." for _ in range(config["ncols"])] for _ in range(config["nrows"])]
    for x, y in preys:
        grid[x][y] = prey_icon
    grid[pos[0]][pos[1]] = inv_predator_icons[dir]
    return "\n".join("".join(row) for row in grid)


def vizfinal(steps, nrows, ncols):
    grid = [["." for _ in range(ncols)] for _ in range(nrows)]
    for step in steps:
        px, py = step["predator"]["pos"]
        dx, dy = step["predator"]["dir"]
        icon = inv_predator_icons[dx + dy * 1j]
        grid[px][py] = icon
    last_step = steps[-1]
    for prey in last_step["eaten"]:
        grid[prey[0]][prey[1]] = "?"
    for prey in last_step["preys"]:
        grid[prey[0]][prey[1]] = "#"
    return "\n".join("".join(row) for row in grid)


def main() -> int:
    args = parser.parse_args()
    config = parse_file(args.mapfile)
    # pos = config["startpos"]
    # dir = config["startdir"]
    # preys: list[tuple[int, int]] = config["preys"]
    # eaten = []
    # state = dict(pos=pos, icon=inv_predator_icons[dir], preys=preys, eaten=eaten)
    print(vizinit(config))
    steps: list[dict] = json.load(args.inputfile)
    print()
    print(vizfinal(steps, config["nrows"], config["ncols"]))
    # for step in steps:
    #     pos = tuple(step["predator"]["pos"])
    #     dir = step["predator"]["dir"][0] + step["predator"]["dir"][1] * 1j
    #     preys = step["preys"]
    #     eaten = step["eaten"]
    #     state = dict(pos=pos, icon=inv_predator_icons[dir], preys=preys, eaten=eaten)
    #     print("\n\n")
    #     print(visualize(state, config["nrows"], config["ncols"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
