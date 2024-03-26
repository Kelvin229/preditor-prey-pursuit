# Predator-Prey (Pursuit)

## Getting Started
1. Install program (macOS, Linux)
```sh
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -e .
```

2. Install program (Windows)
```sh
$ python -m venv .venv
$ .venv\Scripts\activate
$ pip install -e .
```

3. Run the training
```sh
$ pursuit -o=tmp/best-001.ind -c=0.9 -m=0.1 --seed=123
```

4. Generate trace of predator-prey movements
```sh
$ execute -i=tmp/best-001.ind -o=tmp/trace-001.json
```

5. Visualize trace
```sh
$ viz -i=tmp/trace-001.json --mapfile=examples/spredatorafe.txt
```

## References:
   - F.-A. Fortin, F.-M. De Rainville, M.-A. Gardner, M. Parizeau, and C. Gagné, “DEAP: Evolutionary algorithms made easy,” Journal of Machine Learning Research, vol. 13, pp. 2171–2175, jul 2012. https://deap.readthedocs.io/en/master/
