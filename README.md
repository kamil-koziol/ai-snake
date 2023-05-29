# ai-snake

# Info

This project intends to compare 4 different AI implementations 
which can be used when making bots that play games

1. [Simple Decisions](#simple-decisions)
2. [DeepQ Learning](#deepq-learning)
3. [Training based on our gameplay](#training-based-on-our-gameplay)
4. [NEAT](#neat)

# Running project

## Recommended

Use [conda](https://docs.conda.io/en/latest/) for running project.

```
conda create --name <env> --file requirements.txt
```

## Other

Create virtual environment and install dependencies that are listed in Pipfile

```
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

# Implementations

## Simple Decisions

We created simple rules that the 
bot can follow to win

Rules:
1. Move in initial Hamilitionian Cycle for a guarantee win

Pros:
1. Ease of implementation
2. Guarantee of victory

Cons:
1. Unsatisfying
2. Slow victory

## DeepQ Learning

Used [DeepQ]("https://pypi.org/project/DeepQ/") library

Hiperparameters:
1.

Pros:
1.

Cons:
1.

## Training based on our gameplay

Used [Tensorflow]("https://www.tensorflow.org") library

Model:
1.

Pros:
1. Simple implementation

Cons:
1. Collecting data

## NEAT

Used [NEAT-Python]("https://neat-python.readthedocs.io/en/latest/config_file.html") library

Hiperparameters:
1.

Pros:
1.

Cons:
1.