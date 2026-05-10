# econlab-causal-macro-rl

A causal macro-finance simulator with a learning central bank.

## First milestone

This first version includes:

- A minimal macro state transition system
- A Taylor-rule baseline
- A Gymnasium-compatible central bank environment
- A smoke test for random actions and rule-based policy

## Install

```bash
pip install -e .
```

## Run smoke test

```bash
python -m econlab.scripts.smoke_test
```