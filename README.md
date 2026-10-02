# event-ledger-sim

An in-memory, thread-safe financial transaction ledger and spatial anomaly risk engine written purely in Python 3.

## Overview

`event-ledger-sim` is designed to simulate a high-throughput, event-driven banking core that combines ACID-compliant transactional guarantees with machine learning risk evaluation. Built entirely with the Python standard library, it enforces deterministic balance transfers across concurrent threads, filters fraudulent transactions via spatial clustering, and models gross portfolio risk using Monte Carlo simulations.

## Key Features

* **Thread-Safe Core Ledger (`ledger.py`):** Uses an in-memory SQLite backend with thread-level locks to guarantee atomic, double-entry balance updates and eliminate race conditions under concurrent workloads.
* **Spatial Anomaly Filtering (`model.py`):** Implements an unsupervised K-means centroid clustering model to establish a user's geographic baseline and flag out-of-bounds dispatch coordinates.
* **Monte Carlo Risk Engine (`model.py`):** Evaluates flagged anomalous transactions over stochastic simulation loops to estimate expected gross weighted portfolio loss.
* **Comprehensive TDD Suite (`test_suite.py`):** Contains unit, integration, and stress tests validating clustering boundaries, convergence of loss estimates, and multithreaded balance consistency under high contention.

## Project Structure

```text
event-ledger-sim/
├── model.py         # Spatial anomaly detection & Monte Carlo risk engine
├── ledger.py        # Thread-safe in-memory financial ledger
├── test_suite.py    # Test-driven development (TDD) validation suite
└── README.md
