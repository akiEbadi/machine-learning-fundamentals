# Traffic Event Analysis

A small exploratory project that groups synthetic traffic events with K-Means.

## Features

Each sample has six numerical features:

- Speed
- Speed variation
- Direction change
- Duration
- Distance
- Acceleration

## What the script does

1. Scales the features.
2. Runs a simple K-Means implementation written in this file.
3. Uses PCA to project the samples to two dimensions for plotting.
4. Prints cluster sizes and saves a scatter plot.

## Run

```bash
pip install -r requirements.txt
python main.py
```

The dataset is synthetic and is only intended for demonstrating the workflow.
