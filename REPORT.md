# Group Challenge Report

## Changes Implemented
- RandomForestClassifier hyperparameters changed across experiments:
  - max_depth: 5, 3, 10, 5
  - n_estimators: 150, 50, 150, 100
  - random_state: 42
- MLflow used to track experiments.
- DVC used to track datasets and pipeline outputs.

## Experiments & Metrics
| Experiment | max_depth | n_estimators | random_state | Accuracy | F1 Macro |
|------------|-----------|--------------|-------------|----------|----------|
| 1          | 5         | 150          | 42          | 1.0      | 1.0      |
| 2          | 3         | 50           | 42          | 1.0      | 1.0      |
| 3          | 10        | 150          | 42          | 1.0      | 1.0      |
| 4          | 5         | 100          | 42          | 1.0      | 1.0      |

## Notes
- All metrics tracked in MLflow.
- Artifacts and metrics.json tracked with DVC.
- Pipeline reproducible with `dvc repro`.

## MLflow Dashboard Screenshots
![Experiment 1](screenshots/exp1.png)
![Experiment 2](screenshots/exp2.png)
