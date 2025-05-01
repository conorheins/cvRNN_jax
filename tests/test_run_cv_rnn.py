# tests/test_run_cv_rnn.py
import json, subprocess, sys, pytest

# each threshold is the smallest value of (mean acc - 2 * sem_acc) measured across 3 seeds of 100 models each
THRESHOLDS = {
  "2shapes":       0.9623,
  "3shapes":       0.9119, 
  "natural_image": 0.8761,
}

@pytest.mark.parametrize("dataset,image_index", [
  ("2shapes",       0),
  ("2shapes",       1),
  ("2shapes",       2),
  ("3shapes",       0),
  ("3shapes",       1),
  ("3shapes",       2),
  ("natural_image", 0),
])
def test_minimum_accuracy(dataset, image_index):
    cmd = [
      sys.executable, "run_cv_rnn.py",
      "--dataset",       dataset,
      "--image_index",   str(image_index),
      "--seed",          "0",
      "--ensemble_size", "10",
      "--json",
    ]
    out = subprocess.check_output(cmd, text=True)
    metrics = json.loads(out)
    mean_acc = metrics["mean_acc"]
    thresh   = THRESHOLDS[dataset]
    assert mean_acc >= thresh, (
      f"{dataset}[{image_index}]: "
      f"mean acc {mean_acc:.4f} < threshold {thresh:.4f}"
    )
