# Evaluation set

Ground-truth labels so accuracy is a measured number, not a guess. One JSON per clip in
`eval/groundtruth/`. Run `.\.venv\Scripts\python.exe eval\run_eval.py` to score the pipeline.

## Label format

```json
{
  "video": "input/deadlift-1.mov",
  "lift": "deadlift",
  "true_rep_count": 3,
  "per_rep_legal_pass": [true, true, false]
}
```

- **true_rep_count** — how many real reps you actually did.
- **per_rep_legal_pass** — one true/false per rep, in order. For **squat** = did it hit depth
  (hip crease below knee). For **deadlift** = did it fully lock out (knees + hips, shoulders back).
  This is the IPF red-light criterion the tool predicts.

The more clips you add (different gyms, angles, lighting, plate colours), the more trustworthy
the score — and these double as future training data.
