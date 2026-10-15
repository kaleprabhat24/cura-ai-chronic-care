# Generated data (not committed)

Everything in this folder except this README is generated and git-ignored.
Regenerate it with:

```
cura-ml all --config ml/configs/cohort_v1.yaml --out ml/data/v1
```

Layout of `ml/data/<version>/`:

| Folder | Contents | Who may read it |
|---|---|---|
| `records/` | Ordinary synthetic clinical and administrative records, as known at the data cutoff | Feature engineering, API, UI |
| `cohort/` | Follow-up episode definitions (information known at prediction time) | Feature engineering, API |
| `outcomes/` | Episode outcomes and target labels (future information relative to each prediction) | Training/evaluation **as targets only** — never as features |
| `hidden/` | Latent simulation traits and the simulator's internal episode log | Generator validation tests and oracle analysis only |
| `manifest.json` | Seed, config hash, per-file content hashes and row counts | Everyone |
