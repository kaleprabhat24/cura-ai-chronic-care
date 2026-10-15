"""Command line interface: `cura-ml <command> ...`.

Data:      generate | build-episodes | report | all
ML (M2/3): features | train | evaluate | predict
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cura_ml.cohort.report import write_report
from cura_ml.config import load_config
from cura_ml.manifest import write_manifest
from cura_ml.pipeline import build_cohort, generate_records, run_all

DEFAULT_CONFIG = "ml/configs/cohort_v1.yaml"
DEFAULT_TRAIN_CONFIG = "ml/configs/train_v1.yaml"


def _train_config(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cura-ml", description="Cura AI synthetic cohort and ML tools")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("generate", "simulate records and hidden tables"),
        ("build-episodes", "rebuild episodes and labels from persisted records"),
        ("report", "write the cohort summary report"),
        ("all", "generate + build-episodes + manifest + report"),
        ("features", "build the episode feature table and docs/FEATURES.md"),
        ("train", "train, calibrate, evaluate, select and save the model bundle + report"),
        ("evaluate", "reload a saved bundle and recompute its test metrics"),
        ("predict", "score eligible episodes with a saved bundle and explain them"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", default=DEFAULT_CONFIG)
        p.add_argument("--out", default=None, help="dataset directory (default ml/data/<version>)")
        p.add_argument("--report", default=None, help="report path")
        if name in ("train", "evaluate", "predict"):
            p.add_argument("--train-config", default=DEFAULT_TRAIN_CONFIG)
            p.add_argument("--bundle", default=None,
                           help="bundle directory (default ml/artifacts/models/<model_version>)")
        if name == "predict":
            p.add_argument("--episode-id", action="append", default=None,
                           help="episode to score (repeatable); default: episodes open at the cutoff")
            p.add_argument("--limit", type=int, default=3)
            p.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(args.out or f"ml/data/{cfg['dataset_version']}")
    started = time.perf_counter()
    if args.command in ("generate", "build-episodes", "report", "all"):
        report = Path(args.report or f"ml/reports/cohort_{cfg['dataset_version']}_summary.md")
        if args.command == "generate":
            generate_records(cfg, out)
            write_manifest(out, cfg)
        elif args.command == "build-episodes":
            build_cohort(cfg, out)
            write_manifest(out, cfg)
        elif args.command == "report":
            write_report(out, cfg, report)
        else:
            manifest = run_all(cfg, out)
            write_report(out, cfg, report)
            for path, info in manifest["files"].items():
                print(f"{path:45s} rows={info['rows']:>8,}  content={info['content_sha256'][:12]}")
            print(f"report: {report}")
    elif args.command == "features":
        from cura_ml.features.spec import spec_markdown
        from cura_ml.modeling.train import build_feature_table

        feats = build_feature_table(cfg, out)
        Path("docs/FEATURES.md").write_text(spec_markdown(), encoding="utf-8")
        print(f"features: {len(feats):,} eligible episodes × {feats.shape[1] - 4} columns "
              f"-> {out / 'features' / 'episode_features.parquet'}; docs/FEATURES.md updated")
    else:
        from cura_ml.modeling.train import evaluate_bundle, predict_episodes, train_and_evaluate

        tcfg = _train_config(args.train_config)
        bundle = Path(args.bundle or f"ml/artifacts/models/{tcfg['model_version']}")
        if args.command == "train":
            report = Path(args.report or f"ml/reports/model_{tcfg['model_version']}.md")
            res = train_and_evaluate(cfg, tcfg, out, bundle.parent, report)
            ev = res["evaluation"]
            sel = ev["selected_model"]
            for name, m in ev["test"]["point"].items():
                print(f"{name:22s} test PR-AUC={m['pr_auc']:.3f} ROC-AUC={m['roc_auc']:.3f} "
                      f"Brier={m['brier']:.4f} ECE={m['ece']:.3f}")
            print(f"selected: {sel}; bundle: {res['bundle_dir']}; report: {report}")
        elif args.command == "evaluate":
            print(json.dumps(evaluate_bundle(bundle, out), indent=2))
        else:
            rows = predict_episodes(bundle, out, args.episode_id, args.top_k)
            print(json.dumps(rows[: args.limit] if not args.episode_id else rows, indent=2))
            if not args.episode_id:
                print(f"({len(rows)} open eligible episodes scored; showing {min(args.limit, len(rows))})")
    print(f"{args.command} finished in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
