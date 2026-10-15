"""Command line interface: `cura-ml <command> --config ... --out ...`."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cura_ml.cohort.report import write_report
from cura_ml.config import load_config
from cura_ml.manifest import write_manifest
from cura_ml.pipeline import build_cohort, generate_records, run_all


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cura-ml", description="Cura AI synthetic cohort tools")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("generate", "simulate records and hidden tables"),
        ("build-episodes", "rebuild episodes and labels from persisted records"),
        ("report", "write the cohort summary report"),
        ("all", "generate + build-episodes + manifest + report"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", default="ml/configs/cohort_v1.yaml")
        p.add_argument("--out", default=None, help="dataset directory (default ml/data/<version>)")
        p.add_argument("--report", default=None,
                       help="report path (default ml/reports/cohort_<version>_summary.md)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(args.out or f"ml/data/{cfg['dataset_version']}")
    report = Path(args.report or f"ml/reports/cohort_{cfg['dataset_version']}_summary.md")
    started = time.perf_counter()
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
    print(f"{args.command} finished in {time.perf_counter() - started:.1f}s -> {out}")


if __name__ == "__main__":
    main()
