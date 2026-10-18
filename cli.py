#!/usr/bin/env python3
"""
Harvest Playground CLI.

Open-weight models only. See core/target.py for why.

Usage:
    python cli.py distill  --model ./my-llama-checkpoint --topics math,coding --n 50
    python cli.py syntheval --backend openai_compat --base-url http://localhost:8000 \\
                             --model-id my-model --topics physics,history --n 30
    python cli.py verify --batch-file ./captures/batch_1234.json

Run `python cli.py --help` for full options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.target import HFLocalTarget, OpenAICompatLocalTarget
from core.provenance import ProvenanceSigner
from core.capture import run_capture
from strategies.distill import DistillationSampler
from strategies.synth_eval import SynthEvalSampler


def build_target(args) -> "Target":
    if args.backend == "hf_local":
        return HFLocalTarget(args.model, license_tag=args.license_tag)
    elif args.backend == "openai_compat":
        if not args.base_url or not args.model_id:
            sys.exit("--base-url and --model-id are required for openai_compat backend")
        return OpenAICompatLocalTarget(
            base_url=args.base_url, model_id=args.model_id, license_tag=args.license_tag
        )
    else:
        sys.exit(f"Unknown backend: {args.backend}")


def get_or_create_signer(key_path: str) -> ProvenanceSigner:
    p = Path(key_path)
    if p.exists():
        return ProvenanceSigner.from_hex(p.read_text().strip())
    signer = ProvenanceSigner()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(signer.export_seed_hex())
    print(f"[+] New signing key generated and saved to {p}")
    return signer


def parse_topics(spec: str) -> list[str]:
    """Comma-separated topics, or `@path` to a file with one topic per line (# comments allowed)."""
    if spec.startswith("@"):
        lines = Path(spec[1:]).read_text().splitlines()
        return [l.strip() for l in lines if l.strip() and not l.lstrip().startswith("#")]
    return spec.split(",")


def cmd_distill(args):
    target = build_target(args)
    strategy = DistillationSampler(topics=parse_topics(args.topics))
    signer = get_or_create_signer(args.key_file)
    record = run_capture(target, strategy, args.n, signer, out_dir=args.out_dir)
    print(f"[+] Captured {args.n} samples from {target.model_id}")
    print(f"[+] Written to {record['_written_to']}")


def cmd_syntheval(args):
    target = build_target(args)
    strategy = SynthEvalSampler(topics=parse_topics(args.topics))
    signer = get_or_create_signer(args.key_file)
    record = run_capture(target, strategy, args.n, signer, out_dir=args.out_dir)
    print(f"[+] Captured {args.n} eval samples from {target.model_id}")
    print(f"[+] Written to {record['_written_to']}")


def cmd_verify(args):
    record = json.loads(Path(args.batch_file).read_text())
    ok = ProvenanceSigner.verify(record)
    print("VALID signature" if ok else "INVALID signature")
    sys.exit(0 if ok else 1)


def cmd_fetch_mmlu(args):
    from playground import mmlu

    subjects = args.subjects.split(",") if args.subjects else None
    written = mmlu.fetch(args.data_dir, subjects)
    print(f"[+] {len(written)} file(s) downloaded to {args.data_dir}")


def cmd_benchmark(args):
    from playground import benchmark, mmlu, models

    if args.backend == "openai_compat":
        loaded = models.load("openai_compat", args.model_id or "", args.base_url or "")
    else:
        loaded = models.load(args.backend, args.model or "")
    subjects = args.subjects.split(",") if args.subjects else mmlu.available_subjects(args.data_dir)
    if not subjects:
        sys.exit(f"No MMLU data in {args.data_dir}; run `python cli.py fetch-mmlu` first")
    result = benchmark.run_benchmark(loaded.scorer, args.data_dir, subjects, loaded.label,
                                     k_shot=args.k_shot, n_per_subject=args.n_per_subject or None)
    for row in benchmark.comparison_table(result, benchmark.load_references(args.references)):
        avg = "  -  " if row["Average"] is None else f"{row['Average']:5.1f}"
        print(f"  {avg}  {row['name']}  ({row['setting']})")
    print(f"[+] Written to {result.save(args.out_dir)}")


def cmd_ui(args):
    from playground import app

    app.main(["--host", args.host, "--port", str(args.port)] + (["--share"] if args.share else []))


def main():
    parser = argparse.ArgumentParser(description="Harvest Playground — open-weight model capture toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--backend", choices=["hf_local", "openai_compat"], default="hf_local")
        p.add_argument("--model", help="Local model path/name (hf_local backend)")
        p.add_argument("--base-url", help="Server URL (openai_compat backend)")
        p.add_argument("--model-id", help="Model id as known to the server (openai_compat backend)")
        p.add_argument("--license-tag", default="check-model-card")
        p.add_argument("--n", type=int, default=20)
        p.add_argument("--out-dir", default="./captures")
        p.add_argument("--key-file", default="./captures/.signing_key")

    p_distill = sub.add_parser("distill", help="Diverse-coverage capture for distillation datasets")
    add_common(p_distill)
    p_distill.add_argument("--topics", required=True, help="Comma-separated topic list, or @file (one per line)")
    p_distill.set_defaults(func=cmd_distill)

    p_eval = sub.add_parser("syntheval", help="Synthetic eval-set generation")
    add_common(p_eval)
    p_eval.add_argument("--topics", required=True, help="Comma-separated topic list, or @file (one per line)")
    p_eval.set_defaults(func=cmd_syntheval)

    p_verify = sub.add_parser("verify", help="Verify a signed capture batch")
    p_verify.add_argument("--batch-file", required=True)
    p_verify.set_defaults(func=cmd_verify)

    p_fetch = sub.add_parser("fetch-mmlu", help="Download MMLU (dev + test splits) for the benchmark")
    p_fetch.add_argument("--data-dir", default="./data/mmlu")
    p_fetch.add_argument("--subjects", help="Comma-separated subject ids (default: all 57)")
    p_fetch.set_defaults(func=cmd_fetch_mmlu)

    p_bench = sub.add_parser("benchmark", help="Run MMLU and compare against published scores")
    p_bench.add_argument("--backend", choices=["hf_local", "openai_compat", "mock"], default="hf_local")
    p_bench.add_argument("--model", help="Local model path/name (hf_local backend)")
    p_bench.add_argument("--base-url", help="Server URL (openai_compat backend)")
    p_bench.add_argument("--model-id", help="Model id as known to the server (openai_compat backend)")
    p_bench.add_argument("--data-dir", default="./data/mmlu")
    p_bench.add_argument("--subjects", help="Comma-separated subject ids (default: every downloaded subject)")
    p_bench.add_argument("--n-per-subject", type=int, default=0, help="0 = all questions")
    p_bench.add_argument("--k-shot", type=int, default=5)
    p_bench.add_argument("--references", default="./config/reference_scores.json")
    p_bench.add_argument("--out-dir", default="./runs/benchmarks")
    p_bench.set_defaults(func=cmd_benchmark)

    p_ui = sub.add_parser("ui", help="Launch the interactive playground UI")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=7860)
    p_ui.add_argument("--share", action="store_true", help="Create a public Gradio link")
    p_ui.set_defaults(func=cmd_ui)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
