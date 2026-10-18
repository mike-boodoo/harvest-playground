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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
