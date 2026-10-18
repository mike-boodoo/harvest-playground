"""
End-to-end demo using a mock Target, so you can verify the whole
pipeline (strategy -> capture -> signing -> verification) works
without needing to download a real model first.

Run: python examples/mock_target_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.target import Target, GenerationResult
from core.provenance import ProvenanceSigner
from core.capture import run_capture
from strategies.distill import DistillationSampler
from strategies.synth_eval import SynthEvalSampler


class MockTarget(Target):
    """Fake target for testing the pipeline without a real model."""
    model_id = "mock-open-model-7b"
    license_tag = "apache-2.0"

    def generate(self, prompt, **kwargs):
        return GenerationResult(
            prompt=prompt,
            response=f"[mock response to: {prompt[:40]}...]",
            logprobs=None,
            latency_s=0.001,
        )


def main():
    target = MockTarget()
    signer = ProvenanceSigner()

    print("=== Distillation sampler ===")
    strategy = DistillationSampler(topics=["math", "coding", "writing"])
    record = run_capture(target, strategy, n_prompts=5, signer=signer, out_dir="./examples/_out")
    print(f"Captured {len(record['batch']['results'])} samples")
    for r in record["batch"]["results"]:
        print(f"  - {r['prompt'][:60]}")

    print("\n=== Signature verification ===")
    valid = ProvenanceSigner.verify(record)
    print(f"Signature valid: {valid}")

    print("\n=== Tamper check ===")
    tampered = dict(record)
    tampered["batch"] = dict(record["batch"])
    tampered["batch"]["model_id"] = "different-model"
    valid_after_tamper = ProvenanceSigner.verify(tampered)
    print(f"Signature valid after tampering model_id: {valid_after_tamper}")

    print("\n=== SynthEval sampler ===")
    eval_strategy = SynthEvalSampler(topics=["physics", "history"])
    eval_record = run_capture(target, eval_strategy, n_prompts=3, signer=signer, out_dir="./examples/_out")
    for r in eval_record["batch"]["results"]:
        print(f"  - {r['prompt']}")


if __name__ == "__main__":
    main()
