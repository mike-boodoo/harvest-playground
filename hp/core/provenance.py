"""
Provenance logging for captured generations.

Every batch of captures gets ed25519-signed so downstream users of a
dataset (or you, six months later) can verify: which model produced
this, under what sampling strategy, when, and that the log hasn't been
tampered with since.

This mirrors the signing approach from your model lineage tracking
system — same primitive, applied to capture batches instead of model
checkpoints.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder
from nacl.exceptions import BadSignatureError

from .target import GenerationResult


@dataclass
class CaptureBatch:
    model_id: str
    license_tag: str
    strategy_name: str
    created_at: float
    results: list[dict]
    batch_meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


class ProvenanceSigner:
    """Wraps an ed25519 keypair for signing/verifying capture batches."""

    def __init__(self, signing_key: SigningKey | None = None):
        self.signing_key = signing_key or SigningKey.generate()
        self.verify_key = self.signing_key.verify_key

    @classmethod
    def from_hex(cls, hex_seed: str) -> "ProvenanceSigner":
        return cls(SigningKey(hex_seed, encoder=HexEncoder))

    def export_seed_hex(self) -> str:
        return self.signing_key.encode(encoder=HexEncoder).decode()

    def export_pubkey_hex(self) -> str:
        return self.verify_key.encode(encoder=HexEncoder).decode()

    def sign_batch(self, results: list[GenerationResult], model_id: str,
                    license_tag: str, strategy_name: str, batch_meta: dict | None = None) -> dict:
        batch = CaptureBatch(
            model_id=model_id,
            license_tag=license_tag,
            strategy_name=strategy_name,
            created_at=time.time(),
            results=[
                {
                    "prompt": r.prompt,
                    "response": r.response,
                    "content_hash": r.content_hash(),
                    "latency_s": r.latency_s,
                }
                for r in results
            ],
            batch_meta=batch_meta or {},
        )
        # sort_keys + compact separators in to_json() ensure the exact
        # same batch content always serializes to the exact same bytes —
        # required for signatures to verify deterministically regardless
        # of dict ordering.
        payload = batch.to_json().encode()
        signed = self.signing_key.sign(payload)  # PyNaCl prepends the 64-byte signature to the message internally

        return {
            "batch": asdict(batch),
            "signature_hex": signed.signature.hex(),
            "pubkey_hex": self.export_pubkey_hex(),
        }

    @staticmethod
    def verify(signed_record: dict) -> bool:
        """Verify a signed batch record produced by sign_batch()."""
        try:
            vk = VerifyKey(bytes.fromhex(signed_record["pubkey_hex"]))
            # Must re-serialize with IDENTICAL sort_keys/separators settings
            # used at signing time, or the byte payload won't match and
            # verification fails even for an untampered record.
            payload = json.dumps(signed_record["batch"], sort_keys=True, separators=(",", ":")).encode()
            vk.verify(payload, bytes.fromhex(signed_record["signature_hex"]))  # raises BadSignatureError on mismatch
            return True
        except (BadSignatureError, KeyError, ValueError):
            # BadSignatureError: tampered/wrong content. KeyError: malformed
            # record missing expected fields. ValueError: bad hex encoding.
            # All three mean "can't trust this record" — treat uniformly.
            return False

    def save_batch(self, signed_record: dict, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(signed_record["batch"]["created_at"] * 1000)
        path = out_dir / f"batch_{ts}.json"
        path.write_text(json.dumps(signed_record, indent=2))
        return path
