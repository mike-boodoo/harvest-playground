"""
Simulates what "weight/architecture harvesting" actually looks like at
the data level: you download a state_dict (a dict of tensor_name ->
tensor), and everything you could ever do with it — fine-tune diffing,
LoRA-equivalent extraction, architecture inspection — is downstream of
having those tensors in hand, in full, immediately, with zero querying.

This uses small random tensors shaped like a toy transformer's
state_dict (not a real downloaded model — no network access to model
hubs in this sandbox) purely to make the size/shape/diff numbers below
real computed output rather than illustrative prose.

Run: python examples/weight_harvest_demo.py
"""

import torch
import hashlib


def fake_state_dict(seed: int, n_layers: int = 4, d_model: int = 256, d_ff: int = 1024, vocab: int = 8000):
    """
    Builds a state_dict with the same NAMED STRUCTURE a real small
    transformer checkpoint would have — this is exactly what
    `model.state_dict()` returns after `from_pretrained()`, and exactly
    what a .safetensors/.bin file on disk actually contains: a flat
    dict of string keys to tensors, nothing more exotic than that.
    """
    g = torch.Generator().manual_seed(seed)
    sd = {}
    sd["embed.weight"] = torch.randn(vocab, d_model, generator=g)
    for i in range(n_layers):
        sd[f"layer.{i}.attn.q_proj.weight"] = torch.randn(d_model, d_model, generator=g)
        sd[f"layer.{i}.attn.k_proj.weight"] = torch.randn(d_model, d_model, generator=g)
        sd[f"layer.{i}.attn.v_proj.weight"] = torch.randn(d_model, d_model, generator=g)
        sd[f"layer.{i}.attn.out_proj.weight"] = torch.randn(d_model, d_model, generator=g)
        sd[f"layer.{i}.mlp.fc1.weight"] = torch.randn(d_ff, d_model, generator=g)
        sd[f"layer.{i}.mlp.fc2.weight"] = torch.randn(d_model, d_ff, generator=g)
    sd["lm_head.weight"] = torch.randn(vocab, d_model, generator=g)
    return sd


def simulate_finetune(base_sd: dict, seed: int, layers_touched: list[int], perturbation_scale: float = 0.02):
    """
    Simulates what a LoRA-style/full fine-tune actually does to a
    state_dict: MOST tensors are untouched (identical to base), and a
    SUBSET get a small additive delta. This mirrors reality — a
    fine-tune rarely rewrites the embedding table or every layer
    equally; it concentrates change in specific places (often later
    layers, or attention projections, depending on the task).
    """
    g = torch.Generator().manual_seed(seed)
    finetuned_sd = {}
    for name, tensor in base_sd.items():
        touched = any(f"layer.{i}." in name for i in layers_touched) and "mlp" in name
        if touched:
            delta = torch.randn(tensor.shape, generator=g) * perturbation_scale
            finetuned_sd[name] = tensor + delta
        else:
            finetuned_sd[name] = tensor.clone()
    return finetuned_sd


def tensor_hash(t: torch.Tensor) -> str:
    """Content hash of a tensor's raw bytes — how you'd fingerprint a weight file without loading it fully."""
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:12]


def weight_space_diff(base_sd: dict, other_sd: dict) -> list[dict]:
    """
    THE weight-space diff itself — this is the literal, computational
    answer to "harvest diff" from the very first question in this
    conversation. For every tensor name shared between two checkpoints,
    compute: is it identical, and if not, how different (L2 norm of
    the delta, relative to the base tensor's own norm).
    """
    rows = []
    for name in base_sd:
        base_t = base_sd[name]
        other_t = other_sd[name]
        delta = other_t - base_t
        delta_norm = torch.norm(delta).item()
        base_norm = torch.norm(base_t).item()
        rows.append({
            "tensor": name,
            "shape": tuple(base_t.shape),
            "n_params": base_t.numel(),
            "identical": torch.equal(base_t, other_t),
            "delta_l2_norm": delta_norm,
            "relative_change": delta_norm / base_norm if base_norm > 0 else 0.0,
            "base_hash": tensor_hash(base_t),
            "other_hash": tensor_hash(other_t),
        })
    return rows


def main():
    base_sd = fake_state_dict(seed=0)
    # Simulate a fine-tune that only touched the MLP weights in layers 2 and 3
    # (e.g. a task-specific adapter concentrated in later layers).
    finetuned_sd = simulate_finetune(base_sd, seed=1, layers_touched=[2, 3])

    total_params = sum(t.numel() for t in base_sd.values())
    total_bytes_fp32 = total_params * 4  # 4 bytes per float32 param

    print("=== What you get immediately, zero queries ===")
    print(f"Tensors in state_dict: {len(base_sd)}")
    print(f"Total parameters:      {total_params:,}")
    print(f"Size on disk (fp32):   {total_bytes_fp32 / 1e6:.1f} MB")
    print(f"Size on disk (bf16):   {total_bytes_fp32 / 2 / 1e6:.1f} MB  <- what you'd actually download in practice")

    print("\n=== Weight-space diff: base vs. simulated fine-tune ===")
    diff_rows = weight_space_diff(base_sd, finetuned_sd)

    changed = [r for r in diff_rows if not r["identical"]]
    unchanged = [r for r in diff_rows if r["identical"]]

    print(f"Tensors unchanged: {len(unchanged)} / {len(diff_rows)}  <- most of the model, byte-identical")
    print(f"Tensors changed:   {len(changed)} / {len(diff_rows)}   <- exactly where the fine-tune concentrated")

    print("\nChanged tensors, ranked by relative change:")
    for r in sorted(changed, key=lambda x: -x["relative_change"]):
        print(f"  {r['tensor']:30s} shape={str(r['shape']):16s} "
              f"delta_norm={r['delta_l2_norm']:.3f}  relative_change={r['relative_change']:.4f}  "
              f"hash {r['base_hash']} -> {r['other_hash']}")

    print("\nSample of unchanged tensors (hash proves byte-identical, no need to even diff numerically):")
    for r in unchanged[:3]:
        assert r["base_hash"] == r["other_hash"]
        print(f"  {r['tensor']:30s} hash unchanged: {r['base_hash']}")

    print(f"\n... and {len(unchanged) - 3} more unchanged tensors omitted")

    print("\n=== What this proves about 'weight harvesting' ===")
    print("Every number above came from tensors already fully in memory —")
    print("no prompt was sent, no response was sampled, no target.generate()")
    print("was ever called. Contrast this with query-based distillation,")
    print("where you'd need thousands of (prompt, response) pairs to even")
    print("start approximating what this diff shows exactly, instantly.")


if __name__ == "__main__":
    main()
