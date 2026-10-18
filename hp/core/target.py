"""
Target adapters — backends this playground is allowed to talk to.

Design constraint: OPEN-WEIGHT MODELS ONLY.
No adapter is provided for closed/proprietary commercial APIs (OpenAI,
Anthropic, Google, etc). This isn't an oversight — it's the point. The
tool is meant to be safely portable to any prompt engineer without every
recipient having to re-derive "is this okay for the target I'm pointing
it at". Restricting the backend surface does that instead of a runtime
policy check that's easy to edit away.

If you want to point this at your own hosted open-weight model that's
sitting behind an OpenAI-compatible server (vLLM, TGI, Ollama, llama.cpp
server, etc), use `OpenAICompatLocalTarget` and just make sure the
`base_url` is something you control or that is itself serving an
open-weight model under its documented license.
"""

from __future__ import annotations

import time
import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    prompt: str
    response: str
    logprobs: list[float] | None
    latency_s: float
    raw_meta: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        return hashlib.sha256((self.prompt + "||" + self.response).encode()).hexdigest()


class Target(ABC):
    """Base class for a queryable model backend."""

    #: Must be set by subclasses. Used for provenance logging and to make
    #: it obvious in captured data exactly what was queried.
    model_id: str = "unknown"
    license_tag: str = "unspecified"

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        ...

    def generate_batch(self, prompts: list[str], **kwargs) -> list[GenerationResult]:
        return [self.generate(p, **kwargs) for p in prompts]


class HFLocalTarget(Target):
    """
    Local Hugging Face `transformers` model, loaded from disk/hub.

    Only works with models you've downloaded — i.e. open-weight releases
    whose license already permits you to run and derive from them. There
    is deliberately no code path here that queries a remote closed API.
    """

    def __init__(self, model_name_or_path: str, license_tag: str = "check-model-card",
                 device: str = "auto", dtype: str = "auto", max_new_tokens: int = 512):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model_id = model_name_or_path
        self.license_tag = license_tag
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        torch_dtype = torch.bfloat16 if dtype == "auto" else getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, device_map=device
        )

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        import torch

        max_new = kwargs.pop("max_new_tokens", self.max_new_tokens)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                output_scores=True,
                return_dict_in_generate=True,
                **kwargs,
            )
        latency = time.time() - t0

        gen_tokens = out.sequences[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        logprobs = None
        if out.scores:
            logprobs = [
                torch.log_softmax(score[0], dim=-1)[tok].item()
                for score, tok in zip(out.scores, gen_tokens)
            ]

        return GenerationResult(
            prompt=prompt,
            response=text,
            logprobs=logprobs,
            latency_s=latency,
            raw_meta={"model_id": self.model_id, "n_tokens": len(gen_tokens)},
        )


class LlamaCppTarget(Target):
    """
    In-process adapter for GGUF-quantized open-weight models via
    llama-cpp-python (a Python binding over llama.cpp's C++ core).

    Contrast with the other two adapters:
      - Like HFLocalTarget: runs in-process, no network hop, direct
        access to the model's own internals.
      - UNLIKE HFLocalTarget: no PyTorch tensors, no autograd, no
        device_map dispatch — llama.cpp manages its own memory (mmap'd
        GGUF weights, often CPU or partial GPU offload via
        n_gpu_layers) with an entirely different runtime underneath.
      - Like OpenAICompatLocalTarget: exposes a completion-style call
        with a `logprobs` flag — because llama-cpp-python deliberately
        mirrors the OpenAI request/response shape, even though nothing
        here goes over a network.

    Only accepts local GGUF file paths — same "open-weight only"
    constraint as the rest of this module, enforced by construction
    (there's no remote-URL parameter here to point at anything else).
    """

    def __init__(
        self,
        model_path: str,
        license_tag: str = "check-model-card",
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        max_tokens: int = 512,
    ):
        """
        model_path: path to a local .gguf file. There is deliberately
                    no download-by-name convenience here — forcing a
                    literal file path keeps it obvious that you already
                    have the weights on disk under whatever license
                    governs them, rather than this adapter silently
                    pulling something down on your behalf.
        n_ctx: context window size llama.cpp allocates upfront. Unlike
               HF/transformers, this is a hard preallocation — set too
               low and long prompts will silently truncate.
        n_gpu_layers: how many transformer layers to offload to GPU
               (0 = pure CPU inference, -1 = offload all layers).
               This is llama.cpp's equivalent of HFLocalTarget's
               device_map, but manual rather than "auto" — llama.cpp
               has no automatic memory-fitting dispatcher.
        """
        from llama_cpp import Llama

        self.model_id = model_path
        self.license_tag = license_tag
        self.max_tokens = max_tokens

        # One object does what HFLocalTarget splits across
        # AutoTokenizer + AutoModelForCausalLM — llama.cpp's GGUF
        # format bundles tokenizer vocab directly into the weights
        # file, so there's no separate tokenizer to load.
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            logits_all=False,  # only compute logits for generated tokens, not the whole prompt — cheaper
            verbose=False,
        )

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        t0 = time.time()
        # create_completion() is llama-cpp-python's own API, shaped to
        # resemble OpenAI's /v1/completions response — but this is a
        # direct C++ function call under the hood, not an HTTP request.
        # logprobs=1 asks it to return the log-probability of each
        # generated token (top-1 only; unlike HFLocalTarget we don't
        # get the full vocabulary distribution back, just what the
        # library chooses to expose).
        out = self._llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            logprobs=1,
            **kwargs,
        )
        latency = time.time() - t0

        choice = out["choices"][0]
        text = choice["text"]

        # llama-cpp-python's logprobs field, when present, is shaped
        # like OpenAI's legacy completions schema: a dict with a
        # "token_logprobs" list parallel to the generated tokens —
        # NOT the same shape as HFLocalTarget's manually-computed list
        # or OpenAICompatLocalTarget's chat-schema "content" list.
        # Every adapter's logprobs arrive in a different container
        # shape; normalizing that is exactly what GenerationResult is for.
        lp = choice.get("logprobs")
        logprobs = lp["token_logprobs"] if lp and "token_logprobs" in lp else None

        return GenerationResult(
            prompt=prompt,
            response=text,
            logprobs=logprobs,
            latency_s=latency,
            raw_meta={"model_id": self.model_id, "usage": out.get("usage", {})},
        )


class OpenAICompatLocalTarget(Target):
    """
    Any server exposing the OpenAI-compatible /v1/completions or
    /v1/chat/completions schema — vLLM, TGI, llama.cpp server, Ollama,
    text-generation-webui, etc — that YOU are hosting, serving an
    open-weight model.

    `base_url` should point at infrastructure you control. This adapter
    does not include any pre-configured endpoints for third-party
    commercial providers, and you're responsible for pointing it only at
    open-weight model servers.
    """

    def __init__(self, base_url: str, model_id: str, license_tag: str = "check-model-card",
                 api_key: str = "not-needed", max_tokens: int = 512, chat_mode: bool = True):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.license_tag = license_tag
        self.max_tokens = max_tokens
        self.chat_mode = chat_mode
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120.0,
        )

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        t0 = time.time()
        if self.chat_mode:
            resp = self._client.post("/v1/chat/completions", json={
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "logprobs": True,
                **kwargs,
            })
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            lp_data = data["choices"][0].get("logprobs")
        else:
            resp = self._client.post("/v1/completions", json={
                "model": self.model_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "logprobs": 1,
                **kwargs,
            })
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["text"]
            lp_data = data["choices"][0].get("logprobs")

        latency = time.time() - t0

        logprobs = None
        if lp_data and "content" in (lp_data or {}):
            logprobs = [tok.get("logprob") for tok in lp_data["content"]]

        return GenerationResult(
            prompt=prompt,
            response=text,
            logprobs=logprobs,
            latency_s=latency,
            raw_meta={"model_id": self.model_id, "raw_usage": data.get("usage", {})},
        )
