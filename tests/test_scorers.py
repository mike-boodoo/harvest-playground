import pytest

from core.target import GenerationResult, Target
from playground.scorers import GenerationScorer, LogprobScorer, MockScorer


class CannedTarget(Target):
    model_id = "canned"
    license_tag = "test"

    def __init__(self, reply, kwarg_name):
        self.reply = reply
        if kwarg_name == "max_new_tokens":
            self.max_new_tokens = 512
        self.seen_kwargs = None

    def generate(self, prompt, **kwargs):
        self.seen_kwargs = kwargs
        return GenerationResult(prompt=prompt, response=self.reply, logprobs=None, latency_s=0.0)


@pytest.mark.parametrize("reply,expected", [
    (" B", "B"),
    ("The answer is C.", "C"),
    ("(D) because...", "D"),
    ("A) first", "A"),
    ("I am not sure", None),
    ("ABCD", None),  # no standalone letter
])
def test_generation_scorer_parses_first_standalone_letter(reply, expected):
    letter, probs = GenerationScorer(CannedTarget(reply, "max_tokens")).choose("prompt")
    assert letter == expected
    assert probs is None


@pytest.mark.parametrize("kwarg", ["max_new_tokens", "max_tokens"])
def test_generation_scorer_uses_the_targets_length_kwarg(kwarg):
    target = CannedTarget("A", kwarg)
    GenerationScorer(target, max_new_tokens=8).choose("prompt")
    assert target.seen_kwargs == {kwarg: 8}


def test_mock_scorer_is_seeded():
    a, b = MockScorer(seed=1), MockScorer(seed=1)
    seq = [a.choose("p")[0] for _ in range(20)]
    assert seq == [b.choose("p")[0] for _ in range(20)]
    assert set(seq) <= set("ABCD")


def test_logprob_scorer_on_a_real_model(tiny_model_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    scorer = LogprobScorer(model, tok)
    assert len(set(scorer.letter_ids)) == 4, "each letter must map to a distinct token"

    letter, probs = scorer.choose("Question?\nA. x\nB. y\nC. z\nD. w\nAnswer:")
    assert letter in "ABCD"
    assert set(probs) == set("ABCD")
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
    assert letter == max(probs, key=probs.get)

    # prompts longer than the 64-token context must be truncated, not crash
    letter, _ = scorer.choose("filler " * 500 + "Answer:")
    assert letter in "ABCD"
