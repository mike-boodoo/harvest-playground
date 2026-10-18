import csv
from pathlib import Path

import pytest

# Tiny hand-written questions in MMLU's on-disk format (not MMLU content):
# two subjects in different categories, 5 dev rows each for few-shot, 4 test rows.
FIXTURE = {
    "abstract_algebra": [  # STEM
        ("What is 2 + 2?", "3", "4", "5", "6", "B"),
        ("What is 3 * 3?", "9", "6", "12", "3", "A"),
        ("What is 10 - 7?", "2", "4", "3", "1", "C"),
        ("What is 8 / 2?", "2", "6", "16", "4", "D"),
        ("Which is prime?", "4", "6", "7", "9", "C"),
        ("What is 1 + 1?", "2", "3", "1", "0", "A"),
        ("What is 5 + 5?", "5", "10", "15", "20", "B"),
        ("What is 2 ^ 3, with a comma, in it?", "6", "9", "8", "\"12\"", "C"),
        ("What is 9 - 9?", "9", "1", "18", "0", "D"),
    ],
    "world_religions": [  # Humanities
        ("Pick A.", "a", "b", "c", "d", "A"),
        ("Pick B.", "a", "b", "c", "d", "B"),
        ("Pick C.", "a", "b", "c", "d", "C"),
        ("Pick D.", "a", "b", "c", "d", "D"),
        ("Pick A again.", "a", "b", "c", "d", "A"),
        ("Test one: pick B.", "a", "b", "c", "d", "B"),
        ("Test two: pick B.", "a", "b", "c", "d", "B"),
        ("Test three: pick D.", "a", "b", "c", "d", "D"),
        ("Test four: pick A.", "a", "b", "c", "d", "A"),
    ],
}


def write_fixture(root: Path) -> Path:
    for subject, rows in FIXTURE.items():
        for split, part in (("dev", rows[:5]), ("test", rows[5:])):
            path = root / split / f"{subject}_{split}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as f:
                csv.writer(f).writerows(part)
    return root


@pytest.fixture
def mmlu_dir(tmp_path):
    return write_fixture(tmp_path / "mmlu")


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A ~200K-parameter random GPT-2 with a byte-level BPE tokenizer, built offline."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    out = tmp_path_factory.mktemp("tiny-gpt2")
    corpus = ["The following are multiple choice questions (with answers) about math. Answer: A B C D"] * 20
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(corpus, vocab_size=400, min_frequency=1, special_tokens=["<|endoftext|>"])
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer, eos_token="<|endoftext|>",
                                  pad_token="<|endoftext|>")
    tok.save_pretrained(out)
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=len(tok), n_positions=64, n_embd=32, n_layer=1, n_head=2,
                     eos_token_id=tok.eos_token_id, bos_token_id=tok.eos_token_id)
    model = GPT2LMHeadModel(cfg)
    model.generation_config.pad_token_id = tok.eos_token_id
    model.save_pretrained(out)
    return out
