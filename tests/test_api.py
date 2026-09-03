import json
from pathlib import Path


def test_sample_outputs_valid():
    samples_path = Path("data/samples/sample_outputs.json")
    assert samples_path.exists()
    samples = json.loads(samples_path.read_text())
    assert len(samples) == 16

    for s in samples:
        assert "id" in s
        assert "agent_name" in s
        assert "content" in s
        assert len(s["content"]) > 50
        assert "metadata" in s
        assert "expected_verdict" in s["metadata"]
        assert "hallucination_count" in s["metadata"]


def test_sample_outputs_coverage():
    samples = json.loads(Path("data/samples/sample_outputs.json").read_text())
    verdicts = [s["metadata"]["expected_verdict"] for s in samples]
    assert "fully_supported" in verdicts
    assert "partially_hallucinated" in verdicts
    assert "mostly_fabricated" in verdicts
    assert "fully_fabricated" in verdicts


def test_corpus_files_exist():
    corpus = Path("data/corpus")
    files = list(corpus.glob("*.txt"))
    assert len(files) >= 5

    expected_sources = ["nvidia", "microsoft", "apple", "fed", "sec"]
    filenames = [f.stem.lower() for f in files]
    for source in expected_sources:
        assert any(source in fn for fn in filenames), f"Missing corpus for {source}"


def test_corpus_files_nonempty():
    for f in Path("data/corpus").glob("*.txt"):
        content = f.read_text()
        assert len(content) > 100, f"{f.name} is too small"
