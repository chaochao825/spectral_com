from pathlib import Path

from llm_spectral_dynamics.publish_checks import validate_tree


def test_publishable_tree_allows_small_result_pt(tmp_path: Path) -> None:
    payload = tmp_path / "results" / "run" / "eigenvalues.pt"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"small tensor payload")

    checked, errors = validate_tree(tmp_path)

    assert checked == 1
    assert errors == []


def test_publishable_tree_rejects_weights_secrets_and_oversized_files(tmp_path: Path) -> None:
    (tmp_path / "model.pth").write_bytes(b"weight")
    fake_token = "github_pat_" + "abcdefghijklmnopqrstuvwxyz"
    (tmp_path / "notes.txt").write_text(fake_token, encoding="utf-8")
    (tmp_path / "large.csv").write_bytes(b"12345")

    checked, errors = validate_tree(tmp_path, max_file_bytes=4)

    assert checked == 3
    assert any("forbidden model/archive suffix .pth" in error for error in errors)
    assert any("possible GitHub token" in error for error in errors)
    assert any("large.csv: 5 bytes exceeds" in error for error in errors)
