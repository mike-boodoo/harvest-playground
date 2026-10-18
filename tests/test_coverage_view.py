import pytest

from playground import coverage_view


def write(tmp_path, text):
    path = tmp_path / "traffic.csv"
    path.write_text(text)
    return path


def test_load_csv_with_and_without_optional_columns(tmp_path):
    path = write(tmp_path, "input_text,resolved,confidence,cluster\n"
                           "my billing is wrong,yes,0.9,\n"
                           "cannot login,0,,\n"
                           "where is it,true,0.2,shipping\n")
    tracker = coverage_view.load_csv(path, ["billing", "login"], threshold=0.5)
    assert tracker.overall_coverage()["n"] == 3
    rows = {r["cluster"]: r for r in coverage_view.cluster_rows(tracker)}
    assert rows["billing"]["resolved %"] == 100.0
    assert rows["login"]["resolved %"] == 0.0
    assert rows["shipping"]["confident %"] == 0.0  # resolved but under the threshold
    assert coverage_view.drift_rows(tracker) == []  # no timestamps spread -> no drift windows


@pytest.mark.parametrize("text,msg", [
    ("input_text\nhello\n", "missing column"),
    ("input_text,resolved\nhello,maybe\n", "line 2"),
    ("input_text,resolved,timestamp\nhello,true,5\nbye,true,\n", "line 3: 'timestamp' is empty"),
])
def test_load_csv_errors(tmp_path, text, msg):
    with pytest.raises(ValueError, match=msg):
        coverage_view.load_csv(write(tmp_path, text), [])


def test_demo_traffic_flags_the_surging_cluster():
    tracker = coverage_view.demo_tracker()
    assert tracker.overall_coverage()["n"] == 600
    flagged = {r["cluster"]: r for r in coverage_view.drift_rows(tracker)}
    assert "crypto_payment" in flagged
    assert "input share up" in flagged["crypto_payment"]["why"]
    assert coverage_view.cluster_rows(tracker)[0]["cluster"] == "crypto_payment"  # worst first


def test_timestamps_from_csv_drive_drift(tmp_path):
    lines = ["input_text,resolved,timestamp,cluster"]
    lines += [f"q,true,{t},refund" for t in range(0, 50)]
    lines += [f"q,false,{t},refund" for t in range(100, 150)]
    tracker = coverage_view.load_csv(write(tmp_path, "\n".join(lines) + "\n"), [])
    [row] = coverage_view.drift_rows(tracker)
    assert row["cluster"] == "refund" and "coverage dropped" in row["why"]
