from types import SimpleNamespace

from scripts.run_member3 import serialise_metrics


def test_serialise_metrics_exports_detection_metrics_and_f1():
    metrics = SimpleNamespace(
        results_dict={
            "metrics/precision(B)": 0.9,
            "metrics/recall(B)": 0.8,
        },
        box=SimpleNamespace(
            map=0.4,
            map50=0.8,
            mp=0.9,
            mr=0.8,
            ap=[0.4, 0.5],
        ),
    )

    row = serialise_metrics(metrics)

    assert row["precision"] == 0.9
    assert row["recall"] == 0.8
    assert row["map50"] == 0.8
    assert row["map50_95"] == 0.4
    assert row["f1"] == 0.8470588235294118
    assert row["per_class_ap"] == [0.4, 0.5]
