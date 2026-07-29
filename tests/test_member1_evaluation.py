from pathlib import Path

from hripcb_member1.evaluation import evaluate_variants


class FakeBox:
    mp = 0.95
    mr = 0.92
    map50 = 0.91
    map = 0.49


class FakeMetrics:
    box = FakeBox()
    results_dict = {}


class FakeModel:
    instances = []

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.calls = []
        self.__class__.instances.append(self)

    def val(self, **kwargs):
        self.calls.append(kwargs)
        return FakeMetrics()


def test_evaluate_variants_reuses_one_checkpoint_and_same_settings(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    data = {
        variant: tmp_path / variant / "data.yaml"
        for variant in ("original", "gaussian", "bbhe", "gaussian_bbhe")
    }
    for path in data.values():
        path.parent.mkdir()
        path.write_text("test: images\n")

    FakeModel.instances.clear()
    rows = evaluate_variants(
        checkpoint=checkpoint,
        variant_data=data,
        output_root=tmp_path / "model_eval",
        split="test",
        imgsz=1024,
        conf=0.25,
        iou=0.7,
        device="cpu",
        workers=0,
        model_factory=FakeModel,
    )

    assert len(FakeModel.instances) == 1
    model = FakeModel.instances[0]
    assert model.checkpoint == str(checkpoint)
    assert len(model.calls) == 4
    assert {call["imgsz"] for call in model.calls} == {1024}
    assert {call["conf"] for call in model.calls} == {0.25}
    assert {call["iou"] for call in model.calls} == {0.7}
    assert {call["device"] for call in model.calls} == {"cpu"}
    assert {row["variant"] for row in rows} == {
        "original",
        "gaussian",
        "bbhe",
        "gaussian_bbhe",
    }
    assert rows[0]["f1"] == 2 * 0.95 * 0.92 / (0.95 + 0.92)
    assert (tmp_path / "model_eval" / "model_metrics.csv").is_file()
