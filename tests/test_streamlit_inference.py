import numpy as np

import scripts.streamlit_dashboard as dashboard


class _Column:
    def __init__(self, calls):
        self.calls = calls

    def image(self, image, *, caption, width):
        self.calls.append((caption, image, width))


class _StreamlitStub:
    def __init__(self):
        self.calls = []

    def columns(self, count):
        assert count == 4
        return [_Column(self.calls) for _ in range(count)]


def test_inference_pair_runs_original_model_before_preprocessed_model(monkeypatch):
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    calls = []

    def fake_detect(model, input_image):
        calls.append(("detect", input_image.copy()))
        return input_image.copy(), len(calls)

    def fake_apply_candidate(input_image, candidate):
        calls.append(("preprocess", input_image.copy()))
        return input_image + 5

    monkeypatch.setattr(dashboard, "_detect", fake_detect)
    monkeypatch.setattr(dashboard, "apply_candidate", fake_apply_candidate)
    selected = {
        "module": "member1",
        "technique": "gaussian",
        "parameters": {},
    }

    result = dashboard._run_inference_pair(object(), image, selected)

    assert [call[0] for call in calls] == ["detect", "preprocess", "detect"]
    assert np.array_equal(calls[0][1], image)
    assert np.array_equal(calls[2][1], image + 5)
    assert result["original_model_detections"] == 1
    assert result["preprocessed_detections"] == 3


def test_render_inference_result_shows_original_and_preprocessed_detections():
    streamlit = _StreamlitStub()
    item = {
        "original": np.zeros((4, 4, 3), dtype=np.uint8),
        "original_model_result": np.ones((4, 4, 3), dtype=np.uint8),
        "processed": np.full((4, 4, 3), 2, dtype=np.uint8),
        "result": np.full((4, 4, 3), 3, dtype=np.uint8),
    }

    dashboard._render_inference_visual_result(streamlit, item)

    assert [call[0] for call in streamlit.calls] == [
        "Uploaded original",
        "Original model detection",
        "After selected preprocessing",
        "Preprocessed detection result",
    ]
    assert [int(call[1][0, 0, 0]) for call in streamlit.calls] == [0, 1, 2, 3]
