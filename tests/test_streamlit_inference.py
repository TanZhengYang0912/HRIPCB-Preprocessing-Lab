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


def test_clear_image_uploads_advances_uploader_key():
    class Streamlit:
        session_state = {}

    streamlit = Streamlit()

    assert dashboard._image_upload_key(streamlit) == "image_inference_uploads_0"

    dashboard._clear_image_uploads(streamlit)

    assert streamlit.session_state["image_inference_uploader_version"] == 1
    assert dashboard._image_upload_key(streamlit) == "image_inference_uploads_1"


def test_recommended_preset_updates_widget_state_before_rendering_widgets():
    class Streamlit:
        session_state = {}

    streamlit = Streamlit()
    recommended = {
        "id": "member1-gaussian",
        "model_id": "yolov8s",
        "module": "member1",
        "technique": "gaussian_bbhe",
    }

    dashboard._queue_inference_preset(streamlit, recommended, key_prefix="infer")

    assert streamlit.session_state == {
        "infer_pending_preset": {
            "experiment": "member1-gaussian",
            "model": "yolov8s",
            "module": "member1",
            "technique": "gaussian_bbhe",
        }
    }

    dashboard._apply_pending_inference_preset(streamlit, key_prefix="infer")

    assert streamlit.session_state == {
        "infer_model": "yolov8s",
        "infer_module": "member1",
        "infer_technique": "gaussian_bbhe",
        "infer_filtering": "gaussian",
        "infer_contrast": "bbhe",
        "infer_technique_sync": "gaussian_bbhe",
        "infer_experiment": "member1-gaussian",
    }


def test_default_recommendation_seeds_filtering_and_contrast_for_image_and_video():
    recommended = {
        "id": "member5-tv",
        "model_id": "baseline",
        "module": "member5",
        "technique": "tv_top_black_hat",
    }

    for key_prefix in ("infer", "video"):
        class Streamlit:
            session_state = {}

        streamlit = Streamlit()
        dashboard._default_to_recommendation(streamlit, recommended, key_prefix=key_prefix)

        assert streamlit.session_state[f"{key_prefix}_filtering"] == "tv"
        assert streamlit.session_state[f"{key_prefix}_contrast"] == "top_black_hat"
        assert streamlit.session_state[f"{key_prefix}_technique_sync"] == "tv_top_black_hat"


def test_video_experiment_widget_avoids_duplicate_session_default():
    source = dashboard.Path("scripts/streamlit_dashboard.py").read_text(encoding="utf-8")

    assert 'if "video_experiment" not in st.session_state:' in source


def test_module_change_applies_that_modules_best_filtering_contrast_and_experiment():
    class Streamlit:
        def __init__(self):
            self.session_state = {}

    records = [
        {
            "id": "member4_best",
            "model_id": "baseline",
            "module": "member4",
            "technique": "nlm_msr",
            "split": "val",
            "evaluation_type": "ablation",
            "metrics": {"map50_95": 0.51},
        }
    ]

    for key_prefix in ("infer", "video"):
        streamlit = Streamlit()
        dashboard._apply_module_recommendation(streamlit, records, "member4", key_prefix=key_prefix)

        assert streamlit.session_state == {
            f"{key_prefix}_filtering": "nlm",
            f"{key_prefix}_contrast": "msr",
            f"{key_prefix}_technique": "nlm_msr",
            f"{key_prefix}_technique_sync": "nlm_msr",
            f"{key_prefix}_experiment": "member4_best",
            f"{key_prefix}_module_sync": "member4",
        }


def test_progress_update_reports_batch_position():
    class Progress:
        def __init__(self):
            self.calls = []

        def progress(self, value, *, text):
            self.calls.append((value, text))

    progress = Progress()

    dashboard._update_progress(progress, 3, 10, "image")
    dashboard._update_progress(progress, 10, 10, "image")

    assert progress.calls == [
        (0.3, "Processing image 3/10"),
        (1.0, "Processing image 10/10"),
    ]


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


def test_detect_uses_streaming_prediction_to_bound_memory(monkeypatch):
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    calls = []

    class Result:
        boxes = [object()]

        def plot(self):
            return image.copy()

    class Model:
        def predict(self, **kwargs):
            calls.append(kwargs)
            if kwargs.get("stream") is not True:
                raise AssertionError("inference must use stream=True")
            return iter([Result()])

    monkeypatch.setattr(dashboard, "select_device", lambda value: "cpu")

    plotted, detections = dashboard._detect(Model(), image)

    assert detections == 1
    assert plotted.shape == image.shape
    assert calls[0]["stream"] is True


def test_inference_pair_limits_working_image_size(monkeypatch):
    image = np.zeros((2400, 3200, 3), dtype=np.uint8)
    detected_shapes = []

    def fake_detect(model, input_image):
        detected_shapes.append(input_image.shape)
        return input_image.copy(), 0

    monkeypatch.setattr(dashboard, "_detect", fake_detect)
    monkeypatch.setattr(dashboard, "apply_candidate", lambda input_image, candidate: input_image.copy())

    dashboard._run_inference_pair(
        object(),
        image,
        {"module": "member2", "technique": "original", "parameters": {}},
    )

    assert detected_shapes == [(960, 1280, 3), (960, 1280, 3)]


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
