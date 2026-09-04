"""Exercise real batch persistence with tiny images and a fake detector only."""

import importlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from hripcb_preprocessing import runner


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    images = dataset / "val" / "images"
    labels = dataset / "val" / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    image = np.random.default_rng(5).integers(0, 256, (16, 18, 3), dtype=np.uint8)
    assert cv2.imwrite(str(images / "sample.png"), image)
    (labels / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"frozen fake checkpoint")
    data_config = tmp_path / "data.yaml"
    data_config.write_text(yaml.safe_dump({"nc": 1, "names": {0: "defect"}}))
    config = {
        "module": "member5", "split": "val", "checkpoint": str(checkpoint),
        "data_config": str(data_config), "model_id": "baseline",
        "imgsz": 1024, "conf": 0.25, "iou": 0.7, "device": "cpu", "workers": 0,
        "tv_weights": [0.01], "morphology_kernel_sizes": [5],
        "top_hat_amounts": [0.5], "black_hat_amounts": [0.5],
    }
    project = tmp_path / "project" / "results.json"
    project.parent.mkdir()
    existing = [
        {"id": "keep1", "module": "member1", "technique": "gaussian_bbhe", "split": "val", "metrics": {"map50_95": 0.5}},
        {"id": "official", "module": "member5", "technique": "tv_top_black_hat", "evaluation_type": "official_test", "split": "test", "metrics": {"map50_95": 0.8}},
    ]
    project.write_text(json.dumps(existing))
    calls = []

    def evaluate(**kwargs):
        assert kwargs["split"] == "val"
        assert kwargs["imgsz"] == 1024
        assert kwargs["conf"] == 0.25
        assert kwargs["iou"] == 0.7
        calls.append(list(kwargs["variant_data"]))
        return [
            {"variant": candidate_id, "map50_95": 0.9 if candidate_id == "original" else 0.4,
             "map50": 0.7, "precision": 0.6, "recall": 0.5, "f1": 6 / 11}
            for candidate_id in kwargs["variant_data"]
        ]

    monkeypatch.setattr(runner, "evaluate_variants", evaluate)
    return dataset, tmp_path / "output", config, project, calls


def search_module():
    return importlib.import_module("hripcb_preprocessing.member5_search")


def test_batches_persist_resume_and_select_only_combined(experiment, monkeypatch):
    module = search_module()
    dataset, output, config, project, calls = experiment
    evaluate = runner.evaluate_variants

    def fail_second_batch(**kwargs):
        if calls:
            raise RuntimeError("detector stopped")
        return evaluate(**kwargs)

    before = project.read_bytes()
    monkeypatch.setattr(runner, "evaluate_variants", fail_second_batch)
    with pytest.raises(RuntimeError, match="detector stopped"):
        module.run_search(dataset, output, config, batch_size=2, project_results=project)
    state = json.loads((output / "progress.json").read_text())
    assert state["status"] == "running"
    assert len(state["completed_ids"]) == 2
    assert project.read_bytes() == before
    assert not (output / "batches" / "batch_0000" / "variants").exists()
    assert (output / "batches" / "batch_0002" / "variants").exists()
    monkeypatch.setattr(runner, "evaluate_variants", evaluate)
    summary_path = module.run_search(dataset, output, config, batch_size=1, project_results=project)
    state = json.loads((output / "progress.json").read_text())
    records = json.loads((output / "results.json").read_text())
    summary = json.loads(summary_path.read_text())
    assert state["status"] == "complete"
    assert len(records) == len(set(state["completed_ids"])) == 4
    assert [key for batch in calls for key in batch] == state["candidate_ids"]
    assert summary["best_combined"]["technique"] == "tv_top_black_hat"
    assert summary["combined_improvement_vs_original"] == pytest.approx(-0.5)
    assert all((output / record["preview"]).is_file() for record in records)
    assert not list(output.glob("batches/*/variants"))
    assert not list(output.glob("batches/*/model_eval"))
    merged = json.loads(project.read_text())
    assert {"keep1", "official"} <= {record["id"] for record in merged}
    assert len(merged) == 6
    assert (project.parent / "selection.json").is_file()
    call_count = len(calls)
    module.run_search(dataset, output, config, project_results=project)
    assert len(calls) == call_count
    assert len(json.loads(project.read_text())) == 6


def test_keep_variants_preserves_generated_data(experiment):
    dataset, output, config, project, _ = experiment
    search_module().run_search(dataset, output, config, keep_variants=True, project_results=project)
    assert list(output.glob("batches/*/variants/*/images/sample.png"))
    assert list(output.glob("batches/*/model_eval/*/data.yaml"))


@pytest.mark.parametrize("change", ["conf", "checkpoint", "label", "config"])
def test_resume_rejects_changed_inputs(experiment, change):
    dataset, output, config, project, calls = experiment
    module = search_module()
    module.run_search(dataset, output, config, project_results=project)
    before = project.read_bytes()
    if change == "conf":
        config["conf"] = 0.3
    elif change == "checkpoint":
        Path(config["checkpoint"]).write_bytes(b"new checkpoint")
    elif change == "label":
        (dataset / "val" / "labels" / "sample.txt").write_text("")
    else:
        config["jpeg_quality"] = 70
    call_count = len(calls)
    with pytest.raises(ValueError, match="new output directory"):
        module.run_search(dataset, output, config, project_results=project)
    assert len(calls) == call_count
    assert project.read_bytes() == before


@pytest.mark.parametrize("missing", ["label", "image", "checkpoint", "data_config"])
def test_missing_inputs_fail_before_evaluation(experiment, missing):
    dataset, output, config, project, calls = experiment
    target = {
        "label": dataset / "val" / "labels" / "sample.txt",
        "image": dataset / "val" / "images" / "sample.png",
        "checkpoint": Path(config["checkpoint"]), "data_config": Path(config["data_config"]),
    }[missing]
    before = project.read_bytes()
    target.unlink()
    with pytest.raises(FileNotFoundError, match=str(target.parent)):
        search_module().run_search(dataset, output, config, project_results=project)
    assert not calls
    assert project.read_bytes() == before


def test_write_failure_keeps_batch_and_does_not_commit_progress(experiment, monkeypatch):
    module = search_module()
    dataset, output, config, project, _ = experiment
    atomic_write = module._atomic_json
    before = project.read_bytes()

    def fail_results(path, payload):
        if Path(path) == output / "results.json":
            raise OSError("disk full")
        atomic_write(path, payload)

    monkeypatch.setattr(module, "_atomic_json", fail_results)
    with pytest.raises(OSError, match="disk full"):
        module.run_search(dataset, output, config, batch_size=2, project_results=project)
    assert json.loads((output / "progress.json").read_text())["completed_ids"] == []
    assert (output / "batches" / "batch_0000" / "variants").exists()
    assert project.read_bytes() == before
    monkeypatch.setattr(module, "_atomic_json", atomic_write)
    module.run_search(dataset, output, config, project_results=project)
    assert json.loads((output / "progress.json").read_text())["status"] == "complete"


def test_cleanup_rejects_symlink_escape_and_preserves_metadata(tmp_path):
    module = search_module()
    batch = tmp_path / "batch"
    batch.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep")
    (batch / "variants").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="cleanup"):
        module._cleanup_batch(batch)
    assert sentinel.read_text() == "keep"
    (batch / "variants").unlink()
    for name in ("variants", "model_eval", "previews"):
        (batch / name).mkdir()
    (batch / "results.json").write_text("[]")
    module._cleanup_batch(batch)
    assert (batch / "previews").is_dir()
    assert (batch / "results.json").is_file()


@pytest.mark.parametrize("key,value", [("split", "test"), ("workers", -1), ("imgsz", 0), ("conf", float("nan")), ("jpeg_quality", 101), ("primary_metric", "mean_ssim"), ("image_metrics_max_side", 4), ("technique", "unsupported")])
def test_invalid_config_rejected_before_processing(experiment, key, value):
    dataset, output, config, project, calls = experiment
    config[key] = value
    with pytest.raises(ValueError, match=key):
        search_module().run_search(dataset, output, config, project_results=project)
    assert not calls


def test_runner_accepts_explicit_candidate_batch(experiment):
    dataset, output, config, _, calls = experiment
    runner.run_sweep(dataset, output, config, candidates=[
        {"id": "original", "module": "member5", "technique": "original", "parameters": {}}
    ])
    assert calls == [["original"]]
    assert len(json.loads((output / "results.json").read_text())) == 1


def test_failed_progress_commit_recomputes_uncommitted_batch(experiment, monkeypatch):
    module = search_module()
    dataset, output, config, project, calls = experiment
    atomic_write = module._atomic_json
    before = project.read_bytes()

    def fail_commit(path, payload):
        if Path(path) == output / "progress.json" and payload["completed_ids"]:
            raise OSError("progress write failed")
        atomic_write(path, payload)

    monkeypatch.setattr(module, "_atomic_json", fail_commit)
    with pytest.raises(OSError, match="progress write failed"):
        module.run_search(dataset, output, config, batch_size=2, project_results=project)
    assert len(json.loads((output / "results.json").read_text())) == 2
    assert json.loads((output / "progress.json").read_text())["completed_ids"] == []
    assert (output / "batches" / "batch_0000" / "variants").exists()
    assert project.read_bytes() == before
    first = calls[0]
    monkeypatch.setattr(module, "_atomic_json", atomic_write)
    module.run_search(dataset, output, config, batch_size=2, project_results=project)
    assert calls[1] == first
    assert len(json.loads(project.read_text())) == 6


def test_failed_project_publication_retries_without_evaluation(experiment, monkeypatch):
    module = search_module()
    dataset, output, config, project, calls = experiment
    atomic_write = module._atomic_json
    before = project.read_bytes()

    def fail_project(path, payload):
        if Path(path) == project:
            raise OSError("project write failed")
        atomic_write(path, payload)

    monkeypatch.setattr(module, "_atomic_json", fail_project)
    with pytest.raises(OSError, match="project write failed"):
        module.run_search(dataset, output, config, project_results=project)
    assert json.loads((output / "progress.json").read_text())["status"] == "complete"
    assert project.read_bytes() == before
    before_count = len(calls)
    monkeypatch.setattr(module, "_atomic_json", atomic_write)
    module.run_search(dataset, output, config, project_results=project)
    assert len(calls) == before_count
    assert len(json.loads(project.read_text())) == 6


def test_project_path_cannot_overwrite_run_state(experiment):
    dataset, output, config, _, calls = experiment
    with pytest.raises(ValueError, match="separate"):
        search_module().run_search(dataset, output, config, project_results=output / "progress.json")
    assert not calls


def test_invalid_image_fails_before_any_evaluation(experiment):
    dataset, output, config, project, calls = experiment
    (dataset / "val" / "images" / "sample.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Invalid image"):
        search_module().run_search(dataset, output, config, project_results=project)
    assert not calls


@pytest.mark.parametrize("data", [{"nc": "bad", "names": ["defect"]}, {"nc": 2, "names": ["defect"]}, {"nc": 1, "names": {9: "defect"}}])
def test_malformed_data_config_fails_before_evaluation(experiment, data):
    dataset, output, config, project, calls = experiment
    Path(config["data_config"]).write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="data_config"):
        search_module().run_search(dataset, output, config, project_results=project)
    assert not calls


def test_summary_tie_break_matches_shared_dashboard_selection():
    from hripcb_dashboard.filtering import best_by_module

    records = [
        {"id": key, "module": "member5", "technique": "tv_top_black_hat", "split": "val", "metrics": {"map50_95": 0.5}}
        for key in ("combo_a", "combo_b")
    ]
    summary = search_module()._summary(records, 2)
    assert summary["best_combined"]["id"] == best_by_module(records)[0]["id"]


@pytest.mark.parametrize("dependency", ["torch", "ultralytics-opencv-headless"])
def test_detector_dependency_change_rejects_resume(experiment, monkeypatch, dependency):
    module = search_module()
    dataset, output, config, project, calls = experiment
    module.run_search(dataset, output, config, project_results=project)
    original_version = module.version
    monkeypatch.setattr(module, "version", lambda name: "changed" if name == dependency else original_version(name))
    before_count = len(calls)
    with pytest.raises(ValueError, match="new output directory"):
        module.run_search(dataset, output, config, project_results=project)
    assert len(calls) == before_count


@pytest.mark.parametrize("location", ["output_preview_dir", "batch_preview_dir", "preview_file"])
def test_resume_rejects_symlinked_preview_outputs_before_evaluation(experiment, location, monkeypatch):
    module = search_module()
    dataset, output, config, project, calls = experiment
    evaluate = runner.evaluate_variants

    def fail_second_batch(**kwargs):
        if calls:
            raise RuntimeError("detector stopped")
        return evaluate(**kwargs)

    monkeypatch.setattr(runner, "evaluate_variants", fail_second_batch)
    with pytest.raises(RuntimeError, match="detector stopped"):
        module.run_search(dataset, output, config, batch_size=2, project_results=project)
    monkeypatch.setattr(runner, "evaluate_variants", evaluate)
    state_path = output / "progress.json"
    state = json.loads(state_path.read_text())
    next_id = state["candidate_ids"][len(state["records"])]

    outside = output.parent / "outside"
    outside.mkdir()
    sentinel = outside / f"{next_id}.jpg"
    sentinel.write_bytes(b"outside sentinel")
    if location == "output_preview_dir":
        shutil.rmtree(output / "previews")
        (output / "previews").symlink_to(outside, target_is_directory=True)
    elif location == "batch_preview_dir":
        batch_previews = output / "batches" / f"batch_{len(state['records']):04d}" / "previews"
        shutil.rmtree(batch_previews)
        batch_previews.symlink_to(outside, target_is_directory=True)
    else:
        destination = output / "previews" / f"{next_id}.jpg"
        destination.symlink_to(sentinel)

    before_calls = len(calls)
    with pytest.raises(ValueError, match="output directory"):
        module.run_search(dataset, output, config, project_results=project)
    assert len(calls) == before_calls
    assert sentinel.read_bytes() == b"outside sentinel"


def test_running_progress_with_all_records_is_rejected_before_project_merge(experiment):
    module = search_module()
    dataset, output, config, project, _ = experiment
    module.run_search(dataset, output, config, project_results=project)
    progress_path = output / "progress.json"
    state = json.loads(progress_path.read_text())
    assert len(state["records"]) == len(state["candidate_ids"])
    state["status"] = "running"
    progress_path.write_text(json.dumps(state))
    before = project.read_bytes()

    with pytest.raises(ValueError, match="Invalid Member 5 progress"):
        module.run_search(dataset, output, config, project_results=project)
    assert project.read_bytes() == before


@pytest.mark.parametrize("tamper", ["root", "records_none", "metrics_none", "batch_starts"])
def test_malformed_progress_fails_with_new_output_directory(experiment, tamper):
    module = search_module()
    dataset, output, config, project, _ = experiment
    module.run_search(dataset, output, config, project_results=project)
    progress_path = output / "progress.json"
    state = json.loads(progress_path.read_text())
    if tamper == "root":
        progress_path.write_text("[]")
    elif tamper == "records_none":
        state["records"] = None
        module._atomic_json(progress_path, state)
    elif tamper == "metrics_none":
        state["records"][0]["metrics"] = None
        module._atomic_json(progress_path, state)
    else:
        state["batch_starts"] = [True]
        module._atomic_json(progress_path, state)
    before = project.read_bytes()

    with pytest.raises(ValueError, match="new output directory"):
        module.run_search(
            dataset,
            output,
            config,
            keep_variants=tamper == "batch_starts",
            project_results=project,
        )
    assert project.read_bytes() == before


def test_missing_committed_preview_cannot_be_republished(experiment):
    module = search_module()
    dataset, output, config, project, calls = experiment
    module.run_search(dataset, output, config, project_results=project)
    (output / "previews" / "original.jpg").unlink()
    before_calls, before_project = len(calls), project.read_bytes()
    with pytest.raises(ValueError, match="Missing committed preview"):
        module.run_search(dataset, output, config, project_results=project)
    assert len(calls) == before_calls
    assert project.read_bytes() == before_project
