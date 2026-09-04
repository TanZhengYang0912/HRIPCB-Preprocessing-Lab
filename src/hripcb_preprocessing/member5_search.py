"""Durable, bounded validation search for TV + Top-hat/Black-hat."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from importlib.metadata import packages_distributions, version
from pathlib import Path

import cv2
import yaml

from hripcb_dashboard.dashboard import write_dashboard_html
from hripcb_member1.evaluation import select_device
from hripcb_member1.runner import _source_images

from .candidates import build_candidates
from .runner import _metric_view, _write_flat_csv, run_sweep


def _sync_directory(path: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_json(path: Path, payload: object) -> None:
    """Replace a JSON document only after its new contents reach disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(config: dict, key: str, low: float, high: float | None = None, *, integer=False):
    value = config[key]
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value < low
        or (high is not None and value > high)
        or (integer and not isinstance(value, int))
    ):
        raise ValueError(f"Invalid {key}: {value!r}")
    return value


def _validated_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("Member 5 config must be a mapping")
    result = {
        "jpeg_quality": 95, "image_metrics_max_side": 256,
        "primary_metric": "map50_95", "model_id": "baseline",
        "model_label": "Baseline YOLO", "training_preprocessing": "original",
        "evaluation_type": "ablation", **config,
    }
    required = {"module", "split", "checkpoint", "data_config", "imgsz", "conf", "iou", "device", "workers"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"Member 5 config missing keys: {', '.join(sorted(missing))}")
    allowed = required | {
        "dataset", "jpeg_quality", "image_metrics_max_side", "primary_metric", "model_id",
        "model_label", "training_preprocessing", "evaluation_type", "sample", "batch_size",
        "reuse_prepared", "tv_weights", "morphology_kernel_sizes", "top_hat_amounts", "black_hat_amounts",
    }
    unknown = result.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown Member 5 config keys: {', '.join(sorted(unknown))}")
    for key, expected in {
        "module": "member5", "split": "val", "primary_metric": "map50_95",
        "model_id": "baseline", "training_preprocessing": "original", "evaluation_type": "ablation",
    }.items():
        if result[key] != expected:
            raise ValueError(f"Member 5 {key} must be {expected!r}")
    if result.get("reuse_prepared", False):
        raise ValueError("reuse_prepared is unsupported; Member 5 resumes via progress.json")
    _number(result, "imgsz", 1, integer=True)
    _number(result, "workers", 0, integer=True)
    _number(result, "conf", 0, 1)
    _number(result, "iou", 0, 1)
    _number(result, "jpeg_quality", 1, 100, integer=True)
    _number(result, "image_metrics_max_side", 0, integer=True)
    if 0 < result["image_metrics_max_side"] < 7:
        raise ValueError("image_metrics_max_side must be zero or at least 7 for SSIM")
    if result.get("sample") is not None and not isinstance(result["sample"], str):
        raise ValueError("sample must be an image filename")
    if not isinstance(result["device"], str) or not result["device"].strip():
        raise ValueError("device must be a non-empty string")
    for key in ("checkpoint", "data_config"):
        if not isinstance(result[key], (str, Path)) or not str(result[key]):
            raise ValueError(f"{key} must be a file path")
        path = Path(result[key]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{key} not found: {path}")
        result[key] = str(path)
    return result


def _fingerprint(dataset: Path, config: dict, candidates: list[dict]) -> str:
    """Include content identities so changed inputs cannot silently reuse scores."""
    sources = _source_images(dataset, "val")
    sample = config.get("sample")
    if sample and sample not in {source.name for source in sources}:
        raise FileNotFoundError(f"Sample image not found in {dataset / 'val' / 'images'}: {sample}")
    data = yaml.safe_load(Path(config["data_config"]).read_text(encoding="utf-8"))
    try:
        if not isinstance(data, dict):
            raise ValueError("expected a mapping")
        nc, names = data.get("nc"), data.get("names")
        if isinstance(nc, bool) or not isinstance(nc, int) or nc < 1:
            raise ValueError("nc must be a positive integer")
        if not isinstance(names, (list, dict)) or len(names) != nc:
            raise ValueError("names must contain exactly nc class names")
        if isinstance(names, dict) and {int(key) for key in names} != set(range(nc)):
            raise ValueError("names keys must cover 0 through nc - 1")
        if any(not isinstance(name, str) or not name for name in (names.values() if isinstance(names, dict) else names)):
            raise ValueError("class names must be non-empty strings")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Malformed data_config: {config['data_config']} ({error})") from error
    files = []
    for source in sources:
        label = dataset / "val" / "labels" / f"{source.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Label file not found: {label}")
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or min(image.shape[:2]) < 7:
            raise ValueError(f"Invalid image (at least 7 x 7 pixels required for SSIM): {source}")
        if min(_metric_view(image, config["image_metrics_max_side"]).shape[:2]) < 7:
            raise ValueError(f"image_metrics_max_side makes the SSIM image too small: {source}")
        files.append((source.name, _digest(source), _digest(label)))
    # Batch size and retention change resource use only, never detector inputs.
    effective = {key: value for key, value in config.items() if key not in {"batch_size", "keep_variants"}}
    distributions = packages_distributions()
    payload = {
        "schema": 1, "config": effective, "candidates": candidates,
        "dataset": str(dataset), "files": files,
        "checkpoint": _digest(Path(config["checkpoint"])),
        "data_config": _digest(Path(config["data_config"])),
        "libraries": {
            package: {name: version(name) for name in distributions.get(package, [])}
            for package in ("numpy", "skimage", "torch", "torchvision", "ultralytics")
        },
        "resolved_device": select_device(config["device"]),
        "opencv": cv2.__version__,
        "implementation": {
            path.name: _digest(path) for path in (
                Path(__file__), Path(__file__).with_name("filters.py"),
                Path(__file__).with_name("candidates.py"), Path(__file__).with_name("runner.py"),
                Path(__file__).parents[1] / "hripcb_member1" / "evaluation.py",
                Path(__file__).parents[1] / "hripcb_member1" / "metrics.py",
            )
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _batch_path(output: Path, start: int) -> Path:
    batches = output / "batches"
    batch = batches / f"batch_{start:04d}"
    if batches.resolve() != batches or batch.resolve() != batch:
        raise ValueError(f"Unsafe batch directory for cleanup: {batch}")
    return batch


def _validate_artifact_paths(root: Path, candidates: list[dict]) -> None:
    """Reject redirected artifacts before reusing a run's preserved directories."""
    directories = [root / name for name in ("previews", "batches")]
    files = [root / name for name in (
        "progress.json", "results.json", "results.csv", "summary.json", "dashboard.html", "run_manifest.json",
    )] + [root / "previews" / f"{candidate['id']}.jpg" for candidate in candidates]
    for path in directories + files:
        try:
            unsafe = path.is_symlink() or path.resolve() != path
        except (OSError, RuntimeError):
            unsafe = True
        if unsafe or (path.exists() and (not path.is_dir() if path in directories else not path.is_file())):
            raise ValueError(f"Unsafe Member 5 artifact: {path}; choose a new output directory")


def _copy_preview(source: Path, destination: Path) -> None:
    """Atomically retain a preview without following a destination file symlink."""
    if source.is_symlink() or source.resolve() != source or destination.is_symlink() or destination.parent.resolve() != destination.parent:
        raise ValueError("Unsafe Member 5 preview; choose a new output directory")
    descriptor, temporary = tempfile.mkstemp(prefix=".preview-", dir=destination.parent)
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _cleanup_batch(batch: Path) -> None:
    """Remove only the two bulky staging trees, never previews or results."""
    batch = batch.resolve()
    for name in ("variants", "model_eval"):
        target = batch / name
        if target.is_symlink() or target.resolve().parent != batch:
            raise ValueError(f"Unsafe cleanup target: {target}")
        if target.is_dir():
            shutil.rmtree(target)


def _check_records(records: list[dict], candidates: list[dict]) -> None:
    if not isinstance(records, list) or len(records) != len(candidates):
        raise ValueError("Incomplete Member 5 results; choose a new output directory")
    for record, candidate in zip(records, candidates):
        if not isinstance(record, dict) or not isinstance(record.get("metrics"), dict):
            raise ValueError("Malformed Member 5 results; choose a new output directory")
        if any(record.get(key) != candidate[key] for key in ("id", "module", "technique", "parameters")):
            raise ValueError("Member 5 results do not match candidates; choose a new output directory")
        if record.get("preview") != f"previews/{candidate['id']}.jpg":
            raise ValueError("Malformed Member 5 preview path; choose a new output directory")
        if record.get("split") != "val":
            raise ValueError("Member 5 result split must be val")
        for key in ("map50_95", "map50", "precision", "recall", "f1"):
            value = record.get("metrics", {}).get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Missing or invalid detection metric: {key}; choose a new output directory")


def _summary(records: list[dict], total: int) -> dict:
    def ranked(technique):
        return sorted(
            (record for record in records if record["technique"] == technique),
            key=lambda record: (float(record["metrics"]["map50_95"]), record["id"]),
            reverse=True,
        )
    originals = ranked("original")
    tv = ranked("tv")
    morphology = ranked("top_black_hat")
    combined = ranked("tv_top_black_hat")
    original_score = originals[0]["metrics"]["map50_95"] if originals else None
    return {
        "status": "complete" if len(records) == total else "running",
        "primary_metric": "map50_95", "selection_split": "val",
        "candidate_count": len(records), "expected_candidate_count": total,
        "best_tv": tv[0] if tv else None,
        "best_morphology": morphology[0] if morphology else None,
        "best_combined": combined[0] if combined else None,
        "ranked_combined": combined,
        "original_map50_95": original_score,
        "combined_improvement_vs_original": (
            combined[0]["metrics"]["map50_95"] - original_score
            if combined and original_score is not None else None
        ),
    }


def _publish_local(output: Path, records: list[dict], total: int) -> None:
    _atomic_json(output / "results.json", records)
    _write_flat_csv(output / "results.csv", records)
    _atomic_json(output / "summary.json", _summary(records, total))
    write_dashboard_html(output, records, title="Member 5 / TV + Top-hat/Black-hat", primary_metric="map50_95")


def _merge_project(output: Path, project_results: Path, records: list[dict]) -> None:
    # Replacing only validation ablations also preserves official test records
    # should a frozen Member 5 winner be tested in a later, separate task.
    existing = json.loads(project_results.read_text(encoding="utf-8")) if project_results.is_file() else []
    if not isinstance(existing, list):
        raise ValueError(f"Project results must be a list: {project_results}")
    retained = [record for record in existing if not (
        record.get("module") == "member5" and record.get("split") == "val"
        and record.get("evaluation_type", "ablation") == "ablation"
    )]
    merged = retained + [{
        **record, "preview": os.path.relpath(output / record["preview"], project_results.parent),
        "source_run": str(output),
    } for record in records]
    _atomic_json(project_results, merged)
    from scripts.build_project_dashboard import write_project_reports

    write_project_reports(project_results.parent, merged, source_files=[str(output / "results.json")])


def run_search(
    dataset: Path, output: Path, config: dict, *, batch_size: int | None = None,
    keep_variants: bool = False,
    project_results: Path = Path("runs/project_validation_comparison/results.json"),
) -> Path:
    """Evaluate a validation-only grid, resuming committed batches automatically."""
    config = _validated_config(config)
    batch_size = config.get("batch_size", 2) if batch_size is None else batch_size
    _number({"batch_size": batch_size}, "batch_size", 1, integer=True)
    candidates = build_candidates("member5", config)
    dataset, output, project_results = Path(dataset).resolve(), Path(output).resolve(), Path(project_results).resolve()
    if project_results.is_relative_to(output):
        raise ValueError("Project results must be separate from the Member 5 output")
    _validate_artifact_paths(output, candidates)
    fingerprint = _fingerprint(dataset, config, candidates)
    ids = [candidate["id"] for candidate in candidates]
    progress_path = output / "progress.json"
    if progress_path.is_file():
        try:
            state = json.loads(progress_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("Invalid Member 5 progress; choose a new output directory") from error
        if not isinstance(state, dict) or not isinstance(state.get("records"), list):
            raise ValueError("Invalid Member 5 progress; choose a new output directory")
        if state.get("fingerprint") != fingerprint or state.get("candidate_ids") != ids:
            raise ValueError("Member 5 configuration or inputs changed; choose a new output directory")
        records = state.get("records", [])
        _check_records(records, candidates[:len(records)])
        for record in records:
            if not (output / record["preview"]).is_file():
                raise ValueError(
                    f"Missing committed preview: {output / record['preview']}; "
                    "restore the batch preview or choose a new output directory"
                )
        if state.get("completed_ids") != ids[:len(records)] or not isinstance(state.get("status"), str) or state["status"] not in {"running", "complete"}:
            raise ValueError("Invalid Member 5 progress; choose a new output directory")
        if (state["status"] == "complete") != (len(records) == len(candidates)):
            raise ValueError("Invalid Member 5 progress status; choose a new output directory")
        starts = state.get("batch_starts")
        if (
            not isinstance(starts, list)
            or any(type(start) is not int or start < 0 or start >= len(records) for start in starts)
            or starts != sorted(set(starts))
            or bool(starts) != bool(records)
            or (starts and starts[0] != 0)
        ):
            raise ValueError("Invalid Member 5 batch progress; choose a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output has no Member 5 progress; choose a new output directory")
        state = {
            "fingerprint": fingerprint, "candidate_ids": ids, "completed_ids": [],
            "status": "running", "records": [], "batch_starts": [],
        }
        records = []
        _atomic_json(progress_path, state)
    if not keep_variants:
        for start in state["batch_starts"]:
            if not isinstance(start, int) or start < 0 or start >= len(records):
                raise ValueError("Invalid batch progress; choose a new output directory")
            _cleanup_batch(_batch_path(output, start))
    while len(records) < len(candidates):
        start = len(records)
        batch = _batch_path(output, start)
        _validate_artifact_paths(batch, candidates)
        # Staging left by an interrupted evaluation is recomputed from originals.
        _cleanup_batch(batch)
        selected = candidates[start:start + batch_size]
        run_sweep(dataset, batch, config, candidates=selected)
        new_records = json.loads((batch / "results.json").read_text(encoding="utf-8"))
        _check_records(new_records, selected)
        (output / "previews").mkdir(exist_ok=True)
        for record in new_records:
            preview = batch / "previews" / f"{record['id']}.jpg"
            destination = output / "previews" / preview.name
            _copy_preview(preview, destination)
            record["preview"] = str(destination.relative_to(output))
        _sync_directory(output / "previews")
        completed = records + new_records
        _publish_local(output, completed, len(candidates))
        state = {
            **state, "records": completed, "completed_ids": ids[:len(completed)],
            "batch_starts": [*state["batch_starts"], start],
            "status": "complete" if len(completed) == len(candidates) else "running",
        }
        _atomic_json(progress_path, state)
        records = completed
        if not keep_variants:
            _cleanup_batch(batch)
        print(f"Member 5: {len(records)}/{len(candidates)} candidates saved", flush=True)
    # Regenerate derived outputs after a crash and retry final publication without
    # repeating any evaluation. progress.json is the authoritative commit record.
    if state["status"] != "complete":
        raise ValueError("Invalid Member 5 progress status; choose a new output directory")
    _publish_local(output, records, len(candidates))
    _merge_project(output, project_results, records)
    return output / "summary.json"
