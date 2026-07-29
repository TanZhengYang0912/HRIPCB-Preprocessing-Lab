"""Member 1 candidate generation for the generic preprocessing dashboard."""

from __future__ import annotations

from .filters import apply_bbhe, apply_gaussian_filter


def _number_label(value: float) -> str:
    return str(value).replace(".", "p")


def build_member1_candidates(config: dict) -> list[dict]:
    """Build a reasonable, serializable Gaussian/BBHE candidate matrix."""

    candidates = [{"id": "original", "module": "member1", "technique": "original", "parameters": {}}]
    gaussian_presets = config["gaussian_presets"]
    strengths = [float(value) for value in config["bbhe_strengths"]]
    for preset in gaussian_presets:
        parameters = {
            "gaussian_kernel_size": int(preset["kernel_size"]),
            "gaussian_sigma_x": float(preset["sigma_x"]),
        }
        candidates.append({
            "id": f"gaussian_{preset['id']}",
            "module": "member1",
            "technique": "gaussian",
            "parameters": parameters,
        })
    for strength in strengths:
        candidates.append({
            "id": f"bbhe_s{_number_label(strength)}",
            "module": "member1",
            "technique": "bbhe",
            "parameters": {"bbhe_strength": strength},
        })
    for preset in gaussian_presets:
        for strength in strengths:
            candidates.append({
                "id": f"combined_{preset['id']}_b{_number_label(strength)}",
                "module": "member1",
                "technique": "gaussian_bbhe",
                "parameters": {
                    "gaussian_kernel_size": int(preset["kernel_size"]),
                    "gaussian_sigma_x": float(preset["sigma_x"]),
                    "bbhe_strength": strength,
                },
            })
    return candidates


def apply_candidate(image, candidate):
    """Apply a serializable candidate definition to one BGR image."""

    technique = candidate["technique"]
    parameters = candidate.get("parameters", {})
    if technique == "original":
        return image.copy()
    if technique == "gaussian":
        return apply_gaussian_filter(image, int(parameters["gaussian_kernel_size"]), float(parameters["gaussian_sigma_x"]))
    if technique == "bbhe":
        return apply_bbhe(image, strength=float(parameters["bbhe_strength"]))
    if technique == "gaussian_bbhe":
        gaussian = apply_gaussian_filter(image, int(parameters["gaussian_kernel_size"]), float(parameters["gaussian_sigma_x"]))
        return apply_bbhe(gaussian, strength=float(parameters["bbhe_strength"]))
    raise ValueError(f"Unsupported Member 1 technique: {technique}")
