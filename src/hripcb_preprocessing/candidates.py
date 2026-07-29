"""Candidate grids and dispatch for all Member preprocessing modules."""

from __future__ import annotations

from .filters import (
    apply_agcwd,
    apply_bilateral_filter,
    apply_clahe,
    apply_median_filter,
    apply_multi_scale_retinex,
    apply_non_local_means,
)


def _label(value: float) -> str:
    return str(value).replace(".", "p")


def _candidate(candidate_id: str, module: str, technique: str, parameters: dict) -> dict:
    return {
        "id": candidate_id,
        "module": module,
        "technique": technique,
        "parameters": parameters,
    }


def build_candidates(module: str, config: dict) -> list[dict]:
    """Build one original, three single-method, and nine combined candidates."""

    if module == "member1":
        from hripcb_member1.sweep import build_member1_candidates

        return build_member1_candidates(config)

    candidates = [_candidate("original", module, "original", {})]
    if module == "member2":
        median = [int(value) for value in config["median_kernel_sizes"]]
        clahe = config["clahe_presets"]
        for kernel in median:
            candidates.append(_candidate(f"median_k{kernel}", module, "median", {"median_kernel_size": kernel}))
        for preset in clahe:
            parameters = {
                "clahe_clip_limit": float(preset["clip_limit"]),
                "clahe_tile_grid_width": int(preset["tile_grid_size"][0]),
                "clahe_tile_grid_height": int(preset["tile_grid_size"][1]),
            }
            candidates.append(_candidate(f"clahe_{preset['id']}", module, "clahe", parameters))
        for kernel in median:
            for preset in clahe:
                parameters = {
                    "median_kernel_size": kernel,
                    "clahe_clip_limit": float(preset["clip_limit"]),
                    "clahe_tile_grid_width": int(preset["tile_grid_size"][0]),
                    "clahe_tile_grid_height": int(preset["tile_grid_size"][1]),
                }
                candidates.append(_candidate(f"median_k{kernel}_clahe_{preset['id']}", module, "median_clahe", parameters))
        return candidates

    if module == "member3":
        bilateral = config["bilateral_presets"]
        gammas = [float(value) for value in config["agcwd_gammas"]]
        for preset in bilateral:
            parameters = {
                "bilateral_diameter": int(preset["diameter"]),
                "bilateral_sigma_color": float(preset["sigma_color"]),
                "bilateral_sigma_space": float(preset["sigma_space"]),
            }
            candidates.append(_candidate(f"bilateral_{preset['id']}", module, "bilateral", parameters))
        for gamma in gammas:
            candidates.append(_candidate(f"agcwd_g{_label(gamma)}", module, "agcwd", {"agcwd_gamma": gamma}))
        for preset in bilateral:
            for gamma in gammas:
                parameters = {
                    "bilateral_diameter": int(preset["diameter"]),
                    "bilateral_sigma_color": float(preset["sigma_color"]),
                    "bilateral_sigma_space": float(preset["sigma_space"]),
                    "agcwd_gamma": gamma,
                }
                candidates.append(_candidate(f"bilateral_{preset['id']}_agcwd_g{_label(gamma)}", module, "bilateral_agcwd", parameters))
        return candidates

    if module == "member4":
        nlm = config["nlm_presets"]
        msr = config["msr_presets"]
        nlm_processing_max_side = int(config.get("nlm_processing_max_side", 768))
        msr_processing_max_side = int(config.get("msr_processing_max_side", 768))
        for preset in nlm:
            parameters = {
                "nlm_h": float(preset["h"]),
                "nlm_h_color": float(preset["h_color"]),
                "nlm_template_window": int(preset["template_window"]),
                "nlm_search_window": int(preset["search_window"]),
                "nlm_processing_max_side": nlm_processing_max_side,
            }
            candidates.append(_candidate(f"nlm_{preset['id']}", module, "nlm", parameters))
        for preset in msr:
            sigmas = [float(value) for value in preset["sigmas"]]
            parameters = {f"msr_sigma_{index + 1}": value for index, value in enumerate(sigmas)}
            parameters["msr_processing_max_side"] = msr_processing_max_side
            candidates.append(_candidate(f"msr_{preset['id']}", module, "msr", parameters))
        for nlm_preset in nlm:
            for msr_preset in msr:
                sigmas = [float(value) for value in msr_preset["sigmas"]]
                parameters = {
                    "nlm_h": float(nlm_preset["h"]),
                    "nlm_h_color": float(nlm_preset["h_color"]),
                    "nlm_template_window": int(nlm_preset["template_window"]),
                    "nlm_search_window": int(nlm_preset["search_window"]),
                    "nlm_processing_max_side": nlm_processing_max_side,
                    "msr_processing_max_side": msr_processing_max_side,
                    **{f"msr_sigma_{index + 1}": value for index, value in enumerate(sigmas)},
                }
                candidates.append(_candidate(f"nlm_{nlm_preset['id']}_msr_{msr_preset['id']}", module, "nlm_msr", parameters))
        return candidates

    raise ValueError(f"Unsupported module: {module}")


def apply_candidate(image, candidate):
    """Apply one candidate definition to a BGR image."""

    technique = candidate["technique"]
    parameters = candidate.get("parameters", {})
    if technique == "original":
        return image.copy()
    if technique == "median":
        return apply_median_filter(image, int(parameters["median_kernel_size"]))
    if technique == "clahe":
        return apply_clahe(
            image,
            float(parameters["clahe_clip_limit"]),
            (int(parameters["clahe_tile_grid_width"]), int(parameters["clahe_tile_grid_height"])),
        )
    if technique == "median_clahe":
        return apply_clahe(
            apply_median_filter(image, int(parameters["median_kernel_size"])),
            float(parameters["clahe_clip_limit"]),
            (int(parameters["clahe_tile_grid_width"]), int(parameters["clahe_tile_grid_height"])),
        )
    if technique == "bilateral":
        return apply_bilateral_filter(
            image,
            int(parameters["bilateral_diameter"]),
            float(parameters["bilateral_sigma_color"]),
            float(parameters["bilateral_sigma_space"]),
        )
    if technique == "agcwd":
        return apply_agcwd(image, float(parameters["agcwd_gamma"]))
    if technique == "bilateral_agcwd":
        return apply_agcwd(
            apply_bilateral_filter(
                image,
                int(parameters["bilateral_diameter"]),
                float(parameters["bilateral_sigma_color"]),
                float(parameters["bilateral_sigma_space"]),
            ),
            float(parameters["agcwd_gamma"]),
        )
    if technique == "nlm":
        return apply_non_local_means(
            image,
            float(parameters["nlm_h"]),
            float(parameters["nlm_h_color"]),
            int(parameters["nlm_template_window"]),
            int(parameters["nlm_search_window"]),
            int(parameters.get("nlm_processing_max_side", 768)),
        )
    if technique == "msr":
        sigmas = tuple(parameters[f"msr_sigma_{index}"] for index in range(1, 4))
        return apply_multi_scale_retinex(image, sigmas, int(parameters.get("msr_processing_max_side", 768)))
    if technique == "nlm_msr":
        sigmas = tuple(parameters[f"msr_sigma_{index}"] for index in range(1, 4))
        return apply_multi_scale_retinex(
            apply_non_local_means(
                image,
                float(parameters["nlm_h"]),
                float(parameters["nlm_h_color"]),
                int(parameters["nlm_template_window"]),
                int(parameters["nlm_search_window"]),
                int(parameters.get("nlm_processing_max_side", 768)),
            ),
            sigmas,
            int(parameters.get("msr_processing_max_side", 768)),
        )
    if technique in {"gaussian", "bbhe", "gaussian_bbhe"}:
        from hripcb_member1.sweep import apply_candidate as apply_member1_candidate

        return apply_member1_candidate(image, candidate)
    raise ValueError(f"Unsupported technique: {technique}")
