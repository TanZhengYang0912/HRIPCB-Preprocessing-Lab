"""Candidate grids and dispatch for all Member preprocessing modules."""

from __future__ import annotations

from .filters import (
    apply_agcwd,
    apply_bilateral_filter,
    apply_homomorphic_filter,
    apply_multi_scale_retinex,
    apply_non_local_means,
    apply_wavelet_denoise,
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


def _wavelet_parameters(preset: dict) -> dict:
    levels = preset.get("wavelet_levels")
    return {
        "wavelet_name": str(preset.get("wavelet", "sym4")),
        "wavelet_method": str(preset.get("method", "BayesShrink")),
        "wavelet_mode": str(preset.get("mode", "soft")),
        "wavelet_levels": None if levels is None else int(levels),
    }


def _homomorphic_parameters(preset: dict) -> dict:
    return {
        "homomorphic_gamma_low": float(preset.get("gamma_low", 0.5)),
        "homomorphic_gamma_high": float(preset.get("gamma_high", 1.5)),
        "homomorphic_cutoff": float(preset.get("cutoff", 30.0)),
        "homomorphic_sharpness": float(preset.get("sharpness", 1.0)),
    }


def _apply_wavelet(image, parameters: dict):
    return apply_wavelet_denoise(
        image,
        wavelet=str(parameters["wavelet_name"]),
        method=str(parameters["wavelet_method"]),
        mode=str(parameters["wavelet_mode"]),
        wavelet_levels=parameters.get("wavelet_levels"),
    )


def _apply_homomorphic(image, parameters: dict):
    return apply_homomorphic_filter(
        image,
        gamma_low=float(parameters["homomorphic_gamma_low"]),
        gamma_high=float(parameters["homomorphic_gamma_high"]),
        cutoff=float(parameters["homomorphic_cutoff"]),
        sharpness=float(parameters.get("homomorphic_sharpness", 1.0)),
    )


def build_candidates(module: str, config: dict) -> list[dict]:
    """Build one original, three single-method, and nine combined candidates."""

    if module == "member1":
        from hripcb_member1.sweep import build_member1_candidates

        return build_member1_candidates(config)

    candidates = [_candidate("original", module, "original", {})]
    if module == "member2":
        wavelets = config["wavelet_presets"]
        homomorphics = config["homomorphic_presets"]
        for preset in wavelets:
            candidates.append(
                _candidate(f"wavelet_{preset['id']}", module, "wavelet", _wavelet_parameters(preset))
            )
        for preset in homomorphics:
            candidates.append(
                _candidate(
                    f"homomorphic_{preset['id']}",
                    module,
                    "homomorphic",
                    _homomorphic_parameters(preset),
                )
            )
        for wavelet_preset in wavelets:
            for homomorphic_preset in homomorphics:
                parameters = {
                    **_wavelet_parameters(wavelet_preset),
                    **_homomorphic_parameters(homomorphic_preset),
                }
                candidates.append(
                    _candidate(
                        f"wavelet_{wavelet_preset['id']}_homomorphic_{homomorphic_preset['id']}",
                        module,
                        "wavelet_homomorphic",
                        parameters,
                    )
                )
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
    if technique == "wavelet":
        return _apply_wavelet(image, parameters)
    if technique == "homomorphic":
        return _apply_homomorphic(image, parameters)
    if technique == "wavelet_homomorphic":
        return _apply_homomorphic(_apply_wavelet(image, parameters), parameters)
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
