"""Resolve production model with strict MLflow-first priority."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from deployment.fastapi.config import AppConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

SourceResolvedFrom = Literal[
    "mlflow_alias_production",
    "mlflow_stage_production",
    "json_fallback",
]


class ModelResolutionError(RuntimeError):
    """Raised when the model source cannot be resolved."""


@dataclass(frozen=True)
class ResolvedModelMetadata:
    model_path: Path
    model_name: str
    model_version: str
    model_source: str
    run_id: str | None
    alias_or_stage: str | None
    source_resolved_from: SourceResolvedFrom
    metrics_core: dict[str, float] = field(default_factory=dict)
    best_params: dict[str, Any] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def resolve_model(config: AppConfig | None = None) -> ResolvedModelMetadata:
    """Resolve model metadata and artifact path with MLflow-first priority."""
    cfg = config or load_config()
    warnings: list[str] = []
    errors: list[str] = []

    client = _init_mlflow_client(cfg, warnings, errors)
    if client is not None:
        alias_result = _resolve_from_mlflow_alias(client, cfg, warnings, errors)
        if alias_result is not None:
            return alias_result

        stage_result = _resolve_from_mlflow_stage(client, cfg, warnings, errors)
        if stage_result is not None:
            return stage_result

    return _resolve_from_json_fallback(cfg, warnings, errors)


def _init_mlflow_client(cfg: AppConfig, warnings: list[str], errors: list[str]) -> Any | None:
    try:
        import mlflow  # type: ignore
        from mlflow.tracking import MlflowClient  # type: ignore

        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        logger.info("MLflow tracking URI set to: %s", cfg.mlflow_tracking_uri)
        return MlflowClient()
    except Exception as exc:  # pragma: no cover - depends on environment setup
        msg = f"MLflow client unavailable: {exc}"
        logger.warning(msg)
        warnings.append(msg)
        errors.append(msg)
        return None


def _resolve_from_mlflow_alias(
    client: Any,
    cfg: AppConfig,
    warnings: list[str],
    errors: list[str],
) -> ResolvedModelMetadata | None:
    logger.info(
        "Resolving model by MLflow alias '%s' for '%s'.",
        cfg.mlflow_model_alias,
        cfg.mlflow_model_name,
    )
    try:
        model_version = client.get_model_version_by_alias(
            name=cfg.mlflow_model_name,
            alias=cfg.mlflow_model_alias,
        )
    except Exception as exc:
        msg = (
            f"Could not resolve MLflow alias '{cfg.mlflow_model_alias}' "
            f"for model '{cfg.mlflow_model_name}': {exc}"
        )
        logger.warning(msg)
        warnings.append(msg)
        errors.append(msg)
        return None

    try:
        return _build_mlflow_result(
            client=client,
            cfg=cfg,
            model_version=model_version,
            source_resolved_from="mlflow_alias_production",
            alias_or_stage=f"alias:{cfg.mlflow_model_alias}",
            warnings=warnings,
        )
    except Exception as exc:
        msg = f"MLflow alias metadata missing or not parseable: {exc}"
        logger.warning(msg)
        warnings.append(msg)
        errors.append(msg)
        return None


def _resolve_from_mlflow_stage(
    client: Any,
    cfg: AppConfig,
    warnings: list[str],
    errors: list[str],
) -> ResolvedModelMetadata | None:
    logger.info("Resolving model by MLflow stage 'Production' for '%s'.", cfg.mlflow_model_name)
    model_versions: list[Any] = []

    try:
        model_versions = list(client.get_latest_versions(cfg.mlflow_model_name, stages=["Production"]))
    except Exception as exc:
        msg = f"Failed to query MLflow latest Production version: {exc}"
        logger.warning(msg)
        warnings.append(msg)
        errors.append(msg)

    if not model_versions:
        try:
            versions = client.search_model_versions(f"name='{cfg.mlflow_model_name}'")
            model_versions = [mv for mv in versions if str(getattr(mv, "current_stage", "")).lower() == "production"]
        except Exception as exc:
            msg = f"Could not search model versions in MLflow: {exc}"
            logger.warning(msg)
            warnings.append(msg)
            errors.append(msg)
            return None

    if not model_versions:
        msg = f"No Production stage model version found in MLflow for '{cfg.mlflow_model_name}'."
        logger.warning(msg)
        warnings.append(msg)
        errors.append(msg)
        return None

    sorted_versions = sorted(
        model_versions,
        key=lambda mv: int(str(getattr(mv, "version", "0")) or "0"),
        reverse=True,
    )

    for mv in sorted_versions:
        try:
            return _build_mlflow_result(
                client=client,
                cfg=cfg,
                model_version=mv,
                source_resolved_from="mlflow_stage_production",
                alias_or_stage="stage:Production",
                warnings=warnings,
            )
        except Exception as exc:
            msg = f"Production stage version metadata missing or not parseable: {exc}"
            logger.warning(msg)
            warnings.append(msg)
            errors.append(msg)

    return None


def _build_mlflow_result(
    client: Any,
    cfg: AppConfig,
    model_version: Any,
    source_resolved_from: SourceResolvedFrom,
    alias_or_stage: str,
    warnings: list[str],
) -> ResolvedModelMetadata:
    source = str(getattr(model_version, "source", "")).strip()
    if not source:
        raise ModelResolutionError("MLflow model metadata missing source field.")

    model_path = _source_to_path(source)
    if not model_path.exists():
        raise ModelResolutionError(
            f"MLflow source artifact does not exist at '{model_path}'."
        )
    model_path = _resolve_bundle_path_from_mlflow_source(model_path, warnings)

    description_payload = _parse_json_object(
        str(getattr(model_version, "description", "") or ""),
        context="MLflow model version description",
        warnings=warnings,
    )
    tags = _as_dict(getattr(model_version, "tags", {}) or {})
    business_version = str(
        tags.get("business_version")
        or description_payload.get("business_version")
        or getattr(model_version, "version", "unknown")
    )

    candidate_run_id = _normalize_optional(getattr(model_version, "run_id", None))
    run_payload = _resolve_run_payload(
        client=client,
        cfg=cfg,
        run_id=candidate_run_id,
        model_version=business_version,
    )

    metrics_core = run_payload["metrics_core"]
    params = run_payload["params"]
    best_params = run_payload["best_params"]

    if not metrics_core:
        metrics_core = _extract_metrics_core(_extract_description_metrics(description_payload))
    if not best_params:
        best_params = _extract_best_params_from_description(description_payload)

    logger.info(
        "Resolved model from MLflow (%s): name=%s, version=%s, run_id=%s",
        source_resolved_from,
        cfg.mlflow_model_name,
        business_version,
        run_payload["run_id"],
    )

    return ResolvedModelMetadata(
        model_path=model_path,
        model_name=cfg.mlflow_model_name,
        model_version=business_version,
        model_source=source,
        run_id=run_payload["run_id"],
        alias_or_stage=alias_or_stage,
        source_resolved_from=source_resolved_from,
        metrics_core=metrics_core,
        best_params=best_params,
        params=params,
        warnings=list(dict.fromkeys(warnings)),
    )


def _resolve_bundle_path_from_mlflow_source(source_path: Path, warnings: list[str]) -> Path:
    """Map MLflow model source directories to the bundle artifact expected by service loader."""
    if source_path.is_file():
        return source_path

    # In this project, model registry source often points to .../artifacts/serving_model
    # while inference loader expects .../artifacts/models/final_model_bundle.pkl.
    if source_path.is_dir():
        candidate_bundle = source_path.parent / "models" / "final_model_bundle.pkl"
        if candidate_bundle.exists() and candidate_bundle.is_file():
            msg = (
                "MLflow source points to directory artifact; using bundle artifact "
                f"'{candidate_bundle}' for inference."
            )
            logger.info(msg)
            # warnings.append(msg)
            return candidate_bundle

    raise ModelResolutionError(
        "Resolved MLflow artifact is not a loadable model bundle file and no compatible "
        "bundle fallback was found in the same run artifacts."
    )


def _resolve_run_payload(
    client: Any,
    cfg: AppConfig,
    run_id: str | None,
    model_version: str,
) -> dict[str, Any]:
    if run_id:
        payload = _fetch_run_payload(client, run_id, cfg.mlflow_model_name)
        if payload["run_id"]:
            return payload

    candidate_names = _candidate_model_names(cfg.mlflow_model_name)
    for model_name in candidate_names:
        filter_string = (
            f"params.metadata_model_version = '{model_version}' and "
            f"params.final_model_name = '{model_name}'"
        )
        try:
            runs = client.search_runs(
                experiment_ids=[cfg.mlflow_experiment_id],
                filter_string=filter_string,
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
        except Exception:
            runs = []

        if runs:
            selected_run_id = str(runs[0].info.run_id)
            logger.info(
                "Resolved run metadata from experiment %s using model_version=%s and final_model_name=%s -> run_id=%s",
                cfg.mlflow_experiment_id,
                model_version,
                model_name,
                selected_run_id,
            )
            return _fetch_run_payload(client, selected_run_id, cfg.mlflow_model_name)

    logger.warning(
        "Could not map MLflow model version '%s' to a run in experiment %s.",
        model_version,
        cfg.mlflow_experiment_id,
    )
    return {
        "run_id": None,
        "metrics_core": {},
        "params": {},
        "best_params": {},
    }


def _fetch_run_payload(client: Any, run_id: str, model_name: str) -> dict[str, Any]:
    try:
        run = client.get_run(run_id)
    except Exception as exc:
        logger.warning("Failed to fetch MLflow run '%s': %s", run_id, exc)
        return {
            "run_id": None,
            "metrics_core": {},
            "params": {},
            "best_params": {},
        }

    metrics = dict(run.data.metrics or {})
    params = {k: str(v) for k, v in (run.data.params or {}).items()}
    return {
        "run_id": run_id,
        "metrics_core": _extract_metrics_core(metrics),
        "params": params,
        "best_params": _extract_best_params_from_params(params, model_name),
    }


def _resolve_from_json_fallback(
    cfg: AppConfig,
    warnings: list[str],
    errors: list[str],
) -> ResolvedModelMetadata:
    logger.info("Falling back to JSON registry: %s", cfg.model_registry_path)

    entry = _load_json_registry_entry(cfg.model_registry_path)
    model_name = str(entry.get("model_name") or cfg.mlflow_model_name)
    model_version = str(entry.get("version") or "unknown")

    if cfg.model_path_fallback is not None:
        artifact_path = cfg.model_path_fallback
        logger.info("Using MODEL_PATH_FALLBACK override: %s", artifact_path)
    else:
        raw_path = entry.get("artifact_path")
        if not raw_path:
            raise ModelResolutionError(
                "Fallback JSON does not contain a valid artifact path (artifact_path missing)."
            )
        artifact_path = Path(str(raw_path))

    if not artifact_path.is_absolute():
        artifact_path = (cfg.repo_root / artifact_path).resolve()

    if not artifact_path.exists():
        raise ModelResolutionError(
            f"Fallback JSON does not contain a valid artifact path: '{artifact_path}'"
        )

    metrics = _as_dict(entry.get("metrics", {}))
    params = {k: str(v) for k, v in _as_dict(entry.get("params", {})).items()}
    best_params = _extract_best_params_from_json_entry(entry)

    logger.info("Resolved model from JSON fallback: name=%s, version=%s", model_name, model_version)
    return ResolvedModelMetadata(
        model_path=artifact_path,
        model_name=model_name,
        model_version=model_version,
        model_source=str(artifact_path),
        run_id=None,
        alias_or_stage=None,
        source_resolved_from="json_fallback",
        metrics_core=_extract_metrics_core(metrics),
        best_params=best_params,
        params=params,
        warnings=list(dict.fromkeys(warnings + errors)),
    )


def _load_json_registry_entry(registry_path: Path) -> dict[str, Any]:
    if not registry_path.exists():
        raise ModelResolutionError(
            f"Fallback JSON not found at '{registry_path}'."
        )

    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelResolutionError(
            f"Fallback JSON metadata missing or not parseable: {exc}"
        ) from exc

    production_ref = _as_dict(payload.get("production", {}))
    if not production_ref:
        raise ModelResolutionError("Fallback JSON does not define a production model reference.")

    model_name = str(production_ref.get("model_name", "")).strip()
    model_version = str(production_ref.get("version", "")).strip()
    if not model_name or not model_version:
        raise ModelResolutionError("Fallback JSON production reference is incomplete.")

    for model_entry in payload.get("models", []):
        entry = _as_dict(model_entry)
        if str(entry.get("model_name")) == model_name and str(entry.get("version")) == model_version:
            return entry

    raise ModelResolutionError(
        "Fallback JSON production model reference does not match any registry entry."
    )


def _source_to_path(source: str) -> Path:
    parsed = urlparse(source)
    if parsed.scheme == "file":
        unquoted = unquote(parsed.path)
        if re.match(r"^/[A-Za-z]:/", unquoted):
            unquoted = unquoted[1:]
        return Path(unquoted)
    if parsed.scheme in {"", None}:
        return Path(source)
    raise ModelResolutionError(
        f"Unsupported MLflow source URI scheme '{parsed.scheme}' for source '{source}'."
    )


def _extract_metrics_core(metrics: dict[str, Any]) -> dict[str, float]:
    ordered_keys = [
        "final_test_f1_macro",
        "final_test_accuracy",
        "final_test_auc_ovr",
        "final_test_precision_macro",
        "final_test_recall_macro",
        "final_valid_f1_macro",
        "final_valid_accuracy",
        "final_valid_auc_ovr",
        "f1_macro",
        "accuracy",
        "auc_ovr",
        "precision_macro",
        "recall_macro",
    ]
    core: dict[str, float] = {}
    for key in ordered_keys:
        if key in metrics:
            try:
                core[key] = float(metrics[key])
            except (TypeError, ValueError):
                continue
    return core


def _extract_best_params_from_params(params: dict[str, str], model_name: str) -> dict[str, Any]:
    best_params_all = params.get("best_params_all")
    if best_params_all:
        payload = _parse_json_object(best_params_all, context="best_params_all", warnings=[])
        if payload:
            for candidate in _candidate_model_names(model_name):
                candidate_params = payload.get(candidate)
                if isinstance(candidate_params, dict):
                    return candidate_params
            nested = payload.get("best_params")
            if isinstance(nested, dict):
                return nested
            if all(isinstance(v, (int, float, str, bool)) for v in payload.values()):
                return payload

    best_params = params.get("best_params")
    if best_params:
        payload = _parse_json_object(best_params, context="best_params", warnings=[])
        if payload:
            return payload

    return {}


def _extract_best_params_from_description(description_payload: dict[str, Any]) -> dict[str, Any]:
    metrics = _extract_description_metrics(description_payload)
    best_params = metrics.get("best_params")
    if isinstance(best_params, dict):
        return best_params
    return {}


def _extract_best_params_from_json_entry(entry: dict[str, Any]) -> dict[str, Any]:
    metrics = _as_dict(entry.get("metrics", {}))
    if isinstance(metrics.get("best_params"), dict):
        return metrics["best_params"]
    params = _as_dict(entry.get("params", {}))
    if isinstance(params.get("best_params"), dict):
        return params["best_params"]
    return {}


def _extract_description_metrics(description_payload: dict[str, Any]) -> dict[str, Any]:
    metrics = description_payload.get("metrics", {})
    if isinstance(metrics, dict):
        return metrics
    return {}


def _parse_json_object(text: str, context: str, warnings: list[str]) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
    except Exception as exc:
        warnings.append(f"{context} could not be parsed as JSON: {exc}")
        return {}
    if isinstance(parsed, dict):
        return parsed
    warnings.append(f"{context} is not a JSON object.")
    return {}


def _candidate_model_names(model_name: str) -> list[str]:
    candidates = [model_name]
    if model_name.endswith("_serving"):
        candidates.append(model_name.removesuffix("_serving"))
    else:
        candidates.append(f"{model_name}_serving")
    deduped: list[str] = []
    for item in candidates:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _normalize_optional(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    return text or None


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}
