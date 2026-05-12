import json
import logging

from src.utils.paths import artefact_path

_logger = logging.getLogger(__name__)

_user_recs: dict[int, dict] = {}
_global_pop: list[int] = []
_product_names: dict[int, str] = {}
_loaded: bool = False
_TOP_N = 10
_FALLBACK_EXPLANATION = "Popular across the platform"


def load_artefacts() -> None:
    global _global_pop, _product_names, _loaded
    if _loaded:
        return

    recs_path = artefact_path("data", "Task 1", "modelling", "recommendations.json")
    expl_path = artefact_path("data", "Task 1", "modelling", "explanations.json")
    pop_path = artefact_path("data", "Task 1", "processed", "global_popularity.json")

    _logger.info("recommendation: loading recommendations.json")
    with open(recs_path, "r") as fh:
        recs_raw = json.load(fh)
    _logger.info("recommendation: loading explanations.json")
    with open(expl_path, "r") as fh:
        expl_raw = json.load(fh)
    _logger.info("recommendation: loading global_popularity.json")
    with open(pop_path, "r") as fh:
        _global_pop = json.load(fh)

    names: dict[int, str] = {}
    overlap_users = 0
    for user_id_str, buckets in recs_raw.items():
        expl_for_user = expl_raw.get(user_id_str, {})
        merged: dict[str, list[dict]] = {"reorder": [], "new_for_you": []}
        reorder_pids: set[int] = set()
        user_had_overlap = False
        for bucket in ("reorder", "new_for_you"):
            expl_by_pid = {
                e["product_id"]: e for e in expl_for_user.get(bucket, [])
            }
            for rec in buckets.get(bucket, []):
                pid = rec["product_id"]
                if bucket == "new_for_you" and pid in reorder_pids:
                    user_had_overlap = True
                    continue
                if bucket == "reorder":
                    reorder_pids.add(pid)
                name = rec.get("product_name", "")
                names[pid] = name
                expl_text = expl_by_pid.get(pid, {}).get("explanation_text", "")
                merged[bucket].append(
                    {
                        "product_id": pid,
                        "name": name,
                        "score": rec["score"],
                        "explanation": expl_text,
                        "bucket": bucket,
                    }
                )
        if user_had_overlap:
            overlap_users += 1
        _user_recs[int(user_id_str)] = merged

    if overlap_users:
        _logger.warning(
            "recommendation: dropped reorder-bucket pids from new_for_you for %d users (data quality)",
            overlap_users,
        )

    _product_names = names
    _loaded = True
    _logger.info(
        "recommendation: merged recs for %d users, %d product names",
        len(_user_recs),
        len(_product_names),
    )


def model_version() -> str:
    from src.model_management import service as mm

    return mm.version_for("task1a_knn")


def get_recommendations(instacart_user_id: int) -> dict:
    merged = _user_recs.get(instacart_user_id)
    if merged is not None:
        return {
            "user_id": instacart_user_id,
            "reorder": merged["reorder"][:_TOP_N],
            "new_for_you": merged["new_for_you"][:_TOP_N],
        }
    fallback = [
        {
            "product_id": pid,
            "name": _product_names.get(pid, ""),
            "score": 0.0,
            "explanation": _FALLBACK_EXPLANATION,
            "bucket": "new_for_you",
        }
        for pid in _global_pop[:_TOP_N]
    ]
    return {
        "user_id": instacart_user_id,
        "reorder": [],
        "new_for_you": fallback,
    }
