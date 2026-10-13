"""Per-episode metric computation and aggregation for REVERIE-AIM."""

import math

import numpy as np

from . import constants as C


def min_viewpoint_distance(sim, position, view_points) -> float:
    """Geodesic distance from `position` to the nearest ObjectNav viewpoint.

    Returns math.inf when no viewpoint is reachable (counted as failure).
    """
    return min(sim.geodesic_distance(position, vp["agent_state"]["position"])
               for vp in view_points)


def gt_bbox_from_semantic(semantic: np.ndarray, obj_id: int):
    """Tight box [x1, y1, x2, y2] around the target instance, or None if absent.

    Coordinates are inclusive pixel indices, and area is (x2 - x1) * (y2 - y1),
    matching the convention used for the paper numbers.
    """
    mask = semantic == obj_id
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def box_area(box) -> float:
    return float((box[2] - box[0]) * (box[3] - box[1]))


def oracle_visible(gt_box) -> bool:
    return gt_box is not None and box_area(gt_box) >= C.MIN_BOX_AREA


def aggregate(records: dict, num_episodes: int = C.NUM_EPISODES) -> dict:
    """Turn per-episode records into the rates reported in the paper (in %)."""
    if len(records) != num_episodes:
        raise ValueError(f"expected {num_episodes} episodes, got {len(records)}")
    dists = [r["min_vp_dist"] if r["min_vp_dist"] is not None else math.inf
             for r in records.values()]
    out = {"num_episodes": num_episodes}
    for t in C.ONS_THRESHOLDS:
        hits = sum(d <= t for d in dists)
        out[f"ONS@{t}"] = {"count": hits, "rate": 100.0 * hits / num_episodes}
    vis = sum(r["oracle_visible"] for r in records.values())
    out["OracleGS"] = {"count": vis, "rate": 100.0 * vis / num_episodes}
    if all("gs" in r for r in records.values()):
        gs = sum(r["gs"] for r in records.values())
        out["GS"] = {"count": gs, "rate": 100.0 * gs / num_episodes}
    return out
