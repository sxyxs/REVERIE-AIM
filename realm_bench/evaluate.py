"""Evaluate stop poses on REVERIE-AIM val-unseen.

Input: a JSON file mapping every episode id to the agent's final pose
    {"<episode_id>": {"position": [x, y, z], "ori": [w, x, y, z]}, ...}

Output: ONS@{0.1..0.5}, OracleGS and GS (all in % over 1344 episodes),
plus per-episode records.

Example:
    python -m realm_bench.evaluate \
        --endpoints my_agent.json \
        --scenes-dir /path/to/scene_datasets \
        --out results/my_agent.json
"""

import argparse
import json
import math
import os
from collections import defaultdict

from PIL import Image
from tqdm import tqdm

from . import constants as C
from .metrics import aggregate, gt_bbox_from_semantic, min_viewpoint_distance, oracle_visible
from .sim import BenchSim

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = os.path.join(HERE, "data", "reverie_aim_val_unseen.json")
DEFAULT_PHRASES = os.path.join(HERE, "data", "target_phrases.json")


def load_endpoints(path, episode_ids):
    with open(path) as f:
        endpoints = json.load(f)
    missing = sorted(set(episode_ids) - set(endpoints), key=int)
    if missing:
        raise ValueError(f"{len(missing)} episodes have no endpoint, e.g. {missing[:10]}")
    swapped = 0
    for eid in episode_ids:
        e = endpoints[eid]
        if len(e["position"]) != 3 or len(e["ori"]) != 4:
            raise ValueError(f"episode {eid}: position must be [x,y,z] and ori [w,x,y,z]")
        q = [float(v) for v in e["ori"]]
        norm = math.sqrt(sum(v * v for v in q))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(f"episode {eid}: ori {q} is not a unit quaternion (norm {norm:.4f})")
        # A heading-only rotation is [w, 0, y, 0] in [w,x,y,z] order but
        # [0, y, 0, w] in [x,y,z,w] order.
        if abs(q[0]) < 1e-6 and abs(q[2]) < 1e-6:
            swapped += 1
    if swapped > len(episode_ids) / 2:
        raise ValueError(
            f"{swapped}/{len(episode_ids)} quaternions have zero w and y components, which "
            "means they are most likely in [x,y,z,w] order (the dataset's start_rotation "
            "convention). Endpoints must use [w,x,y,z].")
    return endpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoints", required=True, help="JSON with final poses per episode")
    parser.add_argument("--scenes-dir", required=True,
                        help="Habitat scene root; episode scene_id 'mp3d/X/X.glb' is joined to it")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--phrases", default=DEFAULT_PHRASES,
                        help="Pre-computed BERT target phrases used by GS")
    parser.add_argument("--out", required=True, help="Output JSON (summary + per-episode)")
    parser.add_argument("--no-gs", action="store_true", help="Skip OWLv2 (GS not reported)")
    parser.add_argument("--save-rgb", default=None, help="Optional dir to save stop views")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.dataset) as f:
        episodes = json.load(f)["episodes"]
    episode_ids = [str(e["episode_id"]) for e in episodes]
    endpoints = load_endpoints(args.endpoints, episode_ids)

    grounder, phrases = None, None
    if not args.no_gs:
        from .grounding import OWLv2Grounder, iou
        with open(args.phrases) as f:
            phrases = json.load(f)
        grounder = OWLv2Grounder(device=args.device)
    if args.save_rgb:
        os.makedirs(args.save_rgb, exist_ok=True)

    by_scene = defaultdict(list)
    for e in episodes:
        by_scene[e["scene_id"]].append(e)

    records = {}
    with tqdm(total=len(episodes), desc="episodes") as bar:
        for scene_id, scene_eps in by_scene.items():
            sim = BenchSim(os.path.join(args.scenes_dir, scene_id))
            try:
                # Without the semantic mesh every semantic frame is 0, which
                # would silently give OracleGS = GS = 0.
                missing = {e["objId"] for e in scene_eps} - sim.semantic_object_ids()
                if missing:
                    raise RuntimeError(
                        f"{scene_id}: target ids {sorted(missing)} are not in the semantic "
                        "annotation. Check that <scene>_semantic.ply and <scene>.house are "
                        "next to the .glb file.")
                for e in scene_eps:
                    eid = str(e["episode_id"])
                    pose = endpoints[eid]
                    sim.set_pose(pose["position"], pose["ori"])
                    dist = min_viewpoint_distance(sim, pose["position"], e["view_points"])
                    gt_box = gt_bbox_from_semantic(sim.render_semantic(), e["objId"])
                    rec = {
                        "scene": scene_id.split("/")[1],
                        "min_vp_dist": None if math.isinf(dist) else dist,
                        "gt_bbox": gt_box,
                        "oracle_visible": oracle_visible(gt_box),
                    }
                    if grounder is not None or args.save_rgb:
                        rgb = sim.render_rgb()
                        if args.save_rgb:
                            Image.fromarray(rgb).save(os.path.join(args.save_rgb, f"{eid}.png"))
                    if grounder is not None:
                        if phrases[eid]["instruction"] != e["instruction"]["instruction_text"]:
                            raise ValueError(f"episode {eid}: phrase file does not match dataset")
                        phrase = phrases[eid]["target"]
                        hit, score = False, None
                        if rec["oracle_visible"]:
                            score = iou(grounder.best_box(rgb, phrase), gt_box)
                            hit = score >= C.IOU_THRESHOLD
                        rec.update({"target_phrase": phrase, "iou": score, "gs": hit})
                    records[eid] = rec
                    bar.update(1)
            finally:
                sim.close()

    summary = aggregate(records, num_episodes=len(episodes))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"endpoints": os.path.abspath(args.endpoints), "summary": summary,
                   "episodes": records}, f, indent=1)

    line = [f"ONS@{t}={summary[f'ONS@{t}']['rate']:.2f}" for t in C.REPORTED_ONS_THRESHOLDS]
    if "GS" in summary:
        line.append(f"GS={summary['GS']['rate']:.2f}")
    line.append(f"OracleGS={summary['OracleGS']['rate']:.2f}")
    print("  ".join(line), f"(n={summary['num_episodes']})")


if __name__ == "__main__":
    main()
