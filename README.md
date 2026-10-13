# REVERIE-AIM Benchmark

Dataset and evaluation code for REVERIE-AIM, from
**From Region Arrival to Instance-Level Grounding in Vision-and-Language Navigation** (CoRL 2026,
[arXiv:2607.03792](https://arxiv.org/abs/2607.03792)).

REVERIE-AIM checks whether a VLN agent stops where it can actually see and ground the target object,
not just within 3 m of the goal. Run any agent on the 1344 val-unseen episodes, submit its final pose
per episode, and this repo computes the metrics.

```
data/reverie_aim_train.json.gz       3691 training episodes, 53 scenes
data/reverie_aim_val_unseen.json     1344 evaluation episodes, 10 scenes
data/target_phrases.json             target phrase per episode, used by GS
examples/etpnav_ft_endpoints.json    example submission (ETPNav-FT stop poses)
realm_bench/                         evaluation code
```

## Metrics

All rates are percentages over the 1344 val-unseen episodes.

| Metric | Definition |
|---|---|
| **ONS@t** | Geodesic distance from the stop position to the nearest viewpoint from which the target is visible (MP3D ObjectNav annotation) is ≤ *t* m. Reported for *t* = 0.1 and 0.5. |
| **OracleGS** | The target's ground-truth box covers ≥ 1% of the stop view. |
| **GS** | On OracleGS frames, OWLv2 (`google/owlv2-base-patch16-ensemble`) detects the target phrase and its top box reaches IoU ≥ 0.5 with the ground-truth box. |

The stop view is always rendered with the same camera: 224 × 224, 90° HFOV, 1.25 m above the agent base.

## Installation

1. Install habitat-sim 0.1.7 and habitat-lab 0.1.7 (Python 3.8, the VLN-CE setup).
2. `pip install -r requirements.txt`
3. Download the Matterport3D scenes for Habitat
   ([terms of use](https://niessner.github.io/Matterport/)). Each scene folder needs
   `<scene>.glb`, `<scene>.navmesh`, `<scene>_semantic.ply` and `<scene>.house` under
   `<scenes-dir>/mp3d/<scene>/`.

## Evaluation

Write your agent's final poses to a JSON file:

```json
{
  "4": {"position": [14.52, 0.17, -1.82], "ori": [-0.66, 0.0, 0.75, 0.0]},
  "5": {"position": [...], "ori": [...]}
}
```

- one entry per `episode_id` in `data/reverie_aim_val_unseen.json`
- `position`: agent base position on the navmesh (not the camera), Habitat world coordinates
- `ori`: rotation quaternion in `[w, x, y, z]` order (the dataset's `start_rotation` is `[x, y, z, w]`)

Then run:

```bash
python -m realm_bench.evaluate \
    --endpoints my_agent.json \
    --scenes-dir /path/to/scene_datasets \
    --out results/my_agent.json
```

`--no-gs` skips the detector, `--save-rgb DIR` saves the stop views.

To check your setup, evaluate the example file `examples/etpnav_ft_endpoints.json`
(stop poses of [ETPNav](https://github.com/MarSaKi/ETPNav) fine-tuned on REVERIE-CE).
It should print ONS@0.1 = 6.32, ONS@0.5 = 12.20, GS = 4.46, OracleGS = 11.31,
the ETPNav-FT "no move" row of Table 2 in the paper.

## License

Code: [MIT](LICENSE). Data: derived from REVERIE, REVERIE-CE, MP3D ObjectNav and Matterport3D, released
under the Matterport3D Terms of Use and CC BY-NC-SA 3.0 US, see [data/LICENSE](data/LICENSE).

## Citation

```bibtex
@inproceedings{shi2026region,
  title     = {From Region Arrival to Instance-Level Grounding in Vision-and-Language Navigation},
  author    = {Shi, Xiangyu and Yang, Ruoxi and Tao, Wei and Zhang, Jiwen and Qiao, Yanyuan and Wu, Qi},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
  eprint    = {2607.03792},
  archivePrefix = {arXiv}
}
```
