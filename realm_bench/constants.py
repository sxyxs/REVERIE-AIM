"""Fixed protocol constants of the REVERIE-AIM benchmark.

Every number here is part of the benchmark definition. Changing any of them
makes results incomparable with the paper (Table 2).
"""

# ── Camera / simulator (habitat-sim 0.1.7, habitat-lab 0.1.7 defaults) ──
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
HFOV = 90                       # degrees
SENSOR_POSITION = [0.0, 1.25, 0.0]   # camera height 1.25 m above the agent base
FORWARD_STEP_SIZE = 0.25        # metres, only relevant for agents that step in the sim
TURN_ANGLE = 30                 # degrees, only relevant for agents that step in the sim

# ── Object Navigation Success (ONS) ──
# ONS@t: geodesic distance from the stop position to the nearest ObjectNav
# viewpoint of the target instance is <= t metres. Viewpoints lie on a
# ~0.09 m grid, so ONS@0.1 means "standing inside the viewpoint region".
ONS_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
REPORTED_ONS_THRESHOLDS = [0.1, 0.5]

# ── OracleGS ──
# The target instance is visible in the single stop view and its GT box
# covers at least 1% of the image.
MIN_BOX_AREA_RATIO = 0.01
MIN_BOX_AREA = IMAGE_WIDTH * IMAGE_HEIGHT * MIN_BOX_AREA_RATIO   # 501.76 px^2

# ── GS (grounding success) ──
# BERT-Large QA extracts a target phrase from the instruction (pre-computed in
# data/target_phrases.json); OWLv2 detects it in the stop view; the top-1 box
# must reach IoU >= 0.5 with the GT box. Only frames that pass the OracleGS
# area filter can count, so GS <= OracleGS by construction.
OWL_MODEL = "google/owlv2-base-patch16-ensemble"
OWL_SCORE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.5

# All rates are reported over the full split.
NUM_EPISODES = 1344
