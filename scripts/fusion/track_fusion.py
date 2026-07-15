"""
track_fusion.py — temporal post-processing (MOT) on top of center-fusion.

Variant **T1**: smooth + re-score only — NO gap interpolation (that is T2).

The per-frame center-fusion detector is temporally blind: it never looks
across keyframes. This stage tracks detections in the global frame, per
class, per scene, with a constant-velocity Kalman filter, runs a
forward-backward RTS smoother (the eval is offline, so future keyframes are
legal), then replaces each matched detection's velocity (aggressively) and
position (conservatively — position bias propagates, it doesn't average
out) with the smoothed estimate, and re-scores by track support (boost
confirmed tracks, decay flickers) without ever deleting a box.

T1 emits exactly one output box per input box, so it cannot add false
positives — a do-no-harm regression means the smoother is corrupting
position, so back off ``--alpha-pos``.

``--interpolate`` adds **T2** (offline) occlusion-gap interpolation.
``--online`` switches to **T3**, the causal counterpart of both (forward-
filtered state, no RTS backward pass; forward-coast instead of bidirectional
fill) — expected to be dominated by the offline variants, since it
quantifies the cost of discarding the future keyframes the offline eval
makes legal. With ``--online`` off, T1/T2 are byte-identical.

Radar is load-bearing here because the Kalman measurement model is
variable: a radar-associated detection's velocity is treated as an
observation (low noise), a camera-only detection's as merely inferred
(high noise), read from the center-fusion records cache (``--records``).
Shuffling that association collapses the smoothed-mAVE advantage.

center-fusion/late-fusion already emit translation/velocity in the global
frame, so the tracker lives in global BEV and only needs nuScenes for scene
grouping, real keyframe timestamps, and the final NuScenesEval.

Usage
-----
    # mini val (do-no-harm smoke test; baseline NDS 0.3318)
    python scripts/fusion/track_fusion.py --version v1.0-mini --dataroot data/nuscenes-mini \\
        --split mini_val \\
        --det-json results/center_fusion/mini_learned/results.json \\
        --records results/center_fusion/cache/records_mini_val.pkl \\
        --out results/track_fusion/mini

    # full val (do-no-harm gate: NDS >= 0.4115)
    python scripts/fusion/track_fusion.py --version v1.0-trainval --dataroot data/nuscenes \\
        --split val \\
        --det-json results/center_fusion/full_learned/results.json \\
        --records results/center_fusion/cache/records_full_val.pkl \\
        --out results/track_fusion/full
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

# Reuse — do NOT modify late_fusion.py; import its eval + class list only.
from scripts.fusion.late_fusion import evaluate, DETECTION_CLASSES

META = {'use_camera': True, 'use_lidar': False, 'use_radar': True,
        'use_map': False, 'use_external': False}


# ---------------------------------------------------------------------------
# Tracker hyper-parameters
# ---------------------------------------------------------------------------
@dataclass
class TrackParams:
    """Constant-velocity tracker + smoother + re-score knobs.

    Defaults are deliberately do-no-harm: position is smoothed only gently
    (Law 4) and re-scoring is mild. The Kalman noises are in SI units
    (metres, m/s) and chosen around the measured center-fusion error scale
    (mATE ~0.75 m, mAVE ~0.86 m/s).
    """

    # Constant-velocity process noise: std of the unmodelled BEV acceleration.
    sigma_a: float = 2.0           # m/s^2

    # Measurement noise (std).
    r_pos: float = 1.0             # m   — position is always observed
    r_vel_lo: float = 0.5         # m/s — radar-associated det: velocity OBSERVED
    r_vel_hi: float = 4.0         # m/s — camera-only det:      velocity INFERRED

    # Initial state covariance (std) for a freshly spawned track.
    p0_pos: float = 2.0           # m
    p0_vel: float = 4.0           # m/s

    # Association.
    gate_chi2: float = 9.21       # chi^2, 2 dof, ~99% — Mahalanobis position gate
    max_dist: float = 5.0         # m   — hard distance fallback gate
    max_age: int = 2              # keyframes a track may coast before it dies

    # Smoothing blend (0 = keep detection, 1 = use smoothed estimate).
    # Validated on full val: alpha_pos=0.1 *lowers* mATE (0.749->0.725);
    # alpha_pos=0.2 over-smooths and raises it (Law 4). alpha_vel=1.0 is safe
    # because nuScenes matches by 2D center distance only — velocity is a TP
    # attribute error, never a matching criterion.
    alpha_pos: float = 0.1        # conservative (Law 4)
    alpha_vel: float = 1.0        # aggressive

    # Track-support re-scoring (Law 2 — re-weight, never delete).
    rescore: bool = True
    rescore_min_hits: int = 3     # hits at which a track is "confirmed"
    rescore_floor: float = 0.3    # multiplier for a 1-frame flicker

    # --- T2: gap interpolation (additive, default OFF — T1 byte-stable) ---
    # For an occlusion gap inside a track (consecutive recorded keyframes >1
    # apart), bounded by real detections on BOTH sides, emit a NEW box at each
    # missing keyframe from the RTS-smoothed trajectory. Buys recall, spends
    # precision (Law 3) → gate hard on every knob below.
    interpolate: bool = False
    max_interp_gap: int = 2       # max missing keyframes to bridge (<= max_age)
    interp_min_hits: int = 3      # only fill gaps of CONFIRMED tracks
    fill_decay: float = 0.5       # filled score = decay * min(bounding scores)
    fill_score_floor: float = 0.0 # drop a filled box below this (hard floor)

    # --- T3: online causal SORT (orthogonal flag, default OFF → T1/T2 byte-stable) ---
    # The causal counterpart of T1 (RTS smoother) AND T2 (bidirectional fill):
    # at keyframe t use ONLY frames <= t. State estimate = forward-FILTERED
    # ``xfilt`` (no backward pass); re-scoring uses support up to step k only
    # (``support_factor_causal``); ``--online --interpolate`` adds FORWARD-COAST
    # boxes (``coast_track_forward``) instead of T2's bidirectional gap fill.
    # Reuses the SAME alpha_pos/alpha_vel/rescore/fill knobs. Expected to be
    # DOMINATED by T1/T2 — this measures the COST of causality (plan §0/§1.3),
    # it is a faithful contrast, not a win.
    online: bool = False


def F_matrix(dt: float) -> np.ndarray:
    """Constant-velocity state transition for state [px, py, vx, vy]."""
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt
    return F


def Q_matrix(dt: float, sigma_a: float) -> np.ndarray:
    """Discrete white-noise-acceleration process covariance (per axis)."""
    q = sigma_a ** 2
    dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
    Q = np.zeros((4, 4))
    # x-axis (px, vx)
    Q[0, 0] = dt4 / 4; Q[0, 2] = dt3 / 2
    Q[2, 0] = dt3 / 2; Q[2, 2] = dt2
    # y-axis (py, vy)
    Q[1, 1] = dt4 / 4; Q[1, 3] = dt3 / 2
    Q[3, 1] = dt3 / 2; Q[3, 3] = dt2
    return q * Q


# ---------------------------------------------------------------------------
# A single detection handed to the tracker (global BEV, plus carried payload)
# ---------------------------------------------------------------------------
@dataclass
class Det:
    px: float
    py: float
    vx: float
    vy: float
    radar_matched: bool
    token: str
    idx: int          # index of this detection within its sample's det list


# ---------------------------------------------------------------------------
# Per-track Kalman filter (records its steps for an RTS backward pass)
# ---------------------------------------------------------------------------
class KalmanBoxTracker:
    """Constant-velocity BEV track in the global frame.

    State x = [px, py, vx, vy]. Each detection is a full-state measurement
    z = [px, py, vx, vy] with diagonal noise R; the velocity noise is LOW for
    radar-associated detections and HIGH for camera-only ones (variable
    measurement model — this is where radar is load-bearing).

    Steps are recorded **only at frames with a real detection** (no synthetic
    coast steps), so T1 never emits a box where the camera saw nothing. The CV
    transition across a missed keyframe simply uses the larger elapsed dt.
    """

    _count = 0

    def __init__(self, det: Det, frame: int, t_us: int, p: TrackParams):
        self.p = p
        self.x = np.array([det.px, det.py, det.vx, det.vy], dtype=float)
        self.P = np.diag([p.p0_pos ** 2, p.p0_pos ** 2,
                          p.p0_vel ** 2, p.p0_vel ** 2]).astype(float)
        self.last_t = t_us
        self.first_frame = frame
        self.last_frame = frame
        self.hits = 1
        self.misses = 0
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1

        # Recorded filter steps for RTS (initial step: prior == posterior).
        # ``frame``/``t_us`` are carried for T2 gap detection (unused by T1).
        self.steps: list[dict] = [dict(
            xpred=self.x.copy(), Ppred=self.P.copy(),
            xfilt=self.x.copy(), Pfilt=self.P.copy(),
            F=np.eye(4), handle=(det.token, det.idx),
            frame=frame, t_us=t_us,
        )]
        self._pending: dict | None = None

    def predict(self, dt: float) -> np.ndarray:
        """Project state to time +dt WITHOUT committing (so an unmatched track
        keeps its last posterior as the anchor for the next, larger-dt jump)."""
        F = F_matrix(dt)
        xpred = F @ self.x
        Ppred = F @ self.P @ F.T + Q_matrix(dt, self.p.sigma_a)
        self._pending = dict(F=F, xpred=xpred, Ppred=Ppred)
        return xpred

    def update(self, det: Det, frame: int, t_us: int) -> None:
        """Kalman update with a full-state measurement; commit + record step."""
        pend = self._pending
        xpred, Ppred, F = pend['xpred'], pend['Ppred'], pend['F']

        r_vel = self.p.r_vel_lo if det.radar_matched else self.p.r_vel_hi
        R = np.diag([self.p.r_pos ** 2, self.p.r_pos ** 2, r_vel ** 2, r_vel ** 2])

        z = np.array([det.px, det.py, det.vx, det.vy], dtype=float)
        S = Ppred + R                       # H = I_4
        K = Ppred @ np.linalg.inv(S)
        self.x = xpred + K @ (z - xpred)
        self.P = (np.eye(4) - K) @ Ppred

        self.last_t = t_us
        self.last_frame = frame
        self.hits += 1
        self.misses = 0
        self.steps.append(dict(
            xpred=xpred.copy(), Ppred=Ppred.copy(),
            xfilt=self.x.copy(), Pfilt=self.P.copy(),
            F=F.copy(), handle=(det.token, det.idx),
            frame=frame, t_us=t_us,
        ))
        self._pending = None


def rts_smooth(steps: list[dict]) -> list[np.ndarray]:
    """Rauch–Tung–Striebel backward pass over one track's recorded steps.

    Returns the smoothed state at each step (same order as ``steps``).
    """
    n = len(steps)
    xs = [s['xfilt'].copy() for s in steps]
    Ps = [s['Pfilt'].copy() for s in steps]
    for k in range(n - 2, -1, -1):
        s1 = steps[k + 1]
        Pp1 = s1['Ppred']
        C = Ps[k] @ s1['F'].T @ np.linalg.inv(Pp1)
        xs[k] = xs[k] + C @ (xs[k + 1] - s1['xpred'])
        Ps[k] = Ps[k] + C @ (Ps[k + 1] - Pp1) @ C.T
    return xs


# ---------------------------------------------------------------------------
# Forward association over one (scene, class) detection stream
# ---------------------------------------------------------------------------
def track_stream(frames: list[list[Det]], timestamps: list[int],
                 p: TrackParams) -> list[KalmanBoxTracker]:
    """SORT-style forward pass with real-dt prediction and a Mahalanobis gate.

    ``frames[i]`` is the list of same-class detections at keyframe ``i``;
    ``timestamps[i]`` is that keyframe's timestamp (microseconds). Returns every
    track created in this scene/class (finished + still-active).
    """
    active: list[KalmanBoxTracker] = []
    finished: list[KalmanBoxTracker] = []

    for fi, dets in enumerate(frames):
        t_now = timestamps[fi]

        # Predict every active track to the current keyframe.
        preds = []
        for tr in active:
            dt = max((t_now - tr.last_t) / 1e6, 1e-3)
            preds.append(tr.predict(dt))

        matched: list[tuple[int, int]] = []
        if active and dets:
            big = 1e6
            cost = np.full((len(active), len(dets)), big)
            for i, xpred in enumerate(preds):
                # Gate on the predicted (inflated) position innovation covariance.
                Spred = active[i]._pending['Ppred'][:2, :2] + np.eye(2) * p.r_pos ** 2
                Sinv = np.linalg.inv(Spred)
                for j, d in enumerate(dets):
                    dz = np.array([d.px - xpred[0], d.py - xpred[1]])
                    m2 = float(dz @ Sinv @ dz)
                    if m2 <= p.gate_chi2 and np.hypot(*dz) <= p.max_dist:
                        cost[i, j] = np.sqrt(m2)
            row, col = linear_sum_assignment(cost)
            matched = [(i, j) for i, j in zip(row, col) if cost[i, j] < big]

        matched_tr = {i for i, _ in matched}
        matched_dt = {j for _, j in matched}

        for i, j in matched:
            active[i].update(dets[j], fi, t_now)

        # Unmatched active tracks coast; retire once they exceed max_age.
        survivors = []
        for i, tr in enumerate(active):
            if i in matched_tr:
                survivors.append(tr)
                continue
            tr.misses += 1
            tr._pending = None
            if tr.misses > p.max_age:
                finished.append(tr)
            else:
                survivors.append(tr)
        active = survivors

        # Spawn a new track for every unmatched detection.
        for j, d in enumerate(dets):
            if j not in matched_dt:
                active.append(KalmanBoxTracker(d, fi, t_now, p))

    return finished + active


# ---------------------------------------------------------------------------
# Re-scoring (Law 2: re-weight by track support, never delete)
# ---------------------------------------------------------------------------
def support_factor(tr: KalmanBoxTracker, p: TrackParams) -> float:
    """Multiplicative score factor in [floor, 1]. Confirmed, dense tracks keep
    their score (factor 1); short / sparse flickers are decayed toward the
    floor. Monotone re-ranking → boosts confirmed TPs above flickers without
    adding or deleting any box."""
    span = tr.last_frame - tr.first_frame + 1
    conf = min(1.0, tr.hits / max(p.rescore_min_hits, 1))
    density = tr.hits / max(span, 1)
    return p.rescore_floor + (1.0 - p.rescore_floor) * conf * density


def support_factor_causal(tr: KalmanBoxTracker, k: int, p: TrackParams) -> float:
    """Causal counterpart of ``support_factor`` for online (T3) re-scoring.

    The box emitted at recorded step ``k`` may use only support seen up to and
    including step ``k`` — peeking at the track's eventual length is future
    information, illegal online. Steps are recorded only on hits, so
    ``hits_so_far = k + 1``; ``span`` runs from the track's first frame to this
    step's frame. At the LAST step this equals the whole-track
    ``support_factor``; earlier steps are DEMOTED (fewer hits / lower density so
    far) — the honest online penalty on early-track true positives.
    """
    hits_so_far = k + 1
    span = tr.steps[k]['frame'] - tr.first_frame + 1
    conf = min(1.0, hits_so_far / max(p.rescore_min_hits, 1))
    density = hits_so_far / max(span, 1)
    return p.rescore_floor + (1.0 - p.rescore_floor) * conf * density


# ---------------------------------------------------------------------------
# T2: gap interpolation (Law 3 — buys recall, spends precision; gate hard)
# ---------------------------------------------------------------------------
def fill_track_gaps(tr: KalmanBoxTracker, xs: list[np.ndarray],
                    out_results: dict, tokens: list[str],
                    timestamps: list[int], p: TrackParams) -> int:
    """Emit NEW boxes for occlusion gaps of one (RTS-smoothed) track.

    A gap is two consecutive recorded steps whose scene-frame indices are
    >1 apart; by construction it is bounded by a *real* detection on both
    sides (steps record only matched keyframes — §8.1). For each missing
    keyframe we add a box to ``out_results[token]`` with:

      * position/velocity — linearly interpolated between the two bounding
        smoothed states at the missing frame's real timestamp (constant
        velocity ⇒ position interpolation is exact; uses BOTH sides, the
        offline RTS philosophy);
      * size/rotation/z/attribute — carried from the nearer-in-time bounding
        detection (class is fixed — tracking is per-class);
      * score — DECAYED below the bounding boxes (Law 2/3): an interpolated
        box that misses the tight 0.5–1.0 m threshold becomes an FP.

    Hard gates (Law 3): track must be CONFIRMED (``interp_min_hits``), the gap
    no longer than ``max_interp_gap`` (≤ ``max_age``), and the decayed score
    must clear ``fill_score_floor``. Returns the number of boxes added.
    """
    if not p.interpolate or tr.hits < p.interp_min_hits:
        return 0

    added = 0
    for k in range(len(tr.steps) - 1):
        s0, s1 = tr.steps[k], tr.steps[k + 1]
        f0, f1 = s0['frame'], s1['frame']
        gap = f1 - f0 - 1
        if gap < 1 or gap > p.max_interp_gap:
            continue

        tok0, idx0 = s0['handle']
        tok1, idx1 = s1['handle']
        box0, box1 = out_results[tok0][idx0], out_results[tok1][idx1]
        # Bounding scores are post-T1 (re-scored) — keep the gap box ranked
        # consistently below the track it belongs to.
        base = min(box0['detection_score'], box1['detection_score'])
        fill_score = p.fill_decay * base
        if fill_score < p.fill_score_floor:
            continue

        x0, x1 = xs[k], xs[k + 1]
        t0, t1 = float(s0['t_us']), float(s1['t_us'])
        z0, z1 = box0['translation'][2], box1['translation'][2]
        denom = (t1 - t0) if (t1 - t0) != 0 else 1.0

        for f in range(f0 + 1, f1):
            frac = (timestamps[f] - t0) / denom
            xi = (1.0 - frac) * x0 + frac * x1
            src = box0 if frac < 0.5 else box1     # nearer-in-time carry
            new = {
                'sample_token': tokens[f],
                'translation': [float(xi[0]), float(xi[1]),
                                float((1.0 - frac) * z0 + frac * z1)],
                'size': list(src['size']),
                'rotation': list(src['rotation']),
                'velocity': [float(xi[2]), float(xi[3])],
                'detection_name': src['detection_name'],
                'detection_score': float(fill_score),
                'attribute_name': src['attribute_name'],
            }
            out_results[tokens[f]].append(new)
            added += 1

    return added


# ---------------------------------------------------------------------------
# T3: online forward coasting (causal analog of T2 — pure forward extrapolation)
# ---------------------------------------------------------------------------
def coast_track_forward(tr: KalmanBoxTracker, out_results: dict,
                        tokens: list[str], timestamps: list[int],
                        p: TrackParams) -> int:
    """Emit FORWARD-COAST boxes for one confirmed track — the online (causal)
    analog of T2's bidirectional gap fill.

    For each recorded step ``k`` at frame ``f_k`` we forward-propagate the
    **forward-filtered** state ``xfilt[k]`` (causal — no future information) with
    the constant-velocity model to every frame the track stays alive:
    ``track_stream`` retires a track once ``misses > max_age``, so it lives at
    frames ``f_k+1 … f_k+max_age``. We stop early at the next real detection.
    This naturally produces:

      * **gap-internal coasting** — between two real detections (the same frames
        T2 fills) but forward-EXTRAPOLATED from the left only, not
        bidirectionally interpolated; and
      * **tail coasting** — after the track's last detection, pure extrapolation
        with no reappearance: the boxes offline never emits, and where online
        pays its false-positive tax.

    Every carried attribute (size / rotation / z / attribute_name / class) comes
    from the LAST seen detection (step ``k``) — never the future. The score is
    geometrically decayed ``base · fill_decay**(f − f_k)`` from step ``k``'s
    causal-rescored score, dropped below ``fill_score_floor``. Gated on the track
    being CONFIRMED (``interp_min_hits``). Returns the number of boxes added.
    """
    if not p.interpolate or tr.hits < p.interp_min_hits:
        return 0

    n_frames = len(tokens)
    added = 0
    for k in range(len(tr.steps)):
        s = tr.steps[k]
        f_k = s['frame']
        t_k = float(s['t_us'])
        x_k = s['xfilt']
        tok_k, idx_k = s['handle']
        src = out_results[tok_k][idx_k]          # last seen det (causal-rescored)
        base = src['detection_score']
        z_k = src['translation'][2]

        # Coast until the next real detection (gap) or scene end (tail), capped
        # by the track's survival horizon f_k + max_age (alive at f_k+1…f_k+max_age).
        next_frame = (tr.steps[k + 1]['frame']
                      if k + 1 < len(tr.steps) else n_frames)
        coast_end = min(next_frame, f_k + p.max_age + 1)

        for f in range(f_k + 1, coast_end):
            fill_score = base * (p.fill_decay ** (f - f_k))
            if fill_score < p.fill_score_floor:
                continue
            dt = (timestamps[f] - t_k) / 1e6
            xf = F_matrix(dt) @ x_k              # forward extrapolation only
            out_results[tokens[f]].append({
                'sample_token': tokens[f],
                'translation': [float(xf[0]), float(xf[1]), float(z_k)],
                'size': list(src['size']),
                'rotation': list(src['rotation']),
                'velocity': [float(xf[2]), float(xf[3])],
                'detection_name': src['detection_name'],
                'detection_score': float(fill_score),
                'attribute_name': src['attribute_name'],
            })
            added += 1

    return added


# ---------------------------------------------------------------------------
# Scene grouping
# ---------------------------------------------------------------------------
def scene_frames(nusc, split: str) -> list[tuple[str, list[str], list[int]]]:
    """Per-scene ordered (sample tokens, timestamps) for an eval split."""
    from nuscenes.utils.splits import create_splits_scenes
    split_scenes = set(create_splits_scenes()[split])
    scenes = []
    for scene in nusc.scene:
        if scene['name'] not in split_scenes:
            continue
        toks, ts = [], []
        tok = scene['first_sample_token']
        while tok:
            s = nusc.get('sample', tok)
            toks.append(tok)
            ts.append(int(s['timestamp']))
            tok = s['next']
        scenes.append((scene['name'], toks, ts))
    return scenes


def load_radar_matched(records_path: str | None,
                       det_results: dict) -> dict[str, list[bool]]:
    """Map sample_token -> per-detection radar-association flags.

    Sourced from the center-fusion records cache, which is built in the same
    detection order as the submission JSON (verified 1:1). If absent, every
    detection is treated as radar-matched (uniform low velocity noise) and the
    variable measurement model degenerates — a warning is printed.
    """
    if not records_path or not os.path.exists(records_path):
        print(f'[WARN] no records cache ({records_path}); treating all '
              'detections as radar-matched (variable measurement model off).')
        return {tok: [True] * len(dets) for tok, dets in det_results.items()}

    with open(records_path, 'rb') as f:
        records = pickle.load(f)

    rm: dict[str, list[bool]] = {}
    mismatched = 0
    for rec in records:
        tok = rec['sample_token']
        flags = [bool(d.get('has_radar')) for d in rec['dets']]
        rm[tok] = flags
        if tok in det_results and len(det_results[tok]) != len(flags):
            mismatched += 1
    if mismatched:
        raise ValueError(f'records/det-json detection-count mismatch on '
                         f'{mismatched} samples — wrong cache for this JSON?')
    return rm


# ---------------------------------------------------------------------------
# Top-level T1 post-processing
# ---------------------------------------------------------------------------
def run_track_postproc(nusc, det_json: str, split: str, out_dir: str,
                       p: TrackParams, records_path: str | None = None,
                       verbose: bool = True) -> str:
    """Track + RTS-smooth + re-score the center-fusion detections (T1)."""
    os.makedirs(out_dir, exist_ok=True)

    with open(det_json) as f:
        det_results = json.load(f)['results']

    radar_matched = load_radar_matched(records_path, det_results)

    # Deep copy: every input box is emitted; we only overwrite a few fields.
    out_results = {tok: [dict(d) for d in dets]
                   for tok, dets in det_results.items()}

    scenes = scene_frames(nusc, split)
    KalmanBoxTracker._count = 0

    n_dets = n_tracks = n_smoothed = n_interp = 0
    for sc_i, (name, tokens, timestamps) in enumerate(scenes):
        for cls in DETECTION_CLASSES:
            # Build this class's per-frame detection stream for the scene.
            frames: list[list[Det]] = []
            for tok in tokens:
                dets_here = []
                for idx, d in enumerate(det_results.get(tok, [])):
                    if d['detection_name'] != cls:
                        continue
                    tx, ty = d['translation'][0], d['translation'][1]
                    vx, vy = d['velocity'][0], d['velocity'][1]
                    flags = radar_matched.get(tok)
                    rm = bool(flags[idx]) if flags is not None else True
                    dets_here.append(Det(tx, ty, vx, vy, rm, tok, idx))
                frames.append(dets_here)
                n_dets += len(dets_here)

            if not any(frames):
                continue

            tracks = track_stream(frames, timestamps, p)
            n_tracks += len(tracks)

            for tr in tracks:
                # State estimate + re-score factor: RTS backward smoother with
                # whole-track support (T1/T2, offline) OR the forward-FILTERED
                # state with support-up-to-step-k (T3, causal). Same blend math.
                if p.online:
                    xs = [s['xfilt'] for s in tr.steps]
                    factors = ([support_factor_causal(tr, k, p)
                                for k in range(len(tr.steps))]
                               if p.rescore else [1.0] * len(tr.steps))
                else:
                    xs = rts_smooth(tr.steps)
                    factor = support_factor(tr, p) if p.rescore else 1.0

                for k, (step, x_s) in enumerate(zip(tr.steps, xs)):
                    tok, idx = step['handle']
                    out = out_results[tok][idx]
                    # Position: conservative blend, keep z (Law 4).
                    out['translation'][0] = ((1 - p.alpha_pos) * out['translation'][0]
                                             + p.alpha_pos * float(x_s[0]))
                    out['translation'][1] = ((1 - p.alpha_pos) * out['translation'][1]
                                             + p.alpha_pos * float(x_s[1]))
                    # Velocity: aggressive blend.
                    out['velocity'][0] = ((1 - p.alpha_vel) * out['velocity'][0]
                                          + p.alpha_vel * float(x_s[2]))
                    out['velocity'][1] = ((1 - p.alpha_vel) * out['velocity'][1]
                                          + p.alpha_vel * float(x_s[3]))
                    fac = factors[k] if p.online else factor
                    out['detection_score'] = float(out['detection_score'] * fac)
                    n_smoothed += 1

                # Added boxes (additive, flagged via --interpolate): forward
                # coasting (T3, causal) OR bidirectional gap fill (T2, offline).
                # Runs AFTER re-scoring so emitted scores rank below the track.
                if p.online:
                    n_interp += coast_track_forward(tr, out_results,
                                                    tokens, timestamps, p)
                else:
                    n_interp += fill_track_gaps(tr, xs, out_results,
                                                tokens, timestamps, p)

        if verbose and (sc_i + 1) % 25 == 0:
            print(f'  {sc_i + 1}/{len(scenes)} scenes processed...')

    submission = {'meta': META, 'results': out_results}
    results_path = os.path.join(out_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(submission, f)

    if verbose:
        if p.online:
            variant = ('T3-SORT (online forward-coast)' if p.interpolate
                       else 'T3-smooth (online forward-filter)')
        else:
            variant = 'T2 (T1 + gap interp)' if p.interpolate else 'T1'
        print(f'\n{variant} post-processing statistics:')
        print(f'  Scenes:                 {len(scenes)}')
        print(f'  Detections (per-class): {n_dets}')
        print(f'  Tracks formed:          {n_tracks}')
        print(f'  Boxes smoothed/rescored:{n_smoothed}')
        print(f'  alpha_pos={p.alpha_pos}  alpha_vel={p.alpha_vel}  '
              f'rescore={p.rescore} (floor={p.rescore_floor}, '
              f'min_hits={p.rescore_min_hits})  online={p.online}')
        if p.interpolate and p.online:
            print(f'  T3 boxes coasted:       {n_interp}  '
                  f'(horizon=max_age={p.max_age}, min_hits={p.interp_min_hits}, '
                  f'decay={p.fill_decay}^Δf, floor={p.fill_score_floor})')
        elif p.interpolate:
            print(f'  T2 boxes filled:        {n_interp}  '
                  f'(max_gap={p.max_interp_gap}, min_hits={p.interp_min_hits}, '
                  f'decay={p.fill_decay}, floor={p.fill_score_floor})')
        print(f'Wrote results → {results_path}')

    return results_path


# ---------------------------------------------------------------------------
# Evaluation + recall/TP extraction
# ---------------------------------------------------------------------------
def eval_with_recall(nusc, results_path: str, split: str, out_dir: str,
                     verbose: bool = True) -> dict:
    """Run NuScenesEval and additionally extract recall at the operating point.

    Superset of ``late_fusion.evaluate``: same mAP/NDS/mATE/... summary +
    metrics.json, plus the **max-recall** per matching threshold pulled from
    ``metric_data_list``. ``DetectionMetricData.max_recall`` is the recall at
    the lowest-confidence operating point = (#TP / #GT) over the full ranking.
    Number of GT is identical across variants, so a strict increase in mean
    max-recall ⇔ a strict increase in true positives — the T2 recall gate
    (plan §4.2). Lower thresholds (0.5/1.0 m) are the tight precision-sensitive
    ones Law 3 worries about; 2.0 m is the canonical operating point.
    """
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    cfg = config_factory('detection_cvpr_2019')
    evaluator = NuScenesEval(nusc, config=cfg, result_path=results_path,
                             eval_set=split, output_dir=out_dir, verbose=verbose)
    metrics, mdl = evaluator.evaluate()

    summary = {
        'mAP': metrics.mean_ap,
        'NDS': metrics.nd_score,
        'mATE': metrics.tp_errors.get('trans_err', float('nan')),
        'mASE': metrics.tp_errors.get('scale_err', float('nan')),
        'mAOE': metrics.tp_errors.get('orient_err', float('nan')),
        'mAVE': metrics.tp_errors.get('vel_err', float('nan')),
        'mAAE': metrics.tp_errors.get('attr_err', float('nan')),
    }
    for cls in DETECTION_CLASSES:
        summary[f'{cls}_AP'] = metrics.mean_dist_aps.get(cls, float('nan'))

    # Recall at operating point (TP proxy), per matching distance threshold.
    dist_ths = sorted({dth for (_, dth) in mdl.md.keys()})
    all_recalls = []
    for dth in dist_ths:
        recs = [md.max_recall for md, _ in mdl.get_dist_data(dth)]
        summary[f'mean_max_recall@{dth}'] = float(np.mean(recs))
        all_recalls.extend(recs)
    summary['mean_max_recall'] = float(np.mean(all_recalls))

    metrics_path = os.path.join(out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f'\n{"="*60}')
        print('  Track-fusion evaluation')
        print(f'{"="*60}')
        print(f'  mAP : {summary["mAP"]:.4f}')
        print(f'  NDS : {summary["NDS"]:.4f}')
        print(f'  mATE: {summary["mATE"]:.4f}   mAVE: {summary["mAVE"]:.4f}')
        for dth in dist_ths:
            print(f'  mean max-recall @ {dth:>3} m: '
                  f'{summary[f"mean_max_recall@{dth}"]:.4f}')
        print(f'  mean max-recall (all th):  {summary["mean_max_recall"]:.4f}')
        print(f'{"="*60}')
        print(f'Metrics saved → {metrics_path}')

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='T1 temporal post-processing (Kalman + RTS + re-score) '
                    'on center-fusion detections')
    p.add_argument('--version', default='v1.0-mini')
    p.add_argument('--dataroot', default='data/nuscenes-mini')
    p.add_argument('--split', default='mini_val')
    p.add_argument('--det-json', required=True,
                   help='center-fusion detection JSON (nuScenes submission format)')
    p.add_argument('--records', default=None,
                   help='center-fusion records cache (.pkl) for radar_matched flags')
    p.add_argument('--out', default='results/track_fusion/mini')
    p.add_argument('--no-eval', action='store_true')

    # Tracker knobs (defaults = TrackParams).
    d = TrackParams()
    p.add_argument('--sigma-a', type=float, default=d.sigma_a)
    p.add_argument('--r-pos', type=float, default=d.r_pos)
    p.add_argument('--r-vel-lo', type=float, default=d.r_vel_lo)
    p.add_argument('--r-vel-hi', type=float, default=d.r_vel_hi)
    p.add_argument('--gate-chi2', type=float, default=d.gate_chi2)
    p.add_argument('--max-dist', type=float, default=d.max_dist)
    p.add_argument('--max-age', type=int, default=d.max_age)
    p.add_argument('--alpha-pos', type=float, default=d.alpha_pos)
    p.add_argument('--alpha-vel', type=float, default=d.alpha_vel)
    p.add_argument('--no-rescore', action='store_true')
    p.add_argument('--rescore-floor', type=float, default=d.rescore_floor)
    p.add_argument('--rescore-min-hits', type=int, default=d.rescore_min_hits)

    # T2 gap interpolation (additive; default OFF ⇒ byte-stable T1).
    p.add_argument('--interpolate', action='store_true',
                   help='enable T2 occlusion-gap interpolation (default off)')
    p.add_argument('--max-interp-gap', type=int, default=d.max_interp_gap,
                   help='max missing keyframes to bridge (<= max_age)')
    p.add_argument('--interp-min-hits', type=int, default=d.interp_min_hits,
                   help='only fill gaps of tracks with >= this many hits')
    p.add_argument('--fill-decay', type=float, default=d.fill_decay,
                   help='filled-box score = decay * min(bounding scores)')
    p.add_argument('--fill-score-floor', type=float, default=d.fill_score_floor,
                   help='drop a filled box whose decayed score is below this')

    # T3 online causal mode (orthogonal; default OFF ⇒ byte-stable T1/T2).
    p.add_argument('--online', action='store_true',
                   help='T3: online CAUSAL mode — forward-filtered state + '
                        'causal rescore (vs T1 RTS); with --interpolate, '
                        'forward-coast boxes (vs T2 bidirectional fill). A '
                        'faithful contrast, expected DOMINATED by offline T1/T2.')
    return p.parse_args()


if __name__ == '__main__':
    os.chdir(_ROOT)
    from nuscenes.nuscenes import NuScenes

    args = parse_args()
    params = TrackParams(
        sigma_a=args.sigma_a, r_pos=args.r_pos,
        r_vel_lo=args.r_vel_lo, r_vel_hi=args.r_vel_hi,
        gate_chi2=args.gate_chi2, max_dist=args.max_dist, max_age=args.max_age,
        alpha_pos=args.alpha_pos, alpha_vel=args.alpha_vel,
        rescore=not args.no_rescore,
        rescore_floor=args.rescore_floor, rescore_min_hits=args.rescore_min_hits,
        interpolate=args.interpolate, max_interp_gap=args.max_interp_gap,
        interp_min_hits=args.interp_min_hits, fill_decay=args.fill_decay,
        fill_score_floor=args.fill_score_floor,
        online=args.online,
    )

    print(f'Loading nuScenes {args.version} from {args.dataroot}...')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    results_path = run_track_postproc(
        nusc, det_json=args.det_json, split=args.split, out_dir=args.out,
        p=params, records_path=args.records, verbose=True)

    if not args.no_eval:
        eval_with_recall(nusc, results_path=results_path, split=args.split,
                         out_dir=args.out, verbose=True)
