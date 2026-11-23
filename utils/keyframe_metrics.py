"""Metrics for point and interval keyframe predictions."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def compute_keyframe_metrics(preds_by_ep: Dict[Any, Dict[int, int]],
                             windows_by_ep: Dict[Any, List[Tuple[int, int]]]) -> Dict[str, Any]:
    n_episodes = n_windows = n_hit = n_exact = n_missed = n_excess_windows = 0
    total_yes = spurious = perfect_eps = 0
    nk_eps = nk_eps_correct = 0
    nk_frames = nk_frames_zero = 0
    # Positional interval coverage: slot 0 is the first annotated key interval in each episode,
    # slot 1 the second, and so on.  Denominators are recorded explicitly so the metric remains
    # correct even if malformed/legacy data contains fewer intervals in a subset of episodes.
    key_interval_hits: List[int] = []
    key_interval_exact: List[int] = []
    key_interval_episode_counts: List[int] = []

    for ep, preds in preds_by_ep.items():
        n_episodes += 1
        windows = windows_by_ep.get(ep, [])
        yes_frames = {t for t, p in preds.items() if p == 1}
        total_yes += len(yes_frames)

        covered = set()
        ep_perfect = True
        for interval_index, (s, e) in enumerate(windows):
            while len(key_interval_hits) <= interval_index:
                key_interval_hits.append(0)
                key_interval_exact.append(0)
                key_interval_episode_counts.append(0)
            key_interval_episode_counts[interval_index] += 1
            n_windows += 1
            cnt = sum(1 for t in yes_frames if s <= t <= e)
            covered.update(t for t in yes_frames if s <= t <= e)
            if cnt >= 1:
                n_hit += 1
                key_interval_hits[interval_index] += 1
            if cnt == 1:
                n_exact += 1
                key_interval_exact[interval_index] += 1
            elif cnt == 0:
                n_missed += 1
                ep_perfect = False
            else:
                n_excess_windows += 1
                ep_perfect = False
        ep_spurious = len(yes_frames - covered)
        spurious += ep_spurious
        if ep_spurious > 0:
            ep_perfect = False
        if windows and ep_perfect:
            perfect_eps += 1

        in_win = lambda tt: any(s <= tt <= e for (s, e) in windows)
        nk = [t for t in preds if not in_win(t)]
        if nk:
            nk_eps += 1
            nk_yes = sum(1 for t in nk if preds[t] == 1)
            if nk_yes == 0:
                nk_eps_correct += 1
            nk_frames += len(nk)
            nk_frames_zero += sum(1 for t in nk if preds[t] == 0)

    recall = n_hit / n_windows if n_windows else 0.0
    # precision numerator counts windows-hit, denominator counts Yes frames; with DISJOINT windows
    # n_hit <= total_yes so it lands in [0, 1]. Overlapping GT windows (evaluation drops such
    # episodes, but the VLM eval may keep them) can credit one Yes to >1 window and push it past 1 —
    # clamp defensively so precision/f1 stay well-defined.
    precision = min(n_hit / total_yes, 1.0) if total_yes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    segment_acc = n_exact / n_windows if n_windows else 0.0
    non_key_acc = nk_eps_correct / nk_eps if nk_eps else 0.0
    overall_score = (segment_acc + non_key_acc) / 2.0
    key_interval_hit_rates = [
        round(hits / count, 4) if count else 0.0
        for hits, count in zip(key_interval_hits, key_interval_episode_counts)
    ]
    key_interval_acc_rates = [
        round(exact / count, 4) if count else 0.0
        for exact, count in zip(key_interval_exact, key_interval_episode_counts)
    ]
    segment = {
        str(index + 1): {"hit": hit_rate, "acc": acc_rate}
        for index, (hit_rate, acc_rate) in enumerate(
            zip(key_interval_hit_rates, key_interval_acc_rates)
        )
    }
    segment["overall"] = {
        "hit": round(recall, 4),
        "acc": round(segment_acc, 4),
    }

    return {
        # --- the 6 headline metrics ---
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "segment": segment,
        # Compatibility alias for segment["overall"]["acc"].
        "segment_acc": round(segment_acc, 4),
        "non_key_acc": round(non_key_acc, 4),
        "overall_score": round(overall_score, 4),
        # --- diagnostics / counts (back-compat) ---
        "exact_rate": round(segment_acc, 4),          # alias: segment_acc IS the window exact-rate
        "segment_positive": round(n_hit / n_windows, 4) if n_windows else 0.0,
        "non_key_frame_acc": round(nk_frames_zero / nk_frames, 4) if nk_frames else 0.0,
        "n_episodes": n_episodes,
        "n_windows": n_windows,
        "n_segments": n_windows,
        "n_nonkey_episodes": nk_eps,
        "missed": n_missed,
        "excess_windows": n_excess_windows,
        "spurious_total": spurious,
        "spurious_per_ep": round(spurious / n_episodes, 3) if n_episodes else 0.0,
        "total_yes_pred": total_yes,
        "perfect_episodes": perfect_eps,
        # --- ordered GT-interval coverage (>=1 Yes inside the interval) ---
        # Array index 0/1/2/... means the first/second/third/... interval in an episode.
        "key_interval_hit_rates": key_interval_hit_rates,
        "key_interval_acc_rates": key_interval_acc_rates,
        "key_interval_hit_counts": key_interval_hits,
        "key_interval_acc_counts": key_interval_exact,
        "key_interval_episode_counts": key_interval_episode_counts,
    }
