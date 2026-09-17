"""Experimental archive-domain intervention; no writer or estimator changes."""
import hashlib
import json
from pathlib import Path

ARMS = ("full_history", "recent_seven")


def validate_config(config):
    if config["arm"] not in ARMS:
        raise ValueError("Unknown retrieval-domain arm")
    count = config["history_frames"]
    indices = config["recent_indices"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 9:
        raise ValueError("Invalid causal history size")
    if (len(indices) != 7 or any(type(i) is not int for i in indices)
            or indices != sorted(set(indices))
            or not all(8 <= i < count for i in indices)):
        raise ValueError("Expected seven chronological historical decision indices")
    return config


def eligible_indices(config, frame_index, candidate_ceiling):
    validate_config(config)
    if frame_index < config["history_frames"]:
        raise ValueError("The original history has not been fully written")
    if candidate_ceiling != config["history_frames"] - 1:
        raise ValueError("The first-query historical boundary changed")
    high = min(frame_index - 1, candidate_ceiling)
    if config["arm"] == "full_history":
        return list(range(8, high + 1))
    return [i for i in config["recent_indices"] if i <= high]


def install(sparse_module, config_path, audit_path):
    """Wrap only the diagnostic server's class, retaining the native full arm."""
    from MemNavData.certified_relocalization_runtime import (
        CERTIFIED_MINIMUM_ANCHOR, CERTIFIED_CANDIDATE_TOP_K,
        CERTIFIED_CANDIDATE_MIN_GAP)
    assert CERTIFIED_MINIMUM_ANCHOR == 8
    cls = sparse_module.SparseReadout
    original_shortlist, original_read = cls.shortlist_from_scores, cls.read_sparse
    config_path, audit_path = Path(config_path), Path(audit_path)

    def read_config(goal_key):
        config = validate_config(json.loads(config_path.read_text()))
        if config["goal_key"] != goal_key:
            raise RuntimeError("The domain configuration belongs to another goal")
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        return config, fingerprint

    def emit(row):
        with audit_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    def shortlist(self, goal_key, candidate_ceiling, frame_index, visual_cosine):
        config, fingerprint = read_config(goal_key)
        indices = eligible_indices(config, int(frame_index), int(candidate_ceiling))
        key = (goal_key, candidate_ceiling)
        cached = self.shortlists.get(key)
        if cached is not None:
            if getattr(self, "_reviewer_domain_fingerprint", None) != fingerprint:
                raise RuntimeError("Retrieval domain changed during a cached goal session")
        else:
            if frame_index != config["history_frames"]:
                raise RuntimeError("First recall did not occur at the original query boundary")
            self._reviewer_domain_fingerprint = fingerprint
        scores = None
        if config["arm"] == "full_history":
            result = original_shortlist(self, goal_key, candidate_ceiling,
                                        frame_index, visual_cosine)
        elif cached is not None:
            result = [dict(item) for item in cached]
        else:
            scores = visual_cosine.detach().float().cpu().tolist()
            mask = [False] * (frame_index + 1)
            for index in indices:
                mask[index] = True
            result = sparse_module.temporal_nms_candidates(
                scores, mask, top_k=CERTIFIED_CANDIDATE_TOP_K,
                min_frame_gap=CERTIFIED_CANDIDATE_MIN_GAP)
            self.shortlists[key] = [dict(item) for item in result]
        if not all(item["anchor"] in indices for item in result):
            raise RuntimeError("A candidate escaped the permitted archive domain")
        if cached is None and scores is None:
            scores = visual_cosine.detach().float().cpu().tolist()
        emit(dict(event="shortlist", arm=config["arm"], goal_key=goal_key,
                  domain_fingerprint=fingerprint, frame_index=int(frame_index),
                  candidate_ceiling=int(candidate_ceiling), cached=cached is not None,
                  eligible_indices=indices if cached is None else None,
                  scores=scores, candidates=result))
        return result

    def read_sparse(self, goal_jpg_bytes, candidates, **kwargs):
        goal_key = hashlib.md5(goal_jpg_bytes).hexdigest()
        config, fingerprint = read_config(goal_key)
        start = self.goal_start_frames.get(goal_key)
        if start != config["history_frames"]:
            raise RuntimeError("Sparse read lost the original goal-session boundary")
        expected = self.shortlists.get((goal_key, start - 1))
        if expected is None or candidates != expected:
            raise RuntimeError("Sparse localization bypassed the restricted shortlist")
        if getattr(self, "_reviewer_domain_fingerprint", None) != fingerprint:
            raise RuntimeError("Sparse read changed retrieval domain")
        if (kwargs.get("graph_rescue", False) or kwargs.get("allow_learned_rescue", False)
                or kwargs.get("guidance_mode", "endpoint_bearing") != "endpoint_bearing"
                or kwargs.get("authority_policy", "strict_certificate") != "strict_certificate"):
            raise RuntimeError("Unexpected downstream method in the domain experiment")
        result = original_read(self, goal_jpg_bytes, candidates, **kwargs)
        allowed = {item["anchor"] for item in candidates}
        selected = result.get("selected_anchor")
        if selected is not None and selected not in allowed:
            raise RuntimeError("The selected anchor escaped the restricted shortlist")
        if any(item["anchor"] not in allowed for item in result.get("ranked_candidates", [])):
            raise RuntimeError("Geometric verification read an unpermitted candidate")
        emit(dict(event="sparse_read", arm=config["arm"], goal_key=goal_key,
                  domain_fingerprint=fingerprint, candidates=candidates,
                  selected_anchor=selected, accepted=result.get("accepted"),
                  cached=result.get("cached"), reason=result.get("reason")))
        return result

    cls.shortlist_from_scores, cls.read_sparse = shortlist, read_sparse
    return original_shortlist, original_read
