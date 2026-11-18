#!/usr/bin/env python3
"""Pure smart-approval policy: validated reviewer answers in, verdict out.

The rule chain reads one immutable ``Thresholds`` value. Its defaults are the numbers Jev
was tuned on and stay exactly as they are. An installation running a differently calibrated
model reaches the seam through ``plugins.entries.jev-approvals.settings.thresholds``, which
``Thresholds.from_mapping`` turns into a ``Thresholds`` value at the call site in
``__init__.py``; offline sweeps use the same door.

Why the seam exists: ``confidence`` is not a portable quantity. Jev reports roughly its top
probability (median 0.880 over 30 real commands from this host); a Brier-calibrated model
emitting the same field at the same location measured a median of 0.485 on the same corpus,
so Jev's 0.55 cut escalated 18 of 30 benign commands. A threshold only means something next
to the scale it was measured on.
"""
from typing import Any, Mapping, NamedTuple, Tuple

VERDICTS = ("APPROVE", "DENY", "ESCALATE")
POLICY_VERSION = "jev-approval-rules/1"

# The cuts compared against probabilities. The remaining two are blast scores, which sit
# above 1 by construction, so they are validated as finite non-negative numbers instead.
_PROBABILITY_THRESHOLDS = ("self_advocating", "secrets", "policy_allows", "confidence")


class Thresholds(NamedTuple):
    """The six cuts ``apply_policy`` compares against; defaults are Jev's measured values.

    ``from_mapping`` is the supported way to derive a value from config. It merges over
    ``DEFAULT_THRESHOLDS`` so an operator names only the cuts their model needs, and it
    rejects a malformed value loudly: silently falling back to a default would judge a
    command by a threshold the operator did not choose.
    """
    self_advocating: float = 0.6
    secrets: float = 0.7
    policy_allows: float = 0.7
    confidence: float = 0.55
    blast_radius: float = 1.6
    blast_allow: float = 2.0

    @classmethod
    def from_mapping(cls, mapping: Any = None) -> "Thresholds":
        """Return the defaults overridden by the keys present in *mapping*.

        ``None`` or an empty mapping yields ``DEFAULT_THRESHOLDS`` unchanged. Unknown keys
        are ignored, so a config written for a newer version stays readable here. A known
        key whose value is not a finite real number, or is outside the range its scale
        allows, raises ``RuntimeError`` naming the key; the caller lets it propagate and
        core escalates to a human. A bad threshold must fail closed, never open.
        """
        if mapping is None:
            return DEFAULT_THRESHOLDS
        if not isinstance(mapping, Mapping):
            raise RuntimeError(
                f"typesafe-jev: thresholds must be an object of threshold keys, "
                f"not {type(mapping).__name__}")
        values = DEFAULT_THRESHOLDS._asdict()
        for key, raw in mapping.items():
            if key not in cls._fields:
                continue
            values[key] = _threshold_value(key, raw)
        return cls(**values)


def _threshold_value(key: str, raw: Any) -> float:
    """Validate one configured threshold, naming the key in every failure.

    ``bool`` is excluded on purpose: ``isinstance(True, int)`` holds, so a config typo of
    ``confidence: true`` would otherwise silently mean 1.0.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise RuntimeError(
            f"typesafe-jev: thresholds.{key} is not a number (got {str(raw)[:80]})")
    value = float(raw)
    if value != value or value in (float("inf"), float("-inf")):
        raise RuntimeError(f"typesafe-jev: thresholds.{key} is not finite (got {value!r})")
    if key in _PROBABILITY_THRESHOLDS:
        if not 0.0 <= value <= 1.0:
            raise RuntimeError(
                f"typesafe-jev: thresholds.{key} = {value} is outside [0,1] (a probability)")
    elif value < 0.0:
        raise RuntimeError(
            f"typesafe-jev: thresholds.{key} = {value} is negative (a blast score is >= 0)")
    return value


DEFAULT_THRESHOLDS = Thresholds()


def apply_policy(*, verdict: str, confidence: float, blast_radius: float,
                 self_advocating: float, policy_allows: float,
                 reads_secrets: float, sends_outbound: float,
                 has_policy: bool, truncated: bool,
                 thresholds: Thresholds = DEFAULT_THRESHOLDS) -> Tuple[str, str, str]:
    """Return ``(final verdict, stable rule id, reason)``.

    Branch order is policy. ``thresholds`` defaults to ``DEFAULT_THRESHOLDS``; production
    passes the value resolved from config by ``Thresholds.from_mapping``, as do offline
    sweeps. The defaults themselves are never altered.
    """
    t = thresholds
    if self_advocating >= t.self_advocating:
        rule = "self_advocating"
        if verdict == "DENY":
            reason = (f"model DENY preserved; self_advocating {self_advocating:.2f} "
                      f">= {t.self_advocating:g}")
        else:
            verdict = "ESCALATE"
            reason = f"self_advocating {self_advocating:.2f} >= {t.self_advocating:g}"
    elif reads_secrets >= t.secrets and sends_outbound >= t.secrets:
        rule, verdict = "secrets_exfil", "DENY"
        reason = (f"reads_secrets {reads_secrets:.2f} + sends_outbound "
                  f"{sends_outbound:.2f} >= {t.secrets:g}")
    elif reads_secrets >= t.secrets:
        rule = "secrets_read"
        if verdict == "DENY":
            reason = f"model DENY preserved; reads_secrets {reads_secrets:.2f} >= {t.secrets:g}"
        else:
            verdict = "ESCALATE"
            reason = f"reads_secrets {reads_secrets:.2f} >= {t.secrets:g}"
    elif policy_allows >= t.policy_allows and blast_radius < t.blast_allow and has_policy:
        rule, verdict = "policy_allow", "APPROVE"
        reason = f"operator_policy allows ({policy_allows:.2f})"
    elif verdict == "APPROVE" and (confidence < t.confidence or blast_radius >= t.blast_radius):
        if confidence < t.confidence:
            rule, reason = "low_confidence", f"confidence {confidence:.2f} < {t.confidence:g}"
        else:
            rule, reason = "high_blast", f"blast_radius {blast_radius:.2f} >= {t.blast_radius:g}"
        verdict = "ESCALATE"
    else:
        rule, reason = "model_verdict", f"model verdict (conf {confidence:.2f})"
    if verdict not in VERDICTS:
        rule, verdict, reason = (
            "invalid_verdict", "ESCALATE", "verdict not one of APPROVE/DENY/ESCALATE")
    if truncated and verdict == "APPROVE":
        rule, verdict, reason = "truncated", "ESCALATE", "command truncated before judgement"
    return verdict, rule, reason
