"""
Rule-based Risk Scoring Engine for Revio.

Input : fault_label (str), confidence (float 0-1)
Output: risk_tier (str), failure_probability (float %)

Rules ( do not change to ML):
  fault_label contains "normal"  → NORMAL
  confidence >= 0.85             → CRITICAL
  confidence >= 0.60             → WARNING
  confidence <  0.60             → MONITOR
"""

from dataclasses import dataclass

TIER_NORMAL   = "NORMAL"
TIER_MONITOR  = "MONITOR"
TIER_WARNING  = "WARNING"
TIER_CRITICAL = "CRITICAL"

# Colour codes for demo UI
TIER_COLOURS = {
    TIER_NORMAL:   "#2ECC71",   # green
    TIER_MONITOR:  "#F1C40F",   # yellow
    TIER_WARNING:  "#E67E22",   # orange
    TIER_CRITICAL: "#E74C3C",   # red
}

# Confidence thresholds
_CRITICAL_THRESHOLD = 0.85
_WARNING_THRESHOLD  = 0.60


@dataclass
class RiskResult:
    fault_label:         str
    confidence:          float
    risk_tier:           str
    failure_probability: float   # percentage 0-100
    colour:              str
    should_alert:        bool    # True for CRITICAL and WARNING only


def score(fault_label: str, confidence: float) -> RiskResult:
    """Compute risk tier and failure probability from model output.

    Parameters
    ----------
    fault_label : class name returned by CNN (e.g. 'worn_out_brakes')
    confidence  : max softmax probability (0-1)

    Returns
    -------
    RiskResult dataclass
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")

    is_normal = "normal" in fault_label.lower()

    if is_normal:
        tier = TIER_NORMAL
    elif confidence >= _CRITICAL_THRESHOLD:
        tier = TIER_CRITICAL
    elif confidence >= _WARNING_THRESHOLD:
        tier = TIER_WARNING
    else:
        tier = TIER_MONITOR

    failure_prob = _compute_failure_probability(tier, confidence, is_normal)

    return RiskResult(
        fault_label=fault_label,
        confidence=confidence,
        risk_tier=tier,
        failure_probability=failure_prob,
        colour=TIER_COLOURS[tier],
        should_alert=tier in (TIER_CRITICAL, TIER_WARNING),
    )


def _compute_failure_probability(
    tier: str, confidence: float, is_normal: bool
) -> float:
    """Map confidence + tier to a human-readable failure probability %.

    Scaled so the number feels intuitive to a driver:
      NORMAL   → 0-5 %
      MONITOR  → 10-40 %
      WARNING  → 40-70 %
      CRITICAL → 70-99 %
    """
    if is_normal:
        return round(confidence * 5, 1)          # 0-5 %

    if tier == TIER_MONITOR:
        # confidence < 0.60 → map [0, 0.60) → [10, 40)
        return round(10 + (confidence / 0.60) * 30, 1)

    if tier == TIER_WARNING:
        # confidence in [0.60, 0.85) → map → [40, 70)
        normalised = (confidence - 0.60) / 0.25
        return round(40 + normalised * 30, 1)

    # CRITICAL: confidence in [0.85, 1.0] → map → [70, 99]
    normalised = (confidence - 0.85) / 0.15
    return round(70 + normalised * 29, 1)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        ("normal_brakes",       0.97),
        ("worn_out_brakes",     0.91),
        ("worn_out_brakes",     0.72),
        ("worn_out_brakes",     0.45),
        ("normal_engine_idle",  0.88),
        ("low_oil",             0.86),
        ("serpentine_belt",     0.63),
        ("power_steering",      0.51),
    ]
    print(f"{'Label':<25} {'Conf':>5}  {'Tier':<10} {'Fail%':>6}  Alert")
    print("-" * 60)
    for label, conf in cases:
        r = score(label, conf)
        print(f"{label:<25} {conf:>5.2f}  {r.risk_tier:<10} {r.failure_probability:>5.1f}%  "
              f"{'YES' if r.should_alert else 'no'}")
