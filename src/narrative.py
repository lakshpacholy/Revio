"""
Template-based Narrative Generator for Revio.

Produces a short, driver-friendly alert sentence for every
(state, fault_label, risk_tier) combination.

Rules (from CLAUDE.md):
  - No LLM, no API, no internet - fully offline templates
  - One narrative per (state, fault_label, risk_tier)
  - Driver-friendly language, 1-2 sentences, clear action instruction
  - normal label -> "All systems normal. No faults detected."
"""

from src.risk_scorer import TIER_NORMAL, TIER_MONITOR, TIER_WARNING, TIER_CRITICAL

# Template store: {(fault_label_key, risk_tier): narrative_text}
# fault_label_key is a substring matched against the CNN output label.
_NARRATIVES: dict[tuple[str, str], str] = {

    # BRAKING
    ("worn_out_brakes", TIER_CRITICAL): (
        "Critical brake wear detected. Your brake pads are dangerously thin - "
        "stopping distance is severely compromised. Pull over safely and do not "
        "drive until brakes are replaced immediately."
    ),
    ("worn_out_brakes", TIER_WARNING): (
        "Brake wear warning. Unusual braking sounds suggest your pads are worn. "
        "Schedule a brake inspection within the next 24 hours."
    ),
    ("worn_out_brakes", TIER_MONITOR): (
        "Slight brake irregularity detected. Monitor braking feel over the next "
        "few days and have pads checked at your next service."
    ),

    # START-UP
    ("bad_ignition", TIER_CRITICAL): (
        "Critical ignition fault. The engine is struggling to start reliably - "
        "ignition components may be failing. Avoid turning off the engine; "
        "seek roadside assistance immediately."
    ),
    ("bad_ignition", TIER_WARNING): (
        "Ignition irregularity detected. The startup sound suggests the ignition "
        "system needs attention. Have it inspected by a mechanic soon."
    ),
    ("bad_ignition", TIER_MONITOR): (
        "Minor startup anomaly noted. Keep an eye on how the engine starts over "
        "the next few days and report any worsening to your mechanic."
    ),
    ("dead_battery", TIER_CRITICAL): (
        "Critical battery failure. Your battery cannot reliably power the vehicle - "
        "the engine may not restart once stopped. Drive directly to a garage "
        "or call for assistance now."
    ),
    ("dead_battery", TIER_WARNING): (
        "Battery health warning. Startup sounds indicate a weakening battery. "
        "Have the battery tested and replaced if necessary within the next few days."
    ),
    ("dead_battery", TIER_MONITOR): (
        "Battery irregularity detected. The battery may be losing charge capacity. "
        "Consider a battery health check at your next service."
    ),

    # IDLE
    ("low_oil", TIER_CRITICAL): (
        "Critical low oil alert. Engine lubrication is severely insufficient - "
        "continuing to drive risks permanent engine damage. Stop safely, check "
        "the oil level immediately, and do not restart without topping up."
    ),
    ("low_oil", TIER_WARNING): (
        "Low oil level warning. Idle sounds suggest reduced lubrication. "
        "Check and top up engine oil as soon as possible."
    ),
    ("low_oil", TIER_MONITOR): (
        "Possible oil level irregularity. Check your oil dipstick when next "
        "parked and top up if below the minimum mark."
    ),
    ("power_steering", TIER_CRITICAL): (
        "Critical power steering fault. Steering assist may fail suddenly, "
        "making the vehicle very hard to control. Pull over safely and call "
        "for assistance - do not continue driving."
    ),
    ("power_steering", TIER_WARNING): (
        "Power steering irregularity detected. Unusual idle sounds suggest "
        "the power steering system needs inspection. Avoid sharp manoeuvres "
        "and have it checked today."
    ),
    ("power_steering", TIER_MONITOR): (
        "Minor power steering anomaly. Have the steering fluid level and pump "
        "inspected at your next service appointment."
    ),
    ("serpentine_belt", TIER_CRITICAL): (
        "Critical serpentine belt fault. The belt may snap at any moment - "
        "this will disable power steering, the alternator, and cooling. "
        "Stop the vehicle immediately and call for assistance."
    ),
    ("serpentine_belt", TIER_WARNING): (
        "Serpentine belt wear detected. A squealing or irregular idle suggests "
        "the belt is worn or misaligned. Have it inspected urgently - "
        "belt failure leaves you stranded."
    ),
    ("serpentine_belt", TIER_MONITOR): (
        "Possible serpentine belt irregularity. Schedule an inspection within "
        "the next week to avoid unexpected belt failure."
    ),
}

_NORMAL_NARRATIVE = "All systems normal. No faults detected."


def get_narrative(fault_label: str, risk_tier: str) -> str:
    """Return a driver-friendly narrative for the given fault and risk tier.

    Parameters
    ----------
    fault_label : class name from CNN (e.g. 'worn_out_brakes')
    risk_tier   : one of NORMAL / MONITOR / WARNING / CRITICAL

    Returns
    -------
    str - narrative text ready to display and speak
    """
    if risk_tier == TIER_NORMAL or "normal" in fault_label.lower():
        return _NORMAL_NARRATIVE

    for (label_key, tier), text in _NARRATIVES.items():
        if label_key in fault_label and tier == risk_tier:
            return text

    # Fallback - should never be reached with valid inputs
    return (
        f"Anomaly detected in {fault_label.replace('_', ' ')} "
        f"({risk_tier}). Please consult a mechanic."
    )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.risk_scorer import score

    test_cases = [
        ("normal_brakes",    0.95),
        ("worn_out_brakes",  0.91),
        ("worn_out_brakes",  0.72),
        ("worn_out_brakes",  0.45),
        ("bad_ignition",     0.88),
        ("dead_battery",     0.65),
        ("low_oil",          0.91),
        ("power_steering",   0.62),
        ("serpentine_belt",  0.55),
    ]

    for label, conf in test_cases:
        result    = score(label, conf)
        narrative = get_narrative(label, result.risk_tier)
        print(f"\n[{result.risk_tier}] {label} ({conf:.0%} conf)")
        print(f"  Fail prob : {result.failure_probability}%")
        print(f"  Narrative : {narrative}")
