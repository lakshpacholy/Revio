"""
Offline Text-to-Speech alerts for Revio (pyttsx3 only — no internet).

Rules (from CLAUDE.md):
  - Library: pyttsx3 (never gTTS or any API-based TTS)
  - Speech rate : 150
  - Volume      : 0.9
  - Speak only on CRITICAL or WARNING tier
  - NORMAL and MONITOR tiers are silent
  - Mute toggle respected (is_muted flag)
"""

from src.risk_scorer import TIER_CRITICAL, TIER_WARNING


def _get_engine():
    """Initialise and return a configured pyttsx3 engine instance."""
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate",   150)
    engine.setProperty("volume", 0.9)
    return engine


def speak(text: str, block: bool = True) -> None:
    """Speak text synchronously using pyttsx3.

    Parameters
    ----------
    text  : text to speak
    block : if True (default), blocks until speech finishes
    """
    engine = _get_engine()
    engine.say(text)
    if block:
        engine.runAndWait()
    engine.stop()


def alert(
    narrative: str,
    risk_tier: str,
    is_muted: bool = False,
    block: bool = True,
) -> bool:
    """Speak the narrative if the risk tier warrants a spoken alert.

    Parameters
    ----------
    narrative : text from narrative.get_narrative()
    risk_tier : tier string from risk_scorer.score()
    is_muted  : if True, never speak regardless of tier
    block     : passed through to speak()

    Returns
    -------
    bool — True if speech was triggered, False if silent
    """
    if is_muted:
        return False
    if risk_tier not in (TIER_CRITICAL, TIER_WARNING):
        return False

    speak(narrative, block=block)
    return True


def speak_startup_message() -> None:
    """Play a brief system-ready message when the demo UI launches."""
    speak("Revio vehicle health monitor is online. Listening for faults.")


# ---------------------------------------------------------------------------
# Quick self-test (requires speakers)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.risk_scorer import score
    from src.narrative import get_narrative

    test_cases = [
        ("worn_out_brakes",  0.91),   # CRITICAL → speaks
        ("serpentine_belt",  0.65),   # WARNING  → speaks
        ("low_oil",          0.45),   # MONITOR  → silent
        ("normal_brakes",    0.97),   # NORMAL   → silent
    ]

    for label, conf in test_cases:
        result    = score(label, conf)
        narrative = get_narrative(label, result.risk_tier)
        triggered = alert(narrative, result.risk_tier)
        status = "SPEAKING" if triggered else "silent"
        print(f"[{result.risk_tier:<10}] {label:<22} conf={conf:.2f}  -> {status}")
