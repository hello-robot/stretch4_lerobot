"""Speak LeRobot's announcements ("Recording episode 3", "Reset the environment", ...) in the
Stretch 4 voice instead of the system's espeak.
"""

import hashlib
import logging
import os
import subprocess
import threading
from pathlib import Path

log = logging.getLogger(__name__)

CACHE = Path(os.environ.get("LEROBOT_STRETCH4_VOICE_CACHE", Path.home() / ".cache/lerobot_stretch4/voice"))
LANG, TLD = "en", "com.au"      # stretch4_body's voice: gTTS "en-au"

# Phrases lerobot-record / lerobot-rollout say; generated in the background at connect so the
# first announcement is not delayed by a network request.
COMMON = ["Reset the environment", "Re-record episode", "Stop recording", "Exiting",
          *[f"Recording episode {i}" for i in range(30)]]

_upstream_say = None
_lock = threading.Lock()


def wav_for(text: str) -> Path | None:
    """The cached WAV for `text`, generating it if needed; None if it cannot be made."""
    path = CACHE / f"{hashlib.sha1(f'{LANG}|{TLD}|{text}'.encode()).hexdigest()[:16]}.wav"
    if path.exists():
        return path
    with _lock:
        if path.exists():
            return path
        CACHE.mkdir(parents=True, exist_ok=True)
        mp3, tmp_wav = path.with_suffix(".tmp.mp3"), path.with_suffix(".tmp.wav")
        try:
            from gtts import gTTS
            gTTS(text=text, lang=LANG, tld=TLD).save(str(mp3))
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3), str(tmp_wav)],
                           check=True, timeout=30)
            tmp_wav.rename(path)   # atomic: a half-written file is never played
            return path
        except Exception as e:
            log.debug("stretch voice unavailable for %r: %s", text, e)
            return None
        finally:
            mp3.unlink(missing_ok=True)
            tmp_wav.unlink(missing_ok=True)


def _play(path: Path, blocking: bool):
    if blocking:
        subprocess.run(["aplay", "-q", str(path)], check=False)
        return
    try:
        from stretch4_body.core.hello_utils import play_sound   # same player as the robot's own sounds
        play_sound(str(path))
    except Exception:
        subprocess.Popen(["aplay", "-q", str(path)])


def say(text: str, blocking: bool = False):
    path = wav_for(text)
    if path is None:
        return _upstream_say(text, blocking)
    _play(path, blocking)


def install(prewarm: bool = True):
    """Route LeRobot's say() through the Stretch voice (idempotent)."""
    global _upstream_say
    import lerobot.utils.utils as lr_utils
    if lr_utils.say is say:
        return
    _upstream_say = lr_utils.say
    lr_utils.say = say
    if prewarm:
        threading.Thread(target=lambda: [wav_for(t) for t in COMMON], daemon=True,
                         name="stretch4-voice-prewarm").start()


def main():
    """stretch4_say "text": speak a phrase in the Stretch voice (and cache it)."""
    import argparse
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("text")
    args = p.parse_args()
    path = wav_for(args.text)
    if path is None:
        raise SystemExit("could not generate speech (is gTTS installed and the network up?)")
    _play(path, blocking=True)
