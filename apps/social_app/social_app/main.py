import threading
import time

import numpy as np
from pydantic import BaseModel

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.utils import create_head_pose

from social_app.gaze import GazeConfig, GazeController
from social_app.perception import WebcamFaceTracker

LOOP_HZ = 50.0
LOOP_PERIOD_S = 1.0 / LOOP_HZ


class SocialApp(ReachyMiniApp):
    # Optional: URL to a custom configuration page for the app
    # eg. "http://localhost:8042"
    custom_app_url: str | None = "http://0.0.0.0:8042"
    # Optional: specify a media backend ("gstreamer", "gstreamer_no_video", "default", etc.)
    # On the wireless, use gstreamer_no_video to optimise CPU usage if the app does not use video streaming
    request_media_backend: str | None = None

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        t0 = time.monotonic()

        antennas_enabled = True
        sound_play_requested = False

        # You can ignore this part if you don't want to add settings to your app. If you set custom_app_url to None, you have to remove this part as well.
        # === vvv ===
        class AntennaState(BaseModel):
            enabled: bool

        @self.settings_app.post("/antennas")
        def update_antennas_state(state: AntennaState):
            nonlocal antennas_enabled
            antennas_enabled = state.enabled
            return {"antennas_enabled": antennas_enabled}

        @self.settings_app.post("/play_sound")
        def request_sound_play():
            nonlocal sound_play_requested
            sound_play_requested = True

        # === ^^^ ===

        tracker = WebcamFaceTracker()
        gaze = GazeController(GazeConfig())
        tracker.start()
        try:
            next_tick = time.monotonic()
            # Main control loop -- the only place in this app that calls set_target().
            while not stop_event.is_set():
                now = time.monotonic()
                t = now - t0

                yaw_deg, pitch_deg = gaze.update(tracker.latest(), now)
                head_pose = create_head_pose(yaw=yaw_deg, pitch=pitch_deg, degrees=True)

                if antennas_enabled:
                    amp_deg = 25.0
                    a = amp_deg * np.sin(2.0 * np.pi * 0.5 * t)
                    antennas_deg = np.array([a, -a])
                else:
                    antennas_deg = np.array([0.0, 0.0])

                if sound_play_requested:
                    print("Playing sound...")
                    reachy_mini.media.play_sound("wake_up.wav")
                    sound_play_requested = False

                antennas_rad = np.deg2rad(antennas_deg)

                reachy_mini.set_target(
                    head=head_pose,
                    antennas=antennas_rad,
                )

                next_tick += LOOP_PERIOD_S
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.monotonic()
        finally:
            tracker.stop()


if __name__ == "__main__":
    app = SocialApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
