from typing import Optional
from openenv.core.env_server import Action, Observation, State

class RiderSafetyAction(Action):
    decision: str  # IGNORE, MONITOR, DISPATCH_SOS
    message: Optional[str] = None

class RiderSafetyObservation(Observation):
    done: bool
    reward: float
    sensor_summary: str
    audio_transcript: str

class RiderSafetyState(State):
    episode_id: str = ""
    step_count: int = 0
    target_goal: str = "Ensure Rider Safety"
    max_turns: int = 3
    crash_occurred: bool = False
    false_alarm: bool = False
    sos_dispatched: bool = False
    task_name: str = "medium"