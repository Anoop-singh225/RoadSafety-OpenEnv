from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import RiderSafetyAction, RiderSafetyObservation, RiderSafetyState

class RiderSafetyClient(EnvClient[RiderSafetyAction, RiderSafetyObservation, RiderSafetyState]):
    
    def _step_payload(self, action: RiderSafetyAction) -> dict:
        return {
            "decision": action.decision,
            "message": action.message or ""
        }

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        
        observation = RiderSafetyObservation(
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            sensor_summary=obs_data.get("sensor_summary", ""),
            audio_transcript=obs_data.get("audio_transcript", "")
        )
        
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: dict) -> RiderSafetyState:
        return RiderSafetyState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            target_goal=payload.get("target_goal", "Ensure Rider Safety"),
            max_turns=payload.get("max_turns", 3),
            crash_occurred=payload.get("crash_occurred", False),
            false_alarm=payload.get("false_alarm", False),
            sos_dispatched=payload.get("sos_dispatched", False),
            task_name=payload.get("task_name", "medium")
        )