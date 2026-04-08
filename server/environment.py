import random
import uuid
import os
import pandas as pd
from typing import Dict, Any

from openenv.core.env_server import Environment
from models import RiderSafetyAction, RiderSafetyObservation, RiderSafetyState

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "road_accident_imu_dataset_8000.csv")

class RiderSafetyEnv(Environment):
    def __init__(self):
        self._state = RiderSafetyState()
        self._step_count = 0
        self._sequence = []
        
        # Load dataset once
        if os.path.exists(DATASET_PATH):
            self.df = pd.read_csv(DATASET_PATH)
            self.crash_indices = self.df[self.df['Crash_Label'] == 1].index.tolist()
            self.normal_indices = self.df[self.df['Crash_Label'] == 0].index.tolist()
        else:
            self.df = None

    def reset(self, seed=None, episode_id=None, task=None, **kwargs) -> RiderSafetyObservation:
        self._step_count = 0
        self._state = RiderSafetyState()
        self._state.episode_id = episode_id or str(uuid.uuid4())
        
        # Determine task difficulty from kwargs or default
        task_name = kwargs.get("task", task or "medium").lower()
        self._state.task_name = task_name
        self._state.target_goal = f"Successfully complete {task_name} task"
        
        # We simulate 3 to 5 step sequences.
        self._state.max_turns = random.randint(3, 5)

        if self.df is None:
            self._sequence = [{"Speed_kmh": 40, "Acc_X": 0, "Acc_Y": 0, "Crash_Label": 0, "Motion_Intensity": 9.8}] * self._state.max_turns
        else:
            # Pick scenario based on task difficulty
            if task_name == "easy":
                # Easy: Very obvious crash or purely normal driving.
                is_crash = random.choice([True, False])
            elif task_name == "medium":
                # Medium: High variance normal driving and borderline crashes.
                # Biasing towards more crashes
                is_crash = random.random() < 0.6
            else:
                # Hard: Tricky edge cases (e.g. high intensity but no crash, or low intensity crash)
                is_crash = random.random() < 0.7

            if is_crash and self.crash_indices:
                start_idx = max(0, random.choice(self.crash_indices) - self._state.max_turns + 2) # ensure crash is inside sequence
            else:
                start_idx = random.choice(self.normal_indices)
                
            start_idx = min(start_idx, max(0, len(self.df) - self._state.max_turns))
            end_idx = start_idx + self._state.max_turns
            self._sequence = self.df.iloc[start_idx:end_idx].to_dict('records')

        self._state.crash_occurred = any(row.get('Crash_Label', 0) == 1 for row in self._sequence)

        first_obs = self._sequence[0]
        return self._create_observation(first_obs, 0.0, False)

    def step(self, action: RiderSafetyAction) -> RiderSafetyObservation:
        self._step_count += 1
        
        if action.decision == "DISPATCH_SOS":
            self._state.sos_dispatched = True
            
        done = self._step_count >= self._state.max_turns
        
        # Calculate rewards based on Graders criteria (task specific)
        reward = 0.0
        if done:
            reward = self._grade_task()

        # Get current data row, simulate transcript based on crash label
        obs_idx = min(self._step_count, len(self._sequence)-1)
        obs_data = self._sequence[obs_idx]
        
        return self._create_observation(obs_data, reward, done)

    def _grade_task(self) -> float:
        # Grading logic maps directly to openenv grader requirements (0.0 to 1.0)
        # Easy: correct binary response string
        # Medium: partial progress (if model monitors until dispatch is necessary, gets full, otherwise penalised)
        # Hard: High penalty for false alarms
        
        # Baseline True Positive vs False Alarm
        is_true_positive = self._state.crash_occurred and self._state.sos_dispatched
        is_true_negative = not self._state.crash_occurred and not self._state.sos_dispatched
        is_false_alarm = not self._state.crash_occurred and self._state.sos_dispatched
        is_false_negative = self._state.crash_occurred and not self._state.sos_dispatched

        score = 0.0
        if self._state.task_name == "easy":
            # Simple binary grading
            if is_true_positive or is_true_negative: score = 1.0
            else: score = 0.0
            
        elif self._state.task_name == "medium":
            # Partial rewards
            if is_true_positive: score = 1.0
            elif is_true_negative: score = 1.0
            elif is_false_alarm: score = 0.2  # Erring on side of safety
            else: score = 0.0
                
        else: # Hard Task
            if is_true_positive:
                score = 1.0
            elif is_true_negative:
                score = 1.0
            elif is_false_alarm:
                score = 0.0  # Explicitly penalizing False Alarms
            else:
                score = 0.0
        
        # Meta Hackathon Validator requires scores strictly in (0, 1)
        return max(0.001, min(0.999, score))

        
        # IMPORTANT: Meta Hackathon Validator requires scores strictly between 0 and 1
        # (not exactly 0.0 and not exactly 1.0)
        return max(0.001, min(0.999, score))

    def _create_observation(self, row: Dict[str, Any], reward: float, done: bool) -> RiderSafetyObservation:
        # Generate contextual transcript
        transcript = "Normal background noise"
        if row.get('Crash_Label', 0) == 1:
            transcript = "Loud bang! Tires screeching! Screaming!"
        elif row.get('Motion_Intensity', 0.0) > 10.0:
            transcript = "Heavy wind, screeching tires"

        sensor_summary = f"Speed: {row.get('Speed_kmh', 0):.1f}kmph, "\
                         f"Motion Intensity: {row.get('Motion_Intensity', 0):.2f}, "\
                         f"Acc(X/Y/Z): {row.get('Acc_X',0):.2f}/{row.get('Acc_Y',0):.2f}/{row.get('Acc_Z',0):.2f}"

        return RiderSafetyObservation(
            done=done,
            reward=reward,
            sensor_summary=sensor_summary,
            audio_transcript=transcript
        )

    @property
    def state(self) -> RiderSafetyState:
        return self._state