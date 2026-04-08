---
title: RoadSafety OpenEnv
emoji: 🏍️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---
# Road Safety OpenEnv

This is a fully compliant OpenEnv project modeling a real-world task: AI-driven autonomous crash detection and emergency response system.

## Motivation & Domain
Every day, thousands of riders experience accidents. Immediate SOS dispatch saves lives. This environment uses a realistic IMU dataset containing simulated gyroscope, accelerometer, speed, and motion intensity readings representing motorcycle riders in transit. 

The goal of the agent is to read these sequential data frames (simulating an onboard edge AI) and correctly decipher if simple motion variances are normal commuting or a crash requiring SOS dispatch.

## Features
- **Real-World Task Simulation**: Processes 10-second data windows of 6-axis IMU sensors, speed, and overall motion intensity metrics.
- **OpenEnv Spec Compliance**: Implemented using `openenv-core`. Includes `openenv.yaml`. Uses strict Pydantic typed input/output models (`RiderSafetyObservation`, `RiderSafetyAction`).
- **3 Task Graders**:
    - **Easy**: Identifies completely normal driving vs high intensity clear crashes. Graded exactly.
    - **Medium**: Partial rewards given. Evaluates edge case driving behaviors and borderline intense movements.
    - **Hard**: Complex anomalies. False alarms are strictly penalized with 0.0 reward.
- **Robust Reward Design**: Rewards emphasize correct identification and heavily penalize false emergency deployments to mimic a real service constraint.

## Setup and Usage

### Prerequisites
- Python 3.9+
- Docker (optional but recommended for Hugging Face Spaces integration)

### Local Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the OpenEnv server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Running Inference Baseline
This environment includes a completed `inference.py` ready to benchmark LLMs.
```bash
# Set specific credentials or use environment defaults
export API_BASE_URL="https://api.together.xyz/v1"
export MODEL_NAME="meta-llama/Llama-3-8b-chat-hf"
export HF_TOKEN="HF_TOKEN"

# Run inference
python inference.py
```

### Docker Deployment
Ready for HuggingFace Spaces.
```bash
docker build -t roadsafety-env .
docker run -p 8000:8000 roadsafety-env
```

## Environment Spaces
**Observation Space**:
- `step_index` (int): Current step in the sequence.
- `sensor_summary` (str): Summary string of speed, motion intensity, and 3D acceleration.
- `audio_transcript` (str): Environmental context.

**Action Space**:
- `decision` (str): Action to take. Must be one of `IGNORE`, `MONITOR`, or `DISPATCH_SOS`.
- `message` (str): The reasoning behind the decision.

## Tasks Explained
Each sequence dynamically tests the agent's ability to maintain composure or react decisively. We load data from `road_accident_imu_dataset_8000.csv` to ensure varied telemetry. Scores enforce high precision and high recall performance constraints.
