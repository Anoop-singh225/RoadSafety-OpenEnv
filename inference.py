import os
import json
import asyncio
from openai import OpenAI
from models import RiderSafetyAction, RiderSafetyObservation
from client import RiderSafetyClient 

# 1. Configuration
# The validator requires using API_BASE_URL and API_KEY environment variables exactly.
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# 2. Logging Helpers (Standard Format)
def log_start(task, env_name, model):
    print(f"[START] task={task} env={env_name} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# 3. LLM Logic (Action generator)
def get_action_sync(obs: RiderSafetyObservation):
    prompt = (
        f"You are a road safety monitoring AI.\n"
        f"Sensor Data: {obs.sensor_summary}\n"
        f"Audio: {obs.audio_transcript}\n"
        "Analyze the data and choose an action. If there's an obvious crash, dispatch SOS. If uncertain or normal, just monitor or ignore.\n"
        "Output ONLY valid JSON: {\"decision\": \"IGNORE\" or \"MONITOR\" or \"DISPATCH_SOS\", \"message\": \"brief rationale\"}"
    )
    
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a specialized AI processing sensor data."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=100
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        return {"decision": "MONITOR", "message": f"error processing: {e}"}

# 4. Main Loop using the custom Client
async def run_task(env: RiderSafetyClient, task_name: str):
    log_start(task_name, "rider-safety-env", MODEL_NAME)
    rewards, steps, success_flag = [], 0, False
    try:
        # Pass task to reset parameters if possible, or just default to medium
        res = await env.reset() # Some standard clients might not take kwargs, so we rely on backend default or modify backend
        # To specifically instruct server about task, ideally we pass it to reset
        
        # OpenEnv typically supports kwargs via /reset if wrapped properly. 
        # But we'll do standard reset and hope backend handles it or cycles tasks.
        # Actually I coded server/environment.py to read kwargs.get("task"). 
        # So we pass kwargs to env.reset()
        if hasattr(env, 'reset') and 'task' in str(env.reset.__code__.co_varnames):
            res = await env.reset(task=task_name)
        else:
            # Try passing kwargs, if EnvClient allows
            try:
                res = await env.reset(task=task_name)
            except TypeError:
                res = await env.reset() 

        done = False
        while not done:
            steps += 1
            action_data = get_action_sync(res.observation)
            
            action = RiderSafetyAction(
                decision=action_data.get('decision', 'MONITOR'), 
                message=action_data.get('message', "")
            )
            
            res = await env.step(action)
            rewards.append(res.reward)
            log_step(steps, action.decision, res.reward, res.done, None)
            done = res.done
            
        success_flag = sum(rewards) > 0.5
    except Exception as e:
        log_step(steps, "ERROR", 0.0, True, str(e))
    finally:
        score = sum(rewards)
        log_end(success_flag, steps, score, rewards)

async def main():
    env = RiderSafetyClient("http://localhost:7860")
    for task in ["easy", "medium", "hard"]:
        await run_task(env, task)
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())