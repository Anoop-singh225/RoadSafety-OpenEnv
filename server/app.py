from openenv.core.env_server import create_fastapi_app
from .environment import RiderSafetyEnv
from models import RiderSafetyAction, RiderSafetyObservation

# Ye function automatically /ws, /reset, /step endpoints bana dega
app = create_fastapi_app(
    RiderSafetyEnv,
    action_cls=RiderSafetyAction,
    observation_cls=RiderSafetyObservation
)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()