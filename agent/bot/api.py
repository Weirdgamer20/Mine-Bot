from fastapi import FastAPI
from .agent import LearningAgent
from .schemas import Observation, StepRequest, StepResponse, Action

app = FastAPI(title="Minecraft Learning Agent API", version="0.1.0")
agent = LearningAgent()

@app.get("/health")
def health():
    return {"status": "ok", "device": str(agent.device)}

@app.get("/state")
def state():
    return {"device": str(agent.device), "memory": len(agent.memory)}

@app.post("/v1/reset")
def reset():
    agent.reset_episode()
    return {"status": "episode_reset"}

@app.post("/v1/observe")
def observe(obs: Observation):
    action, learning = agent.step(obs)
    return StepResponse(action=Action(**action), learning=learning)

@app.post("/v1/step")
def step(req: StepRequest):
    if req.done:
        agent.reset_episode()
    action, learning = agent.step(req.observation)
    return StepResponse(action=Action(**action), learning=learning)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bot.api:app", host="0.0.0.0", port=8000, reload=False)
