import sys
import os
sys.path.insert(0, '/mnt/d/minecraft_learning_bot/agent')

import torch
from bot.agent import LearningAgent
from bot.planning.latent_planner import LatentMPCPlanner
from bot.memory.replay import PrioritizedSequenceBuffer
from bot.models import RecurrentWorldModel, SkillDiscovery
from bot.training.learner import AsyncLearnerThread

print('[TEST] Imports succeeded!')

# Verify LatentMPCPlanner vectorization
wm = RecurrentWorldModel()
from bot.models import HierarchicalActorCritic
ac = HierarchicalActorCritic()
planner = LatentMPCPlanner(wm, ac, horizon=4, num_candidates=6)
h = torch.zeros(1, wm.hidden_dim)
z = torch.zeros(1, wm.latent_dim)
skill = torch.zeros(1, 8)
motor, prim, score = planner.plan_best_action(h, z, skill)
print(f'[TEST] Vectorized MPC rollout passed! Best prim: {prim}, Score: {score:.4f}')

# Verify LearningAgent init and AsyncLearnerThread
agent = LearningAgent()
print(f'[TEST] LearningAgent initialized on device: {agent.device}')
print(f'[TEST] AsyncLearnerThread alive: {agent.learner_thread.is_alive()}')

# Test replay serialization
state = agent.memory.to_dict()
agent.memory.load_from_dict(state)
print(f'[TEST] Replay serialization passed! Total steps: {len(agent.memory)}')

agent.learner_thread.stop()
print('[TEST] ALL M10 VERIFICATIONS PASSED!')
