import threading
import time
import torch
from typing import Dict, Any, Optional
from ..learning import train_world_model_step, train_actor_critic_imagination

class AsyncLearnerThread(threading.Thread):
    """
    Decoupled Asynchronous Learner Thread.
    Continuously optimizes World Model and Actor-Critic parameters in the background
    without stalling the real-time Minecraft stream actor.
    """
    def __init__(
        self,
        agent,
        batch_size: int = 16,
        seq_len: int = 16,
        horizon: int = 12,
        sleep_interval: float = 0.02,
    ):
        super().__init__(daemon=True)
        self.agent = agent
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.horizon = horizon
        self.sleep_interval = sleep_interval
        self.running = False
        self.latest_metrics: Dict[str, float] = {}
        self.lock = threading.Lock()

    def run(self):
        self.running = True
        print("[Learner] Asynchronous Learner thread started.")

        while self.running:
            # Need minimum number of steps in memory
            if len(self.agent.memory) < (self.batch_size * self.seq_len):
                time.sleep(0.5)
                continue

            try:
                batch = self.agent.memory.sample_sequences(
                    batch_size=self.batch_size,
                    seq_len=self.seq_len,
                    device=self.agent.device,
                )
                if batch is None:
                    time.sleep(0.1)
                    continue

                # 1. World Model & RND Training Step
                wm_loss, wm_metrics, last_h, last_z = train_world_model_step(
                    self.agent.encoder,
                    self.agent.world_model,
                    batch,
                    kl_weight=self.agent.cfg.kl_weight,
                    continuation_weight=self.agent.cfg.continuation_weight,
                    rnd=self.agent.rnd,
                )

                self.agent.wm_opt.zero_grad()
                wm_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.wm_params, max_norm=10.0)
                self.agent.wm_opt.step()

                # 2. Actor-Critic Latent Imagination Step
                ac_loss, ac_metrics = train_actor_critic_imagination(
                    self.agent.actor_critic,
                    self.agent.world_model,
                    self.agent.skill_net,
                    start_h=last_h,
                    start_z=last_z,
                    horizon=self.horizon,
                    gamma=self.agent.cfg.gamma,
                    lambda_gae=self.agent.cfg.lambda_gae,
                )

                self.agent.ac_opt.zero_grad()
                ac_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.actor_critic.parameters(), max_norm=10.0)
                self.agent.ac_opt.step()

                with self.lock:
                    self.latest_metrics = {**wm_metrics, **ac_metrics}

            except (KeyboardInterrupt, SystemExit):
                self.running = False
                break
            except Exception as e:
                if self.running:
                    print(f"[Learner] Background training step warning: {e}")

            time.sleep(self.sleep_interval)

    def get_metrics(self) -> Dict[str, float]:
        """Thread-safe retrieval of latest training metrics."""
        with self.lock:
            return dict(self.latest_metrics)

    def stop(self):
        self.running = False
