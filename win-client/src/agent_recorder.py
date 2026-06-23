import datetime
import logging
from pathlib import Path

import gym
import numpy as np
from src.data_writer import HDF5Writer
from src.settings import cfg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OUNoise:
    def __init__(
        self,
        action_dimension,
        mu=0.0,
        theta=cfg.agent.ou_theta,
        sigma=cfg.agent.ou_sigma,
    ):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


def load_policy(path):
    if path is None:
        logging.info("No policy path provided. Using random action policy.")
        return lambda obs: np.random.uniform(-1.0, 1.0, size=(cfg.action_dim,))

    logging.info(f"Loading policy from {path}")
    return lambda obs: np.random.uniform(-1.0, 1.0, size=(cfg.action_dim,))


def main():
    env_id = "rtgym:real-time-gym-v0"
    logging.info(f"Initializing environment: {env_id}")

    try:
        env = gym.make(env_id)
    except Exception as e:
        logging.error(f"Failed to initialize environment. Error: {e}")
        return

    policy = load_policy(cfg.agent.policy_path)
    ou_noise = OUNoise(cfg.action_dim)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = Path(cfg.data_output_dir) / f"agent_data_{timestamp}.h5"
    writer = HDF5Writer(filepath=filepath)

    obs = env.reset()
    logging.info("Starting data collection loop. Press Ctrl+C to stop.")

    try:
        while True:
            action = policy(obs)

            if cfg.agent.use_noise:
                noise_val = ou_noise.noise() * cfg.agent.noise_scale
                action = action + noise_val
                action = np.clip(action, -1.0, 1.0)

            next_obs, reward, done, info = env.step(action)

            image = obs if isinstance(obs, np.ndarray) else obs[0]

            if image.shape != cfg.image_shape:
                pass

            telemetry = {
                "speed": info.get("speed", 0.0),
                "gear": info.get("gear", 0),
                "rpm": info.get("rpm", 0.0),
            }

            writer.append(image, telemetry, action)

            obs = next_obs

            if done:
                obs = env.reset()
                ou_noise.reset()

    except KeyboardInterrupt:
        logging.info("Data collection stopped by user.")
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
