from __future__ import annotations
import gym
import numpy as np

from reinforcement_learning_example.agent import Agent, exponential_decay, DeepQNetwork
from reinforcement_learning_example.utils import plot_learning_curve, run_agent_in_env

def main():
    """An example training loop."""

    env_factory = lambda: gym.make("LunarLander-v2")

    # Create an environment to get its caracteristics.
    env = env_factory()
    num_actions = env.action_space.n
    observation_space_shape = env.observation_space.shape
    env.close()

    gamma = 0.99
    input_dims = observation_space_shape
    batch_size = 64
    update_target_every_n_steps = 10_000

    epsilon_update_function = exponential_decay

    network_kwargs = {
        "learning_rate": 0.001,
        "input_dims": input_dims,
        "num_actions": num_actions,
        "fc1_dims": 256,
        "fc2_dims": 256,
    }

    epsilon_update_kwargs = {"start": 1, "end": 0.05, "decay": 2000}

    agent = Agent(
        network=DeepQNetwork,
        network_kwargs=network_kwargs,
        gamma=gamma,
        input_dims=input_dims,
        num_actions=num_actions,
        update_target_every_n_steps=update_target_every_n_steps,
        batch_size=batch_size,
        epsilon_update_function=epsilon_update_function,
        epsilon_update_kwargs=epsilon_update_kwargs,
    )

    trainer = Trainer(agent=agent, env_factory=env_factory)

    trainer.train(num_episodes=500)

    run_agent_in_env(env_factory=env_factory, agent=agent, num_episodes_to_show=10)

    return agent


class Trainer:
    def __init__(
        self,
        agent: Agent,
        env_factory: Callable[[Any], gym.Env],
    ):

        self.agent = agent
        self.env_factory = env_factory

        self.scores = []
        self.epsilon_history = []
        self.episode_num = 0

    def train(self, num_episodes: int = 500, average_score_window: int = 100):
        try:
            env = self.env_factory()
            for i in range(num_episodes):
                score = 0
                done = False
                observation = env.reset()
                num_steps = 0
                while not done:
                    num_steps += 1
                    action = self.agent.choose_action(observation)
                    observation_, reward, done, info = env.step(action)
                    score += reward
                    self.agent.store_transition(observation, action, reward, observation_, done)
                    self.agent.learn()
                    observation = observation_
                self.scores.append(score)
                self.epsilon_history.append(self.agent.epsilon)

                average_score = np.mean(self.scores[-average_score_window:])
                self.episode_num += 1

                print(
                    f"episode_num {self.episode_num:>5} / episode {i:>5} / score {score:>8.2f} / "
                    f"average score {average_score:>8.2f} / epsilon {self.agent.epsilon:>3.2f} / "
                    f"num_steps {num_steps:>5}"
                )

        except KeyboardInterrupt:
            """This is used so that you can interrupt your training (e.g in a notebook) and restart it from
            exactly the same state without needing to retrain. It's possible to inspect the agent after stopping which
            can be very useful for debugging."""
            pass

        x = [i_ + 1 for i_ in range(len(self.scores))]
        plot_learning_curve(x, self.scores, self.epsilon_history)


if __name__ == "__main__":
    main()
