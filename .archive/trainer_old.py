import gym
import numpy as np

from reinforcement_learning_example.agent import Agent
from reinforcement_learning_example.utils import plot_learning_curve


def main():
    trainer = Trainer()
    trainer.train()


class Trainer:
    def __init__(
        self,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        batch_size: int = 64,
        num_actions: int = 4,
        epsilon_min: float = 0.01,
        epsilon_dec: float = 5e-5,
        input_dims: list = [8],
        learning_rate: float = 0.001,
    ):

        self.env = gym.make("LunarLander-v2")

        self.agent = Agent(
            gamma=gamma,
            epsilon=epsilon,
            batch_size=batch_size,
            num_actions=num_actions,
            epsilon_min=epsilon_min,
            epsilon_dec=epsilon_dec,
            input_dims=input_dims,
            learning_rate=learning_rate,
        )
        self.scores, self.epsilon_history = [], []
        self.filename = "lunar_lander.png"
        self.episode_num = 0

    def train(self, num_episodes: int = 500, average_score_window: int = 100):
        try:
            for i in range(num_episodes):
                score = 0
                done = False
                observation = self.env.reset()
                while not done:
                    action = self.agent.choose_action(observation)
                    observation_, reward, done, info = self.env.step(action)
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
                    f"average score {average_score:>8.2f} / epsilon {self.agent.epsilon:>3.2f}"
                )

        except KeyboardInterrupt:
            pass

        x = [i_ + 1 for i_ in range(len(self.scores))]

        plot_learning_curve(x, self.scores, self.epsilon_history, self.filename)


if __name__ == "__main__":
    main()
