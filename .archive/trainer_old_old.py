import gym
from reinforcement_learning_example.agent_old_old import Agent
from reinforcement_learning_example.utils import plot_learning_curve
import numpy as np


def main():
    env = gym.make("LunarLander-v2")
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[8], lr=0.001)
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        num_steps = 0
        while not done:
            num_steps += 1
            modulo = 1000
            if num_steps % modulo == modulo - 1:
                print(f"Step {num_steps} of episode")

            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print("episode ", i, "score %.2f" % score, "average score %.2f" % avg_score, "epsilon %.2f" % agent.epsilon)
    x = [i + 1 for i in range(n_games)]
    filename = "lunar_lander.png"
    plot_learning_curve(x, scores, eps_history, filename)


if __name__ == "__main__":
    main()
