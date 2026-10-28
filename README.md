# Solving a Markov Decision Process using Policy Iteration

## Aim

To implement the Policy Iteration algorithm for solving a finite Markov Decision Process using the Gymnasium FrozenLake-v1 environment, by repeatedly performing policy evaluation and policy improvement to obtain the optimal value function and optimal policy.

---

## Problem Statement

In this experiment, the `FrozenLake-v1` environment is solved using the **Policy Iteration** algorithm.

The agent starts from the start state and must reach the goal state without falling into holes. The environment is represented as a finite Markov Decision Process. Policy Iteration is used to repeatedly evaluate the current policy and improve it until the policy becomes stable.

The objective is to find:

1. The optimal state-value function $V^*(s)$
2. The optimal policy $pi^*(s)$

---

## Software Requirements

```bash
pip install gymnasium numpy
```

---

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-world environment where the agent moves over frozen tiles and tries to reach the goal without falling into holes.

For the default 4 × 4 FrozenLake map:

| Component | Description |
|---|---|
| Environment | `FrozenLake-v1` |
| Map size | 4 × 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and hole states |

---

## Theory

Policy Iteration is a Dynamic Programming method used to find the optimal policy of a Markov Decision Process.

It consists of two major steps:

1. **Policy Evaluation**
2. **Policy Improvement**

These two steps are repeated until the policy becomes stable.

---

## Policy Evaluation

Policy evaluation estimates the value function for the current policy.

The Bellman expectation equation is:

$$
V^\pi(s) =
\sum_a \pi(a \mid s)
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $pi(a \mid s)$ | Probability of taking action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $pi$ |

---

## Policy Improvement

Policy improvement updates the policy greedily with respect to the current value function.

The improved policy is obtained as:

$$
\pi'(s) =
\arg\max_a
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

If the improved policy is the same as the old policy, the policy is considered stable.

---

## Algorithm

1. Create the Gymnasium `FrozenLake-v1` environment.
2. Initialize a random policy.
3. Repeat until the policy becomes stable:
   - Evaluate the current policy using iterative policy evaluation.
   - Improve the policy greedily using the current value function.
   - Compare the old policy and the new policy.
4. Stop when the policy does not change.
5. Display the optimal value function and optimal policy.

---

## Python Program

```python

import gymnasium as gym
import numpy as np

# Create FrozenLake Environment
env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=True
)

env = env.unwrapped

n_states = env.observation_space.n
n_actions = env.action_space.n

gamma = 0.99
theta = 1e-8

print("Number of States:", n_states)
print("Number of Actions:", n_actions)
print("Discount Factor:", gamma)

def policy_evaluation(pi, P, gamma, theta=1e-8):
    V = np.zeros(len(P))
    iteration = 0

    while True:
        prev_V = V.copy()
        iteration += 1

        for s in range(len(P)):
            V[s] = 0.0

            for a, action_prob in enumerate(pi[s]):
                for prob, next_state, reward, done in P[s][a]:
                    V[s] += action_prob * prob * (
                        reward
                        + gamma * prev_V[next_state] * (not done)
                    )

        # Check convergence
        if np.max(np.abs(V - prev_V)) < theta:
            break

    return V, iteration


# Initialize a random policy
pi = np.ones((n_states, n_actions)) / n_actions

# Evaluate the policy
V, iterations = policy_evaluation(
    pi, env.P, gamma, theta
)

print("Value Function After Policy Evaluation:")
print(np.round(V.reshape(4, 4), 4))
print("Iterations:", iterations)

def print_policy(policy):
    action_symbols = {
        0: "←",
        1: "↓",
        2: "→",
        3: "↑"
    }

    best_actions = np.argmax(policy, axis=1)

    policy_grid = np.array(
        [action_symbols[action] for action in best_actions]
    ).reshape(4, 4)

    print(policy_grid)


def policy_improvement(V, P, gamma):
    n_states = len(P)
    n_actions = len(P[0])

    # Initialize policy
    pi = np.zeros((n_states, n_actions))

    for s in range(n_states):
        action_values = np.zeros(n_actions)

        for a in range(n_actions):
            for prob, next_state, reward, done in P[s][a]:
                action_values[a] += prob * (
                    reward
                    + gamma * V[next_state] * (not done)
                )

        # Choose the best action
        best_action = np.argmax(action_values)
        pi[s, best_action] = 1.0

    return pi


# Policy Improvement Output
improved_policy = policy_improvement(V, env.P, gamma)

print("\nImproved Policy:")
print_policy(improved_policy)

# Evaluate the improved policy
V_improved, iterations = policy_evaluation(
    improved_policy, env.P, gamma, theta
)

print("\nValue Function After Policy Improvement:")
print(np.round(V_improved.reshape(4, 4), 4))
def policy_iteration(P, gamma, theta=1e-8):
    n_states = len(P)
    n_actions = len(P[0])

    # Initialize a random policy
    pi = np.ones((n_states, n_actions)) / n_actions

    while True:

        # Step 1: Policy Evaluation
        V, iterations = policy_evaluation(
            pi, P, gamma, theta
        )

        # Step 2: Policy Improvement
        new_pi = policy_improvement(V, P, gamma)

        # Check if policy is stable
        if np.array_equal(pi, new_pi):
            break

        pi = new_pi

    return pi, V
def print_value_function(V):
    print("\nOptimal State-Value Function:")
    print(np.round(V.reshape(4, 4), 4))


def print_policy(policy):
    action_symbols = {
        0: "←",
        1: "↓",
        2: "→",
        3: "↑"
    }

    best_actions = np.argmax(policy, axis=1)

    policy_grid = np.array(
        [action_symbols[action] for action in best_actions]
    ).reshape(4, 4)

    print("\nOptimal Policy:")
    print(policy_grid)
# Run Policy Iteration
optimal_policy, optimal_value_function = policy_iteration(
    env.P,
    gamma=gamma,
    theta=theta
)

print("Name:Rakisha R ")
print("Register Number:212225230223 ")

print_value_function(optimal_value_function)
print_policy(optimal_policy)

env.close()

```

## Output
<img width="431" height="135" alt="image" src="https://github.com/user-attachments/assets/77053621-d224-4e7e-aee5-32f89119b972" />

<img width="610" height="232" alt="image" src="https://github.com/user-attachments/assets/10a6f8e3-25d8-466f-9e70-23a73a09c4f2" />


<img width="637" height="451" alt="image" src="https://github.com/user-attachments/assets/8a433510-88fb-4484-918a-762adb1c26bb" />

<img width="561" height="492" alt="image" src="https://github.com/user-attachments/assets/4736dc21-d15b-4075-ae5a-2c9796bdd18c" />

## Result
Thus, the Policy Iteration algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The algorithm performed policy evaluation and policy improvement repeatedly until the policy became stable. The optimal state-value function and optimal policy were obtained and displayed in a 4 × 4 grid.

## Inference
Policy evaluation calculates the value of each state under the current policy.
Policy improvement selects the best action based on the estimated state values.
The algorithm repeats these steps until the policy becomes stable.
The optimal policy guides the agent toward the goal while avoiding holes as much as possible.
