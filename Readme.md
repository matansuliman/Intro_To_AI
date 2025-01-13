# Matan Suliman - MDP Value and Policy Iteration

This repository contains an implementation of Markov Decision Process (MDP) including the Value Iteration and Policy Iteration algorithms. The code allows for the definition of states, actions, and a transition model within a grid, and performs Value Iteration or Policy Iteration to compute the optimal policies for an agent.

## Table of Contents

1. [Overview](#overview)
2. [Classes](#classes)
   - [Action](#action)
   - [State](#state)
   - [Grid](#grid)
   - [MDP](#mdp)
3. [Key Functions](#key-functions)
   - [Value Iteration](#value-iteration)
   - [Policy Iteration](#policy-iteration)
   - [Transition Model](#transition-model)
4. [Helper Functions](#helper-functions)
5. [Usage](#usage)
6. [Plots](#plots)
7. [License](#license)

## Overview

This repository implements a Markov Decision Process (MDP) model to simulate decision-making on a grid-based environment. The agent in the environment can move in different directions based on the available actions and states, which are either walls, free spaces, or goal states.

### Key Components

- **States**: Each position in the grid can either be a free space, a wall, or a goal.
- **Actions**: The agent can move in predefined directions like Up, Down, Left, Right (or Diagonal movements in a different version).
- **Utility & Reward**: Each state has associated utility values and rewards, which are updated based on the agent's actions.

## Classes

### Action

Represents an action taken by the agent. An action consists of a row and column delta and a name.

#### Methods:
- `getDeltaRow()`: Returns the change in the row index for the action.
- `getDeltaCol()`: Returns the change in the column index for the action.
- `getName()`: Returns the name of the action.
- `getSymbol()`: Returns a symbol representing the action (e.g., `^`, `v`, `<`, `>`, etc.).
- `__str__()`: Returns a string representation of the action.

### State

Represents a state in the MDP grid.

#### Methods:
- `getPos()`: Returns the position (row, col) of the state.
- `setPos()`: Sets the position of the state.
- `getStatus()`: Returns the status of the state (wall, free, goal, or invalid).
- `setStatus()`: Sets the status of the state.
- `getReward()`: Returns the reward of the state.
- `setReward()`: Sets the reward of the state.
- `getUtility()`: Returns the utility of the state.
- `setUtility()`: Sets the utility of the state.
- `getActions()`: Returns the list of actions available from this state.
- `setActions()`: Sets the list of actions available from this state.
- `is_free()`: Checks if the state is free.
- `is_wall()`: Checks if the state is a wall.
- `is_goal()`: Checks if the state is a goal.

### Grid

Represents the grid of states.

#### Methods:
- `getRows()`: Returns the number of rows in the grid.
- `getCols()`: Returns the number of columns in the grid.
- `getData()`: Returns the grid data as a 2D list.
- `setState()`: Sets a state at a specific position.
- `getState()`: Gets a state at a specific position.
- `print()`: Prints the grid's information based on the specified type (status, reward, utility, etc.).

### MDP

The main class representing the Markov Decision Process (MDP).

#### Methods:
- `getGrid()`: Returns the grid object.
- `setGrid()`: Sets the grid.
- `getStates()`: Returns all states in the grid.
- `Actions()`: Returns the list of available actions for a given state.
- `update_utilitys()`: Updates the utilities of all states.
- `update_actions()`: Updates the actions of all states.
- `init_utilities()`: Initializes the utilities of all states.
- `init_actions()`: Initializes the actions of all states.
- `Q_value()`: Calculates the Q-value for a given state and action.

## Key Functions

### Value Iteration

The `Value_Iteration` function performs the value iteration algorithm to compute the optimal utility values for all states. It iterates until the utilities converge or the maximum number of iterations is reached.

```python
def Value_Iteration(mdp: MDP, EPSILON: float, max_iteration: int = 200) -> int:
```


### Policy Iteration

The `Policy_Iteration` function performs the policy iteration algorithm, which consists of policy evaluation and policy improvement steps.converge or the maximum number of iterations is reached.

```python
def Policy_Iteration(mdp: MDP, EPSILON: float, max_iteration: int = 200, variation: int =1) -> int:
```

### Transition Model

The `transition_model` function defines the probability distribution for the next state given the current state and action. It incorporates the effects of movement and possible deviations in the agent's actions.

```python
def transition_model(mdp: MDP, state_curr: State, action: Action) -> Iterable[Tuple[float, State]]:
```


## Helper Functions

-   _precision(): Ensures numerical precision using Decimal.
-   _zeros(): Initializes a grid of zeros for utilities.


## Usage
You can use this code to simulate an agent navigating through a grid environment and solve MDPs using Value Iteration or Policy Iteration.

-   Define the grid with states and rewards.
-   Set the transition model (probabilities of moving to neighboring states).
-   Run Value Iteration or Policy Iteration to compute optimal policies.

```python
# Example usage
grid = Grid(5, 5)
mdp = MDP(grid, transition_model, discount_factor=0.9, p=0.8)
Value_Iteration(mdp, EPSILON=0.01)
```


## Plots

The `plt_save` function generates and saves a plot of the utility values and the optimal policy derived from Value Iteration or Policy Iteration.

```python
def plt_save(mdp: MDP, i: int, Q: int):
```