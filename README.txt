Name: Matan suliman
ID: 322982620

python version: 3.8.0
book: 'Artificial Intelligence, a modern approach, 4'th edition'

Background:
  8 puzzle is a sliding tiles problem
  the goal is to line the numbers in ascending order from left to right up to bottom

1. program overview:

  Classes:
    PriorityQueue (reference 1):
      Attributes:
        - key: the sorting function
        - data: the queue data
        - index: tie breaker of similar values
      Methods:
        - push(item): push item to queue
        - pop(): pop from queue
    
    Node:
      Attributes:
        - state: the state of the node in 8 puzzle
        - parent: the parent node
        - action: the action from parent to me
        - path_cost: the total path cost from the initial state to here
        - depth: the number of actions from the initial state to here
      Methods:
        - pathtoRoot(): returns the path of tile actions from the root to the node
        - isCycle(num): return true if the node creates a cycle (of max length -num) on the path to the root, else returns false

    Problem:
      Attributes:
        - initial_state: the initial state of the 8 puzzle
        - goal_states: the goal states of the 8 puzzle
        - action_cost: the cost of each action
        - actions: the actions available
        - transition_model: a function that takes a state and an action and returns the resulting state
      Methods:
        - isGoal(state): return true if a state is the goal state
        - Expand(node): a node and yields child nodes based on the problem

  Helper functions to 8puzzle:
    swap(state, i, j): swap the values in state at indexes i and j
    up(state, i): swap the values in state at index i and 'above' i if exists
    down(state, i): swap the values in state at index i and 'below' i if exists
    left(state, i): swap the values in state at index i and 'left to' i if exists
    right(state, i): swap the values in state at index i and 'right to' i if exists
    transition_model(state, action): if the action is valid returns the resulting state


  Algorithms:
    - bfs: Breadth_First_Search(problem)
    - iddfs: Iterative_Deepening_Search(problem)
    - GBFS: Greedy_Best_First_Search(problem)
    - A*: A_star(problem)

  Helper algorithms:
    - Depth_Limited_Search(problem, l) healping iddfs
    - Best_First_Search(problem, f) healping GBFS and A*


2. State space overview:
  a state is a 9D vector representing the tile's positions
  the actions are up, down, left and right on the empty tile
  not all actions on the empty tile resulting  in a valid state:
    - if at first row than can't prefrom up
    - if at last row than can't prefrom up
    - if at first col than can't prefrom left
    - if at last col than can't prefrom right

3. Heuristic function:
  LinearConflicts(node) (reference 2): 
   - consistent proof: pdf is attached
   - note: this function input is a node but it works on the state. this is only for comfort of code.

4. Cost optimal:
  - at all inputs, the 'BFS', 'IDDFS' and 'A*' gives the optimal solutions (by depth)
  - 'GBFS' does not always give the optimal solution.
    It struggels on long solutions, because it has more room for failure.

5. Input and Output pictures:
  attached as a png files.

References:
  - (1) - https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes
  - (2) - https://mice.cs.columbia.edu/getTechreport.php?techreportID=1026&format=pdf&