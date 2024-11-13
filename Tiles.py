"""
Name: Matan suliman
ID: 322982620
"""

import sys # args
from collections import deque # queue for bfs
import heapq # priority queue
from typing import List # clearity of code

## Classes ##

class PriorityQueue:
    def __init__(self,
                 data: list =[],
                 key= lambda x:x
                 ):
        self.key = key
        self.data = [(key(item), i, item) for i, item in enumerate(data)]
        self.index = len(self.data)
        heapq.heapify(self.data)

    def push(self, item):
        heapq.heappush(self.data, (self.key(item), self.index, item))
        self.index += 1
    
    def pop(self):
        return heapq.heappop(self.data)[-1]

class Node:
    def __init__(self, 
                 state: List[int],
                 parent: 'Node',
                 action,
                 path_cost: int,
                 depth: int
                 ):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = depth
    
    def pathToRoot(self):
        res = []
        pos = self
        while pos.action != None:
            res.append(pos.parent.state[pos.state.index(0)])
            pos = pos.parent
        return res[::-1]

    def isCycle(self, num :int =float('inf')):
        pos = self.parent
        while pos != None and num != 0:
            if pos == Node: return True
            pos = pos.parent
            num -= 1
        return False

class Problem:
    def __init__(self, 
                 initial_state: List[int], 
                 goal_states: List[List[int]],
                 action_cost: int,
                 actions,
                 transition_model,
                 ):
        self.initial_state = initial_state
        self.goal_states = goal_states
        self.action_cost = action_cost
        self.actions = actions
        self.transition_model = transition_model
    
    def isGoal(self, state: List[int]):
        return True if state in self.goal_states else False

    def Expand(self, node: Node):
        """
        yields the node childs according to the given problem transition model
        """
        for action in self.actions:
            child_state = self.transition_model(node.state, action)
            if child_state:
                yield Node(
                    state= child_state,
                    parent= node,
                    action= action,
                    path_cost= node.path_cost +self.action_cost,
                    depth= node.depth +1
                )

## Helper functions to 8puzzle ##

def swap(state: List[int], i: int, j: int):
    """
    swaps the values in state list at indexs i and j
    """
    res_state = state.copy()
    # if i and j in bounds
    if 0 <= i < len(res_state) and 0 <= j < len(res_state):
        res_state[i], res_state[j] = res_state[j], res_state[i]
        return res_state
    return None

def up(state: List[int], i: int):
    """
    apply up action on the tile in index i
    """
    return swap(state, i, i -3)

def down(state: List[int], i: int):
    """
    apply down action on the tile in index i
    """
    return swap(state, i, i +3)

def left(state: List[int], i: int):
    """
    apply left action on the tile in index i
    """
    return swap(state, i, i -1)

def right(state: List[int], i: int):
    """
    apply right action on the tile in index i
    """
    return swap(state, i, i +1)

def transition_model_8puzzle(state: List[int], action):
    """
    the transition model of the 8 puzzle problem
    returns the final state after applying the action on the start state
    restraints:
        first row (idx <= 3) can't prefrom up
        last row (5 <= idx) can't prefrom up
        first col (idx %3 == 0) can't prefrom left
        last col (idx %3 == 2) can't prefrom right
    """
    zero_idx = state.index(0)
    if ((action == up and 3 <= zero_idx) or
        (action == down and zero_idx <= 5) or
        (action == left and zero_idx %3 != 0) or
        (action == right and zero_idx %3 != 2)
    ):
        return action(state, zero_idx)
    else:
        return None

## Algorithms and Helper algorithms ##

def Breadth_First_Search(problem: Problem):
    """
    according to pseudocode at figure 3.9 on page 95 in the book
    can be implemented according to page 94:
        def f(node: Node):
            return node.depth

        return (problem= problem, f= f)
    """
    node = Node(
        state= problem.initial_state,
        parent= None,
        action= None,
        path_cost= 0,
        depth= 0
    )
    if problem.isGoal(node.state): return node
    forntier = deque([node])
    reached = set(tuple(problem.initial_state))
    count = 0
    while forntier:
        node = forntier.pop()
        count += 1
        for child in problem.Expand(node):
            if problem.isGoal(child.state): return child, count
            s = tuple(child.state) # a tuple is hashable
            if s not in reached:
                reached.add(s)
                forntier.appendleft(child)
    return None, count

def Depth_Limited_Search(problem: Problem, l: int):
    """
    according to pseudocode at figure 3.12 on page 99 in the book
    can be implemented according to page 96:
        def f(node: Node):
            return -node.depth # the negative of the depth

        return (problem= problem, f= f)
    """
    forntier = [
        Node(
            state= problem.initial_state,
            parent= None,
            action= None,
            path_cost= 0,
            depth= 0
        )] # stack
    result = False
    count = 0
    while forntier:
        node = forntier.pop()
        count += 1
        if problem.isGoal(node.state): return node, count
        if node.depth > l: result = 'cutoff'
        else:
            for child in problem.Expand(node):
                s = tuple(child.state) # a tuple is hashable
                if not node.isCycle(0): # not a necessety
                    forntier.append(child)
    return result, count

def Iterative_Deepening_Search(problem: Problem):
    """
    according to pseudocode at figure 3.12 on page 99 in the book
    """
    depth = 0
    while True:
        result = Depth_Limited_Search(problem, depth)
        if result[0] != 'cutoff': return result
        else: depth += 1

def Best_First_Search(problem: Problem, f):
    """
    according to pseudocode at figure 3.12 on page 99 in the book
    """
    node = Node(
        state= problem.initial_state,
        parent= None,
        action= None,
        path_cost= 0,
        depth= 0
    )
    forntier = PriorityQueue( # priority queue implemented with heapq (min heap)
        data= [node],
        key= f
    )
    reached = {tuple(problem.initial_state) : node}
    count = 0
    while forntier:
        node = forntier.pop()
        count += 1
        if problem.isGoal(node.state): return node, count
        for child in problem.Expand(node):
            s = tuple(child.state) # a tuple is hashable
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                forntier.push(child)
    return None, count

def LinearConflicts(node):
    """
    returns the 'Linear Conglicts Distance' of the state from the goal state
    this herusistic uses the manhattan distance
    """
    def ManhattanDistance(state):
        sum = 0
        for idx, tile in enumerate(state):
            # transformation from 1D indexs to 2D index :(1,9) -> (3,3)
            x_curr, y_curr = idx //3, idx %3
            x_goal, y_goal = tile //3, tile %3
            sum += abs(x_goal -x_curr) + abs(y_goal -y_curr)
        return sum

    def RowsConflicts(state):
        sum = 0
        row_len = len(state) //3
        for row_number in range(row_len):
            curr_row = state[row_number *row_len:(row_number +1) *row_len]
            # for each pair of tiles in row
            for col_i, tile_i in enumerate(curr_row):
                for tile_j in curr_row[col_i:]:
                    # both tiles are in their goal row
                    if (tile_i != 0 and tile_j != 0 and row_number == (tile_i //3) and row_number == (tile_j //3)):
                        # tiles in conflicts
                        if (tile_i %3) > (tile_j %3):
                            sum += 2
        return sum

    def ColsConflicts(state):
        sum = 0
        col_len = len(state) //3
        for col_number in range(col_len):
            curr_col = state[col_number::col_len]
            # for each pair of tiles in col
            for row_i, tile_i in enumerate(curr_col):
                for tile_j in curr_col[row_i:]:
                    # both tiles are in their goal col
                    if (tile_i != 0 and tile_j != 0 and col_number == (tile_i %3) and col_number == (tile_j %3)):
                        # tiles in conflicts
                        if (tile_i //3) > (tile_j //3):
                            sum += 2
        return sum

    return ManhattanDistance(node.state) + RowsConflicts(node.state) + ColsConflicts(node.state)

def Greedy_Best_First_Search(problem: Problem):
    """
    according to section 3.5.1 in the book
    """
    return Best_First_Search(problem= problem, f= LinearConflicts)

def A_star(problem: Problem):
    """
    according to section 3.5.2 in the book
    """

    def f(node: Node):
        """
        estimated cost of the best path that continues from node to a goal
        """
        return node.path_cost + LinearConflicts(node)

    return Best_First_Search(problem= problem, f= f)

## Driver code ##

def main():
    # init problem
    puzzle8 = Problem(
        initial_state= [int(val) for val in sys.argv[1:]],
        goal_states= [[0, 1, 2, 3, 4, 5, 6, 7, 8]],
        action_cost= 1,
        actions= [up, down, left, right],
        transition_model= transition_model_8puzzle
    )
    # init algorithms and their names
    algorithms = (
        ('BFS', Breadth_First_Search),
        ('IDDFS', Iterative_Deepening_Search),
        ('GBFS', Greedy_Best_First_Search),
        ('A*', A_star)
    )
    # run algorithms
    for name, algorithm_func in algorithms:
        target_node, count = algorithm_func(problem= puzzle8)
        print(f'{name:8} {count}', end='  ')
        if isinstance(target_node, Node): print(target_node.pathToRoot())
        else: print('not found')

if __name__ == '__main__':
    main()