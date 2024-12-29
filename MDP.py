from copy import copy, deepcopy
from typing import * # clearity of code
from decimal import Decimal # precision


PRECISION = 8

class Action:
    def __init__(self,
                 delta_row: int,
                 delta_col: int,
                 name: str
                 ):
        
        self._delta_row = delta_row
        self._delta_col = delta_col
        self._name = name

    def getDeltaRow(self) -> int:
        return self._delta_row
    
    def getDeltaCol(self) -> int:
        return self._delta_col
    
    def getName(self) -> str:
        return self._name
    
    def __str__(self) -> str:
        return f"Action: {self._name}, Deltas: {(self._delta_row, self._delta_col)}"

class State:
    def __init__(self, pos: Tuple[int], status: int =0, reward:float =0, utility: float =0, action: Action =Action(-1, -1, 'None')):
        """
        status: 0 = wall, 1 = free, -1 = goal
        """
        self._pos = pos
        self._status = status
        self._reward = reward
        self._utility = utility
        self._action = [action]
    
    def getPos(self) -> Tuple[int]:
        return self._pos
    
    def setPos(self, pos: Tuple[int]):
        self._pos = pos

    def getRow(self) -> int:
        return self._pos[0]
    
    def setRow(self, row: int):
        self._pos = (row, self._pos[1])
    
    def getCol(self) -> int:
        return self._pos[1]
    
    def setCol(self, col: int):
        self._pos = (self._pos[0], col)

    def getStatus(self) -> int:
        return self._status
    
    def setStatus(self, status: int):
        self._status = status
    
    def getReward(self) -> float:
        return self._reward
    
    def setReward(self, reward: float):
        self._status = reward
    
    def getUtility(self) -> float:
        return self._utility
    
    def setUtility(self, utility: float):
        self._utility = utility
    
    def getActions(self) -> Action:
        return self._action
    
    def setActions(self, actions: List[Action]):
        self._action = actions

    def is_free(self) -> bool:
        return self._status == 1

    def is_wall(self) -> bool:
        return self._status == 0
    
    def is_goal(self) -> bool:
        return self._status == -1

    def __str__(self) -> str:
        return f"Pos: {self._pos}, Status: {self._status}, Reward: {self._reward}, Utility: {self._utility}, {self._action}"

class Grid:
    def __init__(self, rows: int, cols: int):
        self._rows = rows
        self._cols = cols
        self._data = [[State((i, j)) for j in range(cols)] for i in range(rows)]
    
    def getRows(self) -> int:
        return self._rows

    def getCols(self) -> int:
        return self._cols

    def getData(self) -> List[List[State]]:
        return self._data

    def setData(self, data: List[Any]):
        self._data = copy(data)
        self._rows = len(data)
        self._cols = len(data[0])

    def getState(self, pos: Tuple[int]) -> State:
        row, col = pos
        rows, cols = self.getRows(), self.getCols()

        if 0 <= row < rows and 0 <= col < cols:
            return self._data[row][col]
        else: 
            return None

    def setState(self, pos: Tuple[int], status: int =0, reward:float =0, utility: float =0, action: Action =Action(-1, -1, 'None')):
        row, col = pos
        self._data[row][col] = State(pos, status, reward, utility, action)
    
    def __str__(self):
        res = ''
        for row in self._data:
            for cell in row:
                res += f'{cell}  ##  '
            res += f'\n'
        return res
    
    def getCopy(self) -> 'Grid':
        return deepcopy(self)
    
    def print(self, type: str):

        def actions_names(state: State):
            res = ''
            for action in state.getActions():
                res += action.getName() + ', '
            return f'({res})'

        func = None
        if type == 'status': func = lambda state: state.getStatus()
        elif type == 'reward': func = lambda state: state.getReward()
        elif type == 'utility': func = lambda state: round(state.getUtility(), 4)
        elif type == 'action': func = actions_names
        elif type == 'pos': func = lambda state: state.getPos()
        else: raise ValueError("Invalid value")

        res = ''
        for row in self._data:
            for cell in row:
                res += f'{func(cell)} '
            res += f'\n'
        return res

class MDP:
    def __init__(self,
                 grid: Grid,
                 transition_model: Callable,
                 discount_factor: float
                 ):
        self._grid = grid
        self._transition_model = transition_model
        self._discount_factor = discount_factor
    
    def getGrid(self) -> Grid:
        return self._grid
    
    def getStates(self) -> List[State]:
        mat = self._grid.getData()
        res = []
        for row in mat:
            for cell in row:
                res.append(cell)
        return res
    
    def getState(self, pos: Tuple[int]) -> State:
        return self._grid.getState(pos)
    
    def TransiotionModel(self, state_curr: State, action: Action) -> Callable:
        return self._transition_model(self, state_curr, action)

    def getDiscountFactor(self) -> float:
        return self._discount_factor

    def Actions(self, state: State) -> List[Action]:
        rows, cols = self.getGrid().getRows(), self.getGrid().getCols()
        res = []

        # if the state is a wall or a goal state than it has no actions
        if state.getStatus() != 1:
            return res
        # check all 4 directions (up, right, down, left)
        for action in actions.values():
            # as in the book
            res.append(action)
            """
            # as in the maman
            next_state = nextState(self, state, action)
            if next_state == None:
                continue
            next_row, next_col = next_state.getPos()
            bounds = 0 <= next_row < rows and 0 <= next_col < cols
            if bounds and next_state.getStatus() != 0:
                res.append(action)
            """
        return res
    

def nextState(mdp: MDP, state: State, action: Action) -> State:
    row, col = state.getPos()
    delta_row, delta_col = action.getDeltaRow(), action.getDeltaCol()
    next_row, next_col = row + delta_row, col + delta_col
    # as in the book
    next_state = mdp.getState((next_row, next_col))
    if next_state == None or next_state.is_wall():
        return state
    else: return next_state
    """
    # as in the maman
    return mdp.getState((next_row, next_col))
    """

actions = {
    'Up': Action(
        delta_row= -1,
        delta_col= 0,
        name= 'Up'
    ),
    'Right': Action(
        delta_row= 0,
        delta_col= +1,
        name= 'Right'
    ),
    'Down': Action(
        delta_row= +1,
        delta_col= 0,
        name= 'Down'
    ),
    'Left': Action(
        delta_row= 0,
        delta_col= -1,
        name= 'Left'
    )
}
actions_diagonals = {
    'Up-Left': Action(
        delta_row= -1,
        delta_col = -1,
        name= 'Up-Left'
    ),
    'Up-Right': Action(
        delta_row= -1,
        delta_col = +1,
        name= 'Up-Right'
    ),
    'Down-Left': Action(
        delta_row= +1,
        delta_col = -1,
        name= 'Down-Left'
    ),
    'Down-Right': Action(
        delta_row= +1,
        delta_col = +1,
        name= 'Down-Right'
    )
}

def Value_Iteration(mdp: MDP, epsilon: float):
    
    def is_active(state: State) -> bool:
        return state.getStatus() == 1
    
    def init_U_next(): 
        rows, cols = mdp.getGrid().getRows(), mdp.getGrid().getCols()
        return [[0 for _ in range(cols)] for _ in range(rows)]
    
    def update_utilitys(U: List[List[float]]):
        for state in mdp.getStates():
            row, col = state.getPos()
            state.setUtility(U[row][col])

    U_next_value = init_U_next()
    delta = pow(10, -PRECISION -1) # error
    gamma = mdp.getDiscountFactor()
    i = 0
    while True:
        print(f'iteration {i} , delta: {delta}')
        i += 1
        update_utilitys(U_next_value)
        delta = pow(10, -PRECISION -1) # error
        """print(mdp.getGrid().print('action'))
        print(mdp.getGrid().print('utility'))"""
        # loop on states
        for state in mdp.getStates():
            if not is_active(state): continue

            # find the action that produces the maximum Q_value
            maxx_value = float('-inf')
            maxx_actions = []
            for action in mdp.Actions(state):
                q_val = Q_value(mdp, state, action)
                if q_val > maxx_value:
                    maxx_value = q_val
                    maxx_actions = [action]
                elif q_val == maxx_value:
                    maxx_actions.append(action)
                else:
                    pass

            # and store it in U_next, update the actions
            row, col = state.getPos()
            U_next_value[row][col] = maxx_value
            state.setActions(maxx_actions)
            
            # update delta to be the largest error
            error = abs(maxx_value - state.getUtility())
            delta = round(max(delta, error), PRECISION)


        if delta <= epsilon *(1 -gamma) /gamma or i >= 200:
            break

    print(mdp.getGrid().print('action'))
    print(mdp.getGrid().print('utility'))
    return U_next_value

def Q_value(mdp: MDP, state: State, action: Action) -> float:
    gamma = Decimal(str(mdp.getDiscountFactor()))
    summ = Decimal(str(0))
    for inter_probability, inter_state_next in mdp.TransiotionModel(state, action):
        prob = Decimal(str(inter_probability))
        reward = Decimal(str(inter_state_next.getReward()))
        utility = Decimal(str(inter_state_next.getUtility()))
        intermediate = prob * (reward + gamma * utility)
        print(f'    += {prob} * ({reward} + {gamma} * {utility})')# = {prob} * ({reward} + {gamma * utility}) = {prob} * {reward + gamma * utility} = {intermediate}')
        print(f'    += {intermediate}')
        summ += intermediate  # Accumulate the sum
    print(f'from {state.getPos()} doing {action.getName()} got {summ}')
    return summ  # Return as float

def transition_model(mdp: MDP, state_curr: State, action: Action) -> Iterable[Tuple[float, State]]:

    actions_valid = [action]
    probabilities = [0.8, 0.1, 0.1]
    states_next = [nextState(mdp, state_curr, action)]

    action_name = action.getName()
    actions_potensial_str = []
    if action_name in ['Up', 'Down']:
        actions_potensial_str = [f'Right', f'Left']
    elif action_name in ['Right', 'Left']:
        actions_potensial_str = [f'Up', f'Down']
    else: 
        pass

    for action_potensial_str in actions_potensial_str:
        state_next = nextState(mdp, state_curr, actions[action_potensial_str])
        # as in the book
        states_next.append(state_next)
        """
        # as in the maman
        if state_next != None and not state_next.is_wall():
            actions_valid.append(actions[action_potensial_str])
            states_next.append(state_next)
        else:
            probabilities[0] += probabilities.pop()
            """

    """
    actions_potensial_diag = []
    if action_name in ['Up', 'Down']:
        actions_potensial_diag = [f'{action_name}-Right', f'{action_name}-Left']
    elif action_name in ['Right', 'Left']:
        actions_potensial_diag = [f'Up-{action_name}', f'Down-{action_name}']
    else: 
        pass
    
    for action_potensial_diag in actions_potensial_diag:
        state_next = nextState(mdp, state_curr, actions_diagonals[action_potensial_diag])
        if state_next != None and state_next.getStatus() != 0:
            actions_valid.append(actions_diagonals[action_potensial_diag])
        else:
            probabilities[0] += probabilities.pop()
    """
            
    return zip(probabilities, states_next)

from fractions import Fraction
## Driver code ##

def main():
    rows, cols = 3, 4

    def init_grid():
        r = -0.04
        states_rewards = [
            [r, r, r, 1],
            [r, 0, r, -1],
            [r, r, r, r]
        ]
        states_status = [
            [1, 1, 1, -1],
            [1, 0, 1, -1],
            [1, 1, 1, 1]
        ]
        states = Grid(rows, cols)
        for row in range(rows):
            for col in range(cols):
                states.setState((row, col), status= states_status[row][col], reward= states_rewards[row][col])
        return states

    discount_factor = 1
    mdp = MDP(
        grid= init_grid(),
        transition_model= transition_model,
        discount_factor= discount_factor
    )
    
    print(mdp.getGrid().print('pos'))
    print(mdp.getGrid().print('status'))
    print(mdp.getGrid().print('action'))
    print(mdp.getGrid().print('utility'))
    #print(mdp.Actions(0, 1))
    #print(list(mdp.TransiotionModel(states.getState((0, 1)), actions['Left'], None)))
    #print(Q_value(mdp, states.getState((0, 1)), actions['Left'],))
    res = Value_Iteration(mdp, epsilon= 0.001)
    #print(res)
    Q_value(mdp, mdp.getState((2, 2)), actions['Left'])
    Q_value(mdp, mdp.getState((2, 2)), actions['Up'])

if __name__ == '__main__':
    main()
