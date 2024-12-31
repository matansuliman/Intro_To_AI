from copy import copy, deepcopy
from typing import * # clearity of code
from decimal import Decimal, getcontext # precision
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import sys

class Action:
    def __init__(self,
                 delta_row: int,
                 delta_col: int,
                 name: str
                 ):
        
        self._delta_row = delta_row
        self._delta_col = delta_col
        self._name = name

    ## get ##
    def getDeltaRow(self) -> int:
        return self._delta_row
    
    def getDeltaCol(self) -> int:
        return self._delta_col
    
    def getName(self) -> str:
        return self._name
    
    ## others ##
    def getSymbol(self): 
        name_to_symbol = {
            'Up': '^',
            'Down': 'v',
            'Left': '<',
            'Right': '>',
            'None': 'X'
        }
        return name_to_symbol.get(self._name, 'No symbol')

    def __str__(self) -> str:
        return f"Action: {self._name}, Deltas: {(self._delta_row, self._delta_col)}"

class State:
    def __init__(self, pos: Tuple[int] =(-9, -9), status: int =-9, reward:float =-9, utility: float =-9, actions: List[Action] =[]):
        """
        status: 0 = wall, 1 = free, -1 = goal, -9 = invalid
        """
        self._pos = pos
        self._status = status
        self._reward = reward
        self._utility = utility
        self._actions = actions
        if status != 1: # not free
            self._actions = []
    
    ## get set ##
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
    
    def getActions(self) -> List[Action]:
        return self._actions
    
    def setActions(self, actions: List[Action]):
        self._actions = actions

    ## others ##
    def printActions(self, type: str) -> str:

        func = None
        if type == 'n': func = lambda action: action.getName()
        elif type == 's': func = lambda action: action.getSymbol()
        else: raise ValueError("Invalid value")

        return ' '.join(list(map(func, self._actions)))

    def is_free(self) -> bool:
        return self._status == 1

    def is_wall(self) -> bool:
        return self._status == 0
    
    def is_goal(self) -> bool:
        return self._status == -1

    def is_invalid(self) -> bool:
        return self._status == -9

    def __str__(self) -> str:
        return f"Pos: {self._pos}, Status: {self._status}, Reward: {self._reward}, Utility: {self._utility}, {self.printActions('n')}"

class Grid:
    def __init__(self, rows: int, cols: int):
        self._rows = rows
        self._cols = cols
        self._data = [[State((i, j)) for j in range(cols)] for i in range(rows)]
    
    ## get set ##
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
        if 0 <= row < self._rows and 0 <= col < self._cols:
            return self._data[row][col]
        else:
            return State((0, 0), status= -9)

    def setState(self, pos: Tuple[int], status: int =0, reward:float =0, utility: float =0, actions: List[Action] =[]):
        row, col = pos
        self._data[row][col] = State(pos, status, reward, utility, actions)
    
    ## others ##

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

        func = None
        if type == 's': func = lambda state: state.getStatus()
        elif type == 'r': func = lambda state: state.getReward()
        elif type == 'u': func = lambda state: round(state.getUtility(), 4)
        elif type == 'a': func = lambda state: state.printActions('s')
        elif type == 'p': func = lambda state: state.getPos()
        else: raise ValueError("Invalid value")

        mat = [[func(cell) for cell in row] for row in self._data]
        df = pd.DataFrame(mat)
        return f'\n{type} grid:\n{df}\n'

    def mats_to_grid(mat_status: List[List[int]], mat_reward: List[List[int]], action_defalut_str: str =None):
        rows, cols = len(mat_status), len(mat_status[0])
        grid = Grid(rows, cols)
        for row in range(rows):
            for col in range(cols):
                if action_defalut_str == None:
                    grid.setState((row, col), status= mat_status[row][col], reward= mat_reward[row][col])
                else:
                    grid.setState((row, col), status= mat_status[row][col], reward= mat_reward[row][col], actions= [actions[action_defalut_str]])

        return grid

class MDP:
    def __init__(self,
                 grid: Grid,
                 transition_model: Callable,
                 discount_factor: float,
                 p: float
                 ):
        self._grid = grid
        self._transition_model = transition_model
        self._discount_factor = discount_factor
        self._p = p
    
    ## get set ##
    def getGrid(self) -> Grid:
        return self._grid
    
    def setGrid(self, grid: Grid) -> None:
        self._grid = grid.getCopy()
    
    def getTransiotionModel(self) -> Callable:
        return self._transition_model
    
    def setTransiotionModel(self, transition_model: callable) -> None:
        self._transition_model = transition_model

    def getDiscountFactor(self) -> float:
        return self._discount_factor

    def setDiscountFactor(self, discount_factor: int) -> None:
        self._discount_factor = discount_factor

    def getP(self):
        return self._p
    
    def setP(self, p: int) -> None:
        self._p = p

    ## others ## 
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
    
    def Actions(self, state: State) -> List[Action]:
        rows, cols = self.getGrid().getRows(), self.getGrid().getCols()
        res = []

        # if the state is a wall or a goal state than it has no actions
        if state.is_invalid():
            raise 'Error, tried to search actions of invalid state'
        if state.is_wall() or state.is_goal():
            return res
        # check all 4 directions (up, right, down, left)
        for action in actions.values():
            next_state = self._nextState(state, action)
            if next_state.is_free() or next_state.is_goal():
                res.append(action)
            """
            # as in the book
            res.append(action)
            """
        return res

    def update_utilitys(self, U: List[List[float]]):
        for state in self.getStates():
            row, col = state.getPos()
            state.setUtility(U[row][col])
    
    def init_utilities(self):
        rows, cols = self.getGrid().getRows(), self.getGrid().getCols()
        self.update_utilitys(_zeros(rows, cols))

    def update_actions(self, actions: List[Action]):
        for state in self.getStates():
            state.setActions(actions)

    def init_actions(self):
        self.update_actions([])
    
    ## helper functions ##
    def _nextState(self, state: State, action: Action) -> State:
        row, col = state.getPos()
        delta_row, delta_col = action.getDeltaRow(), action.getDeltaCol()
        next_row, next_col = row + delta_row, col + delta_col
        return self.getState((next_row, next_col))
        """
        # as in the book
        if next_state == None or next_state.is_wall():
            return state
        else: return next_state
        """

actions = {
    'Left': Action(
        delta_row= 0,
        delta_col= -1,
        name= 'Left'
    ),
    'Down': Action(
        delta_row= +1,
        delta_col= 0,
        name= 'Down'
    ),
    'Up': Action(
        delta_row= -1,
        delta_col= 0,
        name= 'Up'
    ),
    'Right': Action(
        delta_row= 0,
        delta_col= +1,
        name= 'Right'
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

def Value_Iteration(mdp: MDP, EPSILON: float, max_iteration: int = 200) -> int:
    
    PRECISION = int(np.log10(EPSILON) -1)
    gamma = mdp.getDiscountFactor()
    rows, cols = mdp.getGrid().getRows(), mdp.getGrid().getCols()
    U_next_value = _zeros(rows, cols)
    threshold = EPSILON *(1 -gamma) /gamma if gamma not in [0, 1] else EPSILON

    # do while
    i = 0

    while True:
        i += 1
        mdp.update_utilitys(U_next_value)
        delta = pow(10, PRECISION)

        # loop on states
        for state in mdp.getStates():
            if not state.is_free(): continue

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

            # and store it in U_next, update the actions
            row, col = state.getPos()
            U_next_value[row][col] = maxx_value
            state.setActions(maxx_actions)
            
            # update delta to be the largest error
            error = abs(maxx_value - state.getUtility())
            delta = max(delta, error)
        
        if delta <= threshold or i >= max_iteration: break
    
    return i

def Policy_Iteration(mdp: MDP, EPSILON: float, max_iteration: int = 200, variation: int =1) -> int:
    
    rows, cols = mdp.getGrid().getRows(), mdp.getGrid().getCols()
    U_next_value = _zeros(rows, cols)

    

    def policy_eval(toprint: bool):

        PRECISION = int(np.log10(EPSILON) -1)
        rows, cols = mdp.getGrid().getRows(), mdp.getGrid().getCols()
        U_next_value = _zeros(rows, cols)
        threshold = EPSILON

        # do while
        j = 0

        while True:
            j += 1
            mdp.update_utilitys(U_next_value)
            delta = pow(10, PRECISION)
            print(mdp.getGrid().print('u'))

            # loop on states
            for state in mdp.getStates():
                if not state.is_free(): continue

                # calculate the q value for the current action
                action = state.getActions()[0]
                q_val = Q_value(mdp, state, action)
                print(f'    {state.getPos()}: qva: {q_val}', end= ' ')

                # and store it in U_next
                row, col = state.getPos()
                U_next_value[row][col] = q_val
                
                # update delta to be the largest error
                error = abs(q_val - state.getUtility())
                if toprint:
                    print(f'    error: {round(error, 4)}')
                delta = max(delta, error)
            if toprint:
                print('\n', j, delta, threshold, '\n')
            print()
            if delta <= threshold or j >= max_iteration: break
        
        
        return j

    def policy_improvement():

        unchanged = True
        # loop on states
        for state in mdp.getStates():
            if not state.is_free(): continue

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

            # and store it in U_next, update the actions
            if round(maxx_value, 10) > round(state.getUtility(), 10):
                row, col = state.getPos()
                U_next_value[row][col] = maxx_value
                state.setActions(maxx_actions)
                unchanged = False
        
        return unchanged

    # do while
    x = []
    y = []
    i = 0

    while True:
        i += 1

        toprint = True if i == 1 else False
        j = policy_eval(toprint)
        unchanged = policy_improvement()

        x.append(i)
        y.append(j)

        if unchanged or i >= max_iteration: break

    print(x, y)
    return i, x, y

def Q_value(mdp: MDP, state: State, action: Action) -> float:

    getcontext().prec = 50
    gamma = _precision(mdp.getDiscountFactor())
    summ = _precision(0)
    for inter_probability, inter_state_next in mdp.TransiotionModel(state, action):
        prob, reward, utility = _precision(inter_probability), _precision(inter_state_next.getReward()), _precision(inter_state_next.getUtility())
        summ += prob * (reward + gamma * utility)
    
    return summ

def transition_model(mdp: MDP, state_curr: State, action: Action) -> Iterable[Tuple[float, State]]:

    p, q = mdp.getP(), round((1 -mdp.getP()) /2, 1)
    probabilities_next = [p, q, q]
    states_next = [mdp._nextState(state_curr, action)]
    action_map = {
        'Up': ['Up-Right', 'Up-Left'],
        'Down': ['Down-Right', 'Down-Left'],
        'Right': ['Up-Right', 'Down-Right'],
        'Left': ['Up-Left', 'Down-Left']
    }
    actions_diag_potential_str = action_map.get(action.getName(), [])
    
    for action_diag_potential_str in actions_diag_potential_str:
        state_next = mdp._nextState(state_curr, actions_diagonals[action_diag_potential_str])
        if not (state_next.is_invalid() or state_next.is_wall()):
            states_next.append(state_next)
        else:
            probabilities_next[0] += probabilities_next.pop()

    return zip(probabilities_next, states_next)
    """
    # as in the book
    action_map = {
        'Up': ['Right', 'Left'],
        'Down': ['Right', 'Left'],
        'Right': ['Up', 'Down'],
        'Left': ['Up', 'Down']
    }
    actions_potential_str = action_map.get(action.getName(), [])

    for action_potential_str in actions_potential_str:
        state_next = mdp._nextState(state_curr, actions[action_potential_str])
        # as in the book
        states_next.append(state_next)
    """            

## Helper Functions ##
def _precision(number: int) -> Decimal:
    return Decimal(str(number))

def _zeros(rows: int, cols: int) -> List[List[int]]:
    return [[0 for _ in range(cols)] for _ in range(rows)]

## Driver code ##

def plt_save(mdp: MDP, i: int, Q: int):

    rows, cols = mdp.getGrid().getRows(), mdp.getGrid().getCols()
    utilities_mat = [[mdp.getState((row, col)).getUtility() for col in range(cols)] for row in range(rows)]
    action_mat = [[mdp.getState((row, col)).printActions('s') for col in range(cols)] for row in range(rows)]

    # Define the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Name:  Matan Suliman\n\nIterations Until Convergens:  {i}')

    # Plot the utilities matrix
    Z = np.array(utilities_mat, dtype=float)
    im = ax1.imshow(Z, cmap='seismic')
    ax1.set_xticks(np.arange(cols))  # Add x-tick labels
    ax1.set_yticks(np.arange(rows))  # Add y-tick labels
    ax1.set_yticklabels(np.arange(rows)[::-1])  # Flip y-tick labels
    ax1.set_xticklabels(np.arange(cols))
    fig.colorbar(im, ax=ax1, shrink=0.8) # Add colorbar to the utilities plot

    # Annotate the cells in the utilities matrix
    for row in range(rows):
        for col in range(cols):
            value = utilities_mat[row][col]
            background_color = im.cmap(im.norm(Z[row, col]))  # Get the background color
            # Calculate luminance (perceived brightness) of the background
            r, g, b, _ = background_color  # RGBA values
            luminance = 0.299 * r + 0.587 * g + 0.114 * b  # Standard luminance formula
            text_color = "white" if luminance < 0.5 else "black"  # White text on dark background
            
            ax1.text(
                col, row, f"{value:.4f}",
                ha="center", va="center", color=text_color
            )

    # Plot the action policy table
    ax2.set_axis_off()
    table = ax2.table(cellText=action_mat, cellLoc='center', loc='center')  # Create the table
    table.scale(1, 2)  # Adjust cell size for better readability

    # Save the combined figure
    val = {
        '2.1': 'ValueIteration_Combined',
        '2.2': f'ValueIteration_gamma{mdp.getDiscountFactor()}',
        '2.3': f'ValueIteration_p{mdp.getP()}',
        '3.1': f'PolicyIteration_Combined'
    }
    file_name = f'input_{val[Q]}_MatanSuliman.jpg'
    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.savefig(file_name, dpi=300)
    plt.close()

def graph_save(mdp: MDP, x: List[int], y: List[int], Q: int):

    plt.scatter(x, y)
    plt.xticks(x)  # Add x-tick labels

    # Save the combined figure
    val = {
        '3.2.1': f'Graph_variation1'
    }
    file_name = f'input_PolicyIteration_{val[Q]}_MatanSuliman.jpg'
    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.savefig(file_name, dpi=300)
    plt.close()

def init_mdp(status_mat: List[List[int]], rewards_mat: List[List[float]], transition_model: Callable, discount_factor: float, p: float):
    
    return 

def create_npz(name: str, r: int):
    """
    status: 0 = wall, 1 = free, -1 = goal, -9 = invalid
    """
    rewards = [
        [r, r, r, 1],
        [r, 0, r, -1],
        [r, r, r, r]
    ]
    status = [
        [1, 1, 1, -1],
        [1, 0, 1, -1],
        [1, 1, 1, 1]
    ]
    np.savez(f'{name}.npz', status, rewards)

def handle_input():
    if len(sys.argv) != 3:
        raise "Usage: python MDP.py <path_to_npz_file> <ValueIteration / PolicyIteration>"
    else:
        return sys.argv[1], sys.argv[2]

def load_data_from_npz(npz_file_path: str):
    data = np.load(npz_file_path)
    return data['arr_0'].tolist(), data['arr_1'].tolist()

def main():
    npz_file_path, iteration_type = handle_input()
    status_mat, reward_mat = load_data_from_npz(npz_file_path)

    GAMMA = 0.9
    P = 0.8
    EPSILON = pow(10, -15)
    if iteration_type == 'ValueIteration':
        
        mdp = MDP(
            grid= Grid.mats_to_grid(status_mat, reward_mat),
            transition_model= transition_model,
            discount_factor= GAMMA,
            p= P
        )
        
        i = Value_Iteration(mdp, EPSILON)
        plt_save(mdp, i, Q= '2.1')
        print('done Q2.1')

        for gamma_loop in np.linspace(0, 1, 5):
            mdp.init_utilities()
            mdp.setDiscountFactor(gamma_loop)
            i = Value_Iteration(mdp, EPSILON)
            plt_save(mdp, i, Q= '2.2')
        print('done Q2.2')

        mdp.setDiscountFactor(GAMMA)
        for p_loop in np.linspace(0.4, 1, 4):
            mdp.init_utilities()
            mdp.setP(p_loop)
            i = Value_Iteration(mdp, EPSILON)
            plt_save(mdp, i, Q= '2.3')
        print('done Q2.3')
    
    elif iteration_type == 'PolicyIteration':

        mdp = MDP(
            grid= Grid.mats_to_grid(status_mat, reward_mat, action_defalut_str='Up'),
            transition_model= transition_model,
            discount_factor= GAMMA,
            p= P
        )
        
        i, x, y = Policy_Iteration(mdp, EPSILON, variation=1)
        plt_save(mdp, i, Q= '3.1')
        graph_save(mdp, x, y, Q= '3.2.1')
        print('done Q3.2.1')

        """mdp.init_utilities()
        mdp.update_actions([actions['Up']])

        i, x, y = Policy_Iteration(mdp, EPSILON, variation=1)
        plt_save(mdp, i, Q= '3.1')
        graph_save(mdp, x, y, Q= '3.2.1')
        print('done Q3.2.1')"""

    else:
        raise ValueError('<ValueIteration / PolicyIteration>')

if __name__ == '__main__':
    for r in [-0.04, -0.01]:
        create_npz(f'test{r}', r)
    main()