"""
Name: Matan suliman
"""

# import dependencies
from Action import *
from State import *
from Grid import *
from Plot import *

from decimal import Decimal, getcontext # precision
import numpy as np
import sys
from typing import * # clearity of code


# global vars
NUM_ARGS = 3
DEFAULT_MAX_ITERATIONS = 100
getcontext().prec = 150

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

    def getP(self) -> float:
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

    def _update_actions(self, action_str: str) -> None:
        for state in self.getStates():
            valid_actions = self.Actions(state)
            
            if len(valid_actions) == 0:
                 state.setActions([])
            elif actions[action_str] in valid_actions:
                state.setActions([actions[action_str]])
            else:
                state.setActions(valid_actions[:1])

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

def Value_Iteration(mdp: MDP, EPSILON: float, max_iteration: int = DEFAULT_MAX_ITERATIONS) -> int:
    
    # init variables
    PRECISION = -int(np.log10(EPSILON) +1)
    gamma = mdp.getDiscountFactor()
    rows, cols = mdp.getGrid().getRows(), mdp.getGrid().getCols()
    U_next_value = _zeros(rows, cols)
    threshold = EPSILON *(1 -gamma) /gamma if gamma not in [0, 1] else EPSILON

    # do while
    i = 0

    while True:
        i += 1
        mdp.update_utilitys(U_next_value)
        delta = pow(10, -PRECISION -3)

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

def Policy_Iteration(mdp: MDP, EPSILON: float, max_iteration: int = DEFAULT_MAX_ITERATIONS, variation: int =1) -> int:
    
    # init variables
    PRECISION = -int(np.log10(EPSILON) +1)
    gamma = mdp.getDiscountFactor()

    def policy_eval():

        if variation == 1:
            rows, cols = mdp.getGrid().getRows(), mdp.getGrid().getCols()
            U_next_value = _zeros(rows, cols)
        else: # variation == 2:
            U_next_value = [[cell.getUtility() for cell in row] for row in mdp.getGrid().getData()]
        
        threshold = EPSILON *(1 -gamma) /gamma if gamma not in [0, 1] else EPSILON

        # do while
        j = 0

        while True:
            j += 1
            mdp.update_utilitys(U_next_value)
            delta = threshold -1

            # loop on states
            for state in mdp.getStates():
                if not state.is_free(): continue

                # calculate the q value for the current action
                action = state.getActions()[0]
                q_val = Q_value(mdp, state, action)

                # and store it in U_next
                row, col = state.getPos()
                U_next_value[row][col] = q_val
                
                # update delta to be the largest error
                error = abs(q_val - state.getUtility())
                delta = max(delta, error)
            
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
            """
            3 conditions:
                1) maxx_value == float('-inf'): there are no possible actions from state there fore maxx_value is not changed
                2) round(maxx_value, PRECISION) > round(state.getUtility(), PRECISION): here maxx_value is not -inf therefore if the maxx_value is bigger
                    than the current utility, make an update
                3) the value hasent increaced but an actio was added to the maxx_actions
            """
            if maxx_value == float('-inf') or \
                round(maxx_value, PRECISION) > round(state.getUtility(), PRECISION) or \
                len(maxx_actions) > len(state.getActions()):

                state.setActions(maxx_actions)
                unchanged = False
            
        return unchanged

    # do while
    x = []
    y = []
    i = 1

    while True:

        x.append(i)
        y.append(policy_eval())
        unchanged = policy_improvement()
        i += 1

        if unchanged or i >= max_iteration: break

    return i, x, y

def Q_value(mdp: MDP, state: State, action: Action) -> float:
    
    gamma = _precision(mdp.getDiscountFactor())
    summ = _precision(0)
    for inter_probability, inter_state_next in mdp.TransiotionModel(state, action):
        prob, reward, utility = _precision(inter_probability), _precision(inter_state_next.getReward()), _precision(inter_state_next.getUtility())
        summ += prob * (reward + gamma * utility)
    
    return summ

def transition_model(mdp: MDP, state_curr: State, action: Action) -> Iterable[Tuple[float, State]]:

    # init variables
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

## Helper Functions ##
def _precision(number: int) -> Decimal:
    return Decimal(str(number))

def _zeros(rows: int, cols: int) -> List[List[int]]:
    return [[0 for _ in range(cols)] for _ in range(rows)]

def _setEPSILON(reward_mat: List[List[int]]) -> float:
    minimum = np.min(np.array(reward_mat))
    return pow(10, np.log10(abs(minimum)) -8) if minimum != 0 else pow(10, -8)

## Driver code ##

def handle_input():
    if len(sys.argv) != NUM_ARGS:
        raise "Usage: python MDP.py <path_to_npz_file> <ValueIteration / PolicyIteration>"
    else:
        return sys.argv[1], sys.argv[2]

def load_data_from_npz(npz_file_path: str):
    data = np.load(npz_file_path)
    return data['states'].tolist(), data['rewards'].tolist()

def main():
    npz_file_path, iteration_type = handle_input()
    status_mat, reward_mat = load_data_from_npz(npz_file_path)
    EPSILON = _setEPSILON(reward_mat)

    mdp = MDP(
            grid= Grid.mats_to_grid(status_mat, reward_mat),
            transition_model= transition_model,
            discount_factor= 0.9,
            p= 0.8
        )
    
    
    """print(mdp.getGrid().print('s'))
    print(mdp.getGrid().print('u'))
    print(mdp.getGrid().print('r'))"""

    if iteration_type == 'ValueIteration':
        
        mdp = MDP(
            grid= Grid.mats_to_grid(status_mat, reward_mat),
            transition_model= transition_model,
            discount_factor= 0.9,
            p= 0.8
        )
        
        i = Value_Iteration(mdp, EPSILON)
        plt_save(mdp, i, Q= '2.1', npz_file_path=npz_file_path)
        print('done Q2.1')
        
        for gamma_loop in np.linspace(0, 1, 5):
            mdp.init_utilities()
            mdp.setDiscountFactor(gamma_loop)
            i = Value_Iteration(mdp, EPSILON)
            plt_save(mdp, i, Q= '2.2', npz_file_path=npz_file_path)
        print('done Q2.2')

        mdp.setDiscountFactor(0.9)
        for p_loop in np.linspace(0.4, 1, 4):
            mdp.init_utilities()
            mdp.setP(p_loop)
            i = Value_Iteration(mdp, EPSILON)
            plt_save(mdp, i, Q= '2.3', npz_file_path=npz_file_path)
        print('done Q2.3')
    
    elif iteration_type == 'PolicyIteration':

        mdp = MDP(
            grid= Grid.mats_to_grid(status_mat, reward_mat),
            transition_model= transition_model,
            discount_factor= 0.9,
            p= 0.8
        )
        mdp._update_actions('Up')

        i, x, y = Policy_Iteration(mdp, EPSILON, variation= 2)
        plt_save(mdp, i, Q= '3.1', npz_file_path=npz_file_path)
        print('done Q3.1')
        
        graph_save(mdp, x, y, Q= '3.2.2', npz_file_path=npz_file_path)
        print('done Q3.2.2')
        
        mdp.init_utilities()
        mdp._update_actions('Up')

        i, x, y = Policy_Iteration(mdp, EPSILON, variation= 1)
        graph_save(mdp, x, y, Q= '3.2.1', npz_file_path=npz_file_path)
        print('done Q3.2.1')
        
    else:
        raise ValueError('<ValueIteration / PolicyIteration>')

if __name__ == '__main__':
    main()