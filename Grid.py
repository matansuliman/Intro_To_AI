"""
Name: Matan suliman
"""

# import dependencies
from State import *
from copy import copy, deepcopy
import pandas as pd
from typing import * # clearity of code

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

    def mats_to_grid(mat_status: List[List[int]], mat_reward: List[List[int]]) -> 'Grid':
        rows, cols = len(mat_status), len(mat_status[0])
        grid = Grid(rows, cols)
        for row in range(rows):
            for col in range(cols):
                grid.setState((row, col), status= mat_status[row][col], reward= mat_reward[row][col])

        return grid