"""
Name: Matan suliman
"""

# import dependencies
from Action import *
from typing import * # clearity of code


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