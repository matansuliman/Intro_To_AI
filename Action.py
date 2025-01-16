"""
Name: Matan suliman
"""

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