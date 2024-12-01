"""
Name: Matan suliman
ID: 322982620
"""

import argparse
from typing import List # clearity of code
from copy import deepcopy
import random as rd

## Classes ##

class Game():
    def __init__(self,
                 name: str,
                 size: int,
                 cutoff: int =1,
                 wieghts: List[int] =[[1, 1, 1, 1], [1, 1, 1, 1]]
                 ):
        self._name = name
        self._size = size
        self._cutoff = cutoff
        self._weights = wieghts

    def InitialState(self):
        """
        params:
            None
        return:
            the initial state of the game
        """
        n = self._size
        # all zeros
        initial_state = [[0 for i in range(n)] for j in range(n)]
        # 4 center pieces
        initial_state[n //2 -1][n //2 -1] = 1
        initial_state[n //2 -1][n //2] = -1
        initial_state[n //2][n //2 -1] = -1
        initial_state[n //2][n //2] = 1
        return initial_state

    def ToMove(self, state: List[List[int]]):
        """
        params:
            state - the curr state
        return:
            The player whise turn it is to move in state s
        """
        count = 0
        n = len(state)
        for i in range(n):
            for j in range(n):
                if state[i][j] != 0:
                    count += 1
        return 1 if count %2 == 0 else -1

    def Actions(self, state: List[List[int]], player: int =None):
        """
        params:
            state - the curr state
        return:
            the set of legal moves is state s
        """
        res = []
        player = self.ToMove(state) if player is None else player
        n = len(state)

        # check all 8 directions (up, up-right, right, right-down, down, down-left, left, left-up) for a valid flip
        def directions_from_position(i, j):
            temp_directions = {}

            for direction_name, direction_func in directions.items():
                for k in range(1, n): # k is length -1
                    new_i, new_j, bounds = direction_func(i, j, k, n)
                    if not bounds: # out of bounds
                        break
                    if state[new_i][new_j] == 0: break # spaces are not allowed
                    elif state[new_i][new_j] == player:
                        if k +1 == 2:
                            # length is 2 and both the same
                            break
                        else: # k +1 > 2
                            #print(f'found {state[new_i][new_j]} at [{new_i}][{new_j}] returning [{i}][{j}]')
                            temp_directions[direction_name] = direction_func
                            break
                    else: # state[new_i][new_j] == -player
                        pass
            if len(temp_directions) != 0:
                res.append({
                    "X": i,
                    "Y": j,
                    "directions": temp_directions
                })
        
        for i in range(n):
            for j in range(n):
                if state[i][j] == 0:
                    # for every blank spot: append( i, j, directions)
                    directions_from_position(i, j)
        
        return res
    
    def Result(self, state: List[List[int]], action: dict):
        """
        params:
            state - the curr state
            action - action to perform
        return:
            the resulting state
        """
        res_state = deepcopy(state)
        player = self.ToMove(state)
        res_state[action["X"]][action["Y"]] = player # place the piece
        n = len(state)
        # flip pieces of opponent
        for direction_func in action["directions"].values():
            for k in range(1, n): # k is length -1
                new_i, new_j, _ = direction_func(action["X"], action["Y"], k, n)
                if res_state[new_i][new_j] == -player:
                    res_state[new_i][new_j] = player
                else:
                    break
        return res_state
    
    def isTerminal(self, state: List[List[int]]):
        """
        params:
            state - the curr state
        return:
            true if state is an end game and, false otherwise
        """
        return len(self.Actions(state)) == 0

    def isCutOff(self, state: List[List[int]], depth: int):
        """
        params:
            state - the curr state
            depth - the number of moves played from the searching state
        return:
            true if state is at max depth specified by cutoff param or is a terminal state, false otherwise
        """
        return self.isTerminal(state) or depth >= self._cutoff
    
    def Utility(self, state: List[List[int]]):
        """
        params:
            state - the curr state
        return:
            a numeric value corresponds to a cutoff state and a player
        """
        player = self.ToMove(state)
        w_idx = (player +1) //2 # mapping -1 to 0 and 1 to 1
        return Eval(self, state, player, self._weights[w_idx])

# helper function and data structurces
directions = {
    'Up' : lambda i, j, k, n: (i -k, j, 0 <= i -k),
    'Down' : lambda i, j, k, n: (i +k, j, i +k < n),
    'Right' : lambda i, j, k, n: (i, j +k, j +k < n),
    'Left' : lambda i, j, k, n: (i, j -k, 0 <= j -k),
    'Up_Right' : lambda i, j, k, n: (i -k, j +k, i -k and j +k < n),
    'Up_Left' : lambda i, j, k, n: (i -k, j -k, i -k and 0 <= j -k),
    'Down_Right' : lambda i, j, k, n: (i +k, j +k, i +k < n and j +k < n),
    'Down_Left' : lambda i, j, k, n: (i +k, j -k, i +k < n and 0 <= j -k)
}

def Count(state: List[List[int]], player: int):
    """
    params:
        state - the curr state
        player - the player turn
    return:
        the player piece count
    """
    n = len(state)
    count = 0
    for i in range(n):
        for j in range(n):
            if state[i][j] == player: 
                count += 1
    return count

def coinParity(game: Game, state: List[List[int]], player: int):
    """
    params:
        game - the game played
        state - the curr state
        player - the player turn
    return:
        the player piece count minus the opponent piece count
    """
    countMy = Count(state, player= player)
    countOp = Count(state, player= -player)
    coinParity = abs(countMy - countOp) / (countMy + countOp)
    return coinParity if countMy > countOp else -coinParity

def mobility(game: Game, state: List[List[int]], player: int):
    """
    mobility is the number of actions a player can do
    params:
        state - the curr state
        player - the player turn
    return:
        the player mobility minus the opponent mobility
    """
    mobilityMy = len(game.Actions(state, player= player))
    mobilityOp = len(game.Actions(state, player= -player))
    if (mobilityMy + mobilityOp) == 0: mobility = 0
    else: mobility = abs(mobilityMy - mobilityOp) / (mobilityMy + mobilityOp)
    return mobility if mobilityMy > mobilityOp else -mobility

def cornerControl(game: Game, state: List[List[int]], player: int):
    """
    params:
        game - the game played
        state - the curr state
        player - the player turn
    return:
        the player corner control minus the opponent corner control
    """
    n = len(state)
    corners = [(0, 0), (n -1, 0), (0, n -1), (n -1, n -1)]
    res = 0
    for x, y in corners:
        if state[x][y] == player: res += 1
        elif state[x][y] == -player: res -= 1
        else: pass
    return res

def EdgeControl(game: Game, state: List[List[int]], player: int):
    """
    params:
        game - the game played
        state - the curr state
        player - the player turn
    return:
        the player edge control minus the opponent edge control
    """
    n = len(state)
    edges = []
    for i in range(1, n-1):
        edges.append((0, i))  # Top row
        edges.append((n-1, i))  # Bottom row
        edges.append((i, 0))  # Left column
        edges.append((i, n-1))  # Right column

    res = 0
    for x, y in edges:
        if state[x][y] == player: res += 1
        elif state[x][y] == -player: res -= 1
        else: pass
    return res

def Eval(game: Game, state: List[List[int]], player: int, weights: List[int]):
    """
    wieghted linear function
    params:
        game - the game played
        state - the curr state
        player - the player turn
        weights - the weights to applay
    return:
        a numeric value corresponds to a state and based on a heuristic evaluation function
    """
    normalized_weights = [w / sum(weights) for w in weights] 
    functions = [coinParity, mobility, cornerControl, EdgeControl]

    res = 0
    for w, f in zip(normalized_weights, functions):
        res += w * f(game, state, player)
    return res 

def printState(state: List[List[int]], action: dict =None, player: int =None, depth: int =0):
    """
    prints state information
    params:
        state - the curr state
        action - the action played to get to this state
        player - the player that did the action
        depth - the depth of the state from the start state
    return:
        None
    """
    output = {
        1: "X",
        -1: 'O',
        0: '-'
    }
    
    if action == None: print(f'State {depth}')
    else: print(f'State {depth}, {output[player]} moved, Action{action["X"], action["Y"], list(action["directions"].keys())}')

    n = len(state)
    for i in range(n):
        for j in range(n):
            print(output[state[i][j]], end='')
        print()
    print()

def Display(game: Game, state: List[List[int]], action: dict =None, player: int =None, depth: int =0):
    """
    prints a source state, a target state after an action and the result of that action count wise
    params:
        game - the game played
        state - the curr state
        action - the action to preform
        player - the player that will preform the action
        depth - the depth of the state from the start state
    return:
        None
    """

    def DisplaySource():
        print('Source state: ', end='')
        printState(state, depth=depth)
    
    def DisplayTarget():
        print('Target state: ', end='')
        printState(state= game.Result(state, action2), action= action2, player= -player, depth= depth +1)
    
    def DisplayResult():
        countX = Count(game.Result(state, action2), 1)
        countO = Count(game.Result(state, action2), -1)
        print(f'Result: X: {countX}, O: {countO}, Total: {countX + countO} disks')
    
    for action2 in game.Actions(state):
        DisplaySource()
        DisplayTarget()
        DisplayResult()
        print('###################################')


## Algorithms ##

def displayAllActions(game: Game, num: int):
    """
    shows all the actions avialiable and their results from a valid state in depth num
    params:
        game - the game played
        num - the depth of the start state
    return:
        None
    """
    # init state, action, depth
    curr_state = game.InitialState()
    curr_action = None
    curr_depth = 0
    
    while curr_depth < num -4:
        curr_action = methodical_search_func(game= game, state= curr_state)
        curr_state = game.Result(curr_state, curr_action)
        curr_depth += 1
    
    Display(game=game, state= curr_state, action= curr_action, player= -game.ToMove(curr_state), depth= curr_depth)

def methodical(game: Game, n: int):
    """
    simlulates a game with methodical action choosing
    prints the first n states of the game and the end state
    params:
        game - the game played
        n - the depth of the start state
    return:
        None
    """
    simulate(game= game, n= n, search_func= methodical_search_func)

def random(game: Game, n: int):
    """
    simlulates a game with random action choosing
    prints the first n states of the game and the end state
    params:
        game - the game played
        n - the depth of the start state
    return:
        None
    """
    simulate(game= game, n= n, search_func= random_search_func)


"""def find(game):
    for i in range(100):
        h1, h2 = [], []
        for i in range(4):
            h1.append(rd.randint(0, 20))
            h2.append(rd.randint(0, 20))

        temp = game._weights # store prev weights

        # apply weights
        game._weights = [h1, h2]
        
        simulate(game= game)

        game._weights = temp # backtrack"""

def H(game: Game, huristics: List[str]):
    """
    simlulates a game with MiniMax tree search for choosing action, based on huristics for each player
    prints end state
    params:
        game - the game played
        huristics - the huristic of each player, e.g ['H1', 'H2']
    return:
        None
    """
    #find(game)
    huristics_name_to_weights = {
        'H1': [19, 15, 7, 6],
        'H2': [17, 12, 8, 19]
    }
    
    temp = game._weights # store prev weights

    # apply weights
    game._weights = [
        huristics_name_to_weights[huristics[0]],
        huristics_name_to_weights[huristics[1]]
    ]
    
    simulate(game= game)

    game._weights = temp # backtrack

## Helper algorithms ##

def H_MaxValue(game: Game, state: List[List[int]], depth: int):
    """
    computes the maximum utility value for a given game state using the MiniMax algorithm
    params:
        game - the game played
        state - the curr state
        depth - The depth limit for the MiniMax search.
    return:
        the minimum utility value achievable from the given state and the corresponding action
    """
    if game.isCutOff(state, depth):
        return game.Utility(state), None
    else:
        val, action = float('-inf'), None
        for action2 in game.Actions(state):
            val2, _ = H_MinValue(game, game.Result(state, action2), depth +1)
            if val2 > val: # if found bigger val: save it and the move to it
                val, action = val2, action2
        return val, action

def H_MinValue(game: Game, state: List[List[int]], depth: int):
    """
    computes the minimum  utility value for a given game state using the MiniMax algorithm
    params:
        game - the game played
        state - the curr state
        depth - The depth limit for the MiniMax search.
    return:
        the minimum utility value achievable from the given state and the corresponding action
    """
    if game.isCutOff(state, depth):
        return game.Utility(state), None
    else:
        val, action = float('inf'), None
        for action2 in game.Actions(state):
            val2, _ = H_MaxValue(game, game.Result(state, action2), depth +1)
            if val2 < val: # if found bigger val: save it and the move to it
                val, action = val2, action2
        return val, action

def H_MiniMaxSearch(game: Game, state: List[List[int]]):
    """
    executes the MiniMax algorithm to choose the best action for the current state
    params:
        game - the game played
        state - the curr state
    return:
        the optimal action determined by the MiniMax algorithm.
    """
    player = game.ToMove(state)
    if player == 1: # first player
        _ , action = H_MaxValue(game, state, depth= 0)
    else: # player == -1 : second player
        _ , action = H_MinValue(game, state, depth= 0)
    return action

def methodical_search_func(game: Game, state: List[List[int]]):
    """
    params:
        game - the game played
        state - the curr state
    return:
        the first action that the player whose turn in the state can perform
    """
    valid_actions = game.Actions(state)
    return valid_actions[0] if valid_actions else None

def random_search_func(game: Game, state: List[List[int]]):
    """
    params:
        game - the game played
        state - the curr state
    return:
        a random action that the player whose turn in the state can perform
    """
    valid_actions = game.Actions(state)
    return rd.choice(valid_actions) if valid_actions else None

def simulate(game: Game, n: int =0, search_func= H_MiniMaxSearch):
    """
    simulates a game by repeatedly applying the search function to make moves
    prints the first n states of the game and the end state
    params:
        game - the game played
        n - the number of moves to print, defaults 0
        search_func - the search function to determine the best move, defaults to `H_MiniMaxSearch`
    return:
        the utility value or result after simulating the game for `n` moves
    """
    # init state, action, depth
    curr_state = game.InitialState()
    curr_action = None
    curr_depth = 0

    while True:
        if n > 0:
            printState(state= curr_state, action= curr_action, player= -game.ToMove(curr_state), depth= curr_depth)
            n -= 1
        # display the n first moves and states
        if curr_action := search_func(game= game, state= curr_state):
            curr_state = game.Result(curr_state, curr_action)
            curr_depth += 1
        else:
            break
    printState(state= curr_state, action= curr_action, player= -game.ToMove(curr_state), depth= curr_depth)

    countX = Count(curr_state, 1)
    countO = Count(curr_state, -1)
    print(f'Result: X: {countX}, O: {countO}, Total: {countX + countO} disks')

## Driver code ##

def main():

    def add_arguments(parser):
        
        # flags
        parser.add_argument('--displayAllActions', type=int, metavar='num', help='Display all possible actions')
        parser.add_argument('--methodical',        type=int, metavar='n',   help='Run the methodical mode with specified parameter')
        parser.add_argument('--random',            type=int, metavar='n',   help='Run the random mode with specified parameter')
        parser.add_argument('--ahead',             type=int, metavar='2',   help='Set look-ahead level for heuristic mode')

        # Positional arguments for players
        parser.add_argument('Heuristics', nargs='*', help="Heuristics configuration (H1, H2, or both)")

        args = parser.parse_args() # Parse arguments

        # Validate 'Heuristics' if provided
        valid_Heuristics = {'H1', 'H2'}
        if args.Heuristics and not all(heuristic in valid_Heuristics for heuristic in args.Heuristics):
            parser.error(f"Invalid choice: {args.Heuristics} (choose from 'H1', 'H2')")

    # init game
    reversi = Game(name= 'Reversi', size= 8)

    # parser
    parser = argparse.ArgumentParser(description="Reversi Game Runner")
    add_arguments(parser)
    args = parser.parse_args()

    # Logic to handle parsed arguments
    if args.displayAllActions is not None: displayAllActions(game= reversi, num= args.displayAllActions)
    elif args.methodical is not None: methodical(game= reversi, n= args.methodical)
    elif args.random is not None: random(game= reversi, n= args.random)
    elif args.ahead is not None and args.Heuristics:
        reversi._cutoff = 2
        if len(args.Heuristics) == 1: args.Heuristics = args.Heuristics *2
        H(game= reversi, huristics=args.Heuristics)
    elif args.Heuristics:
        if len(args.Heuristics) == 1: args.Heuristics = args.Heuristics *2
        H(game= reversi, huristics=args.Heuristics)
    else:
        print("Invalid or incomplete arguments provided.")

if __name__ == '__main__':
    main()