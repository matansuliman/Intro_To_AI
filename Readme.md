- Name: Matan suliman
- ID: 
- python version: 3.8.0
- book: 'Artificial Intelligence, a modern approach, 4'th edition'
---

# Background:
  Reversi game is a adversarial (2 player) game
  Setting: n x n board, n^2 pieces, each piece is black and white. each player picks a color.
  Goal: have the most amount of pieces of your color on the board in the end of the game.

# Program Overview:
notes:
- a **Player** and his **Piece** is represented with 1:black, 0:empty, -1:white
- a **State** is a n x n matrix representing the positions on the board.
- an **Action** is a dict with keys: ['X', 'Y', 'directions'] representing piece placed on (x,y) that affects the pieces in the directions

## Classes:
- **Game()**:
    - Attributes:
        - name: the name of the game (str)
        - size: the size of the borad (int)
        - cutoff: the depth of the MiniMax tree when searching, deafult 1
        - weights: the weights of the Utility function, deafult [[1, 1, 1, 1], [1, 1, 1, 1]]
    - Methods:
        - InitialState(): returns the initial state of the game
        - ToMove(s): The player whise turn it is to move in state s
        - Actions(s): the set of legal moves is state s
        - Result(s, a): gets a state and an action and returns the resulting state
        - isTerminal(s): return true if state is an end game and false otherwise
        - isCutOff(s): return true if state is at max depth specified by cutoff param or is a terminal state
        - Utility(s, p): return a numeric value corresponds to a cutoff state

## Helper functions to Reversi:
- Count(state, player): return the player piece count
- coinParity(game, state, player): returns the player piece count minus the opponent piece count
- mobility(game, state, player): returns the player mobility minus the opponent mobility
- cornerControl(game, state, player): returns the player corner control minus the opponent corner control
- EdgeControl(game, state, player): returns the player edge control minus the opponent edge control
- Eval(game, state, player, weights): returns a numeric value corresponds to a state and based on a heuristic evaluation function
- printState(state, action, player, depth):  prints state information
- Display(game, state, action, player, depth): prints a source state, a target state after an action and the result of that action count wise


## Algorithms:
- displayAllActions(game, num): shows all the actions avialiable and their results from a valid state in depth num
- methodical(game, n): simlulates a game with methodical action choosing
- methodical(game, n): simlulates a game with random action choosing
- H(game, huristics): simlulates a game with MiniMax tree search for choosing action, based on huristics for each player

## Helper algorithms:
- H_MaxValue(game, state, depth): computes the maximum utility value for a given game state using the MiniMax algorithm
- H_MinValue(game, state, depth): computes the minimum  utility value for a given game state using the MiniMax algorithm
- H_MiniMaxSearch(game, state): executes the MiniMax algorithm to choose the best action for the current state
- methodical_search_func(game, state): returns the first action that the player whose turn in the state can perform
- random_search_func(game, state): returns a random action that the player whose turn in the state can perform
- simulate(game, n , search_func): simulates a game by repeatedly applying the search function to make moves

# Answers to open questions
## Answer 1
- A. The size of the state space is 3^n (each position is either empty, black or white).
- B. Examples for valid and inaccessible states:
<br>
invalid state:
<br>
_ _ _ _ _ _ _ _<br>
_X _ X _X _ _<br>
_ _ _ _ _ _ _ _<br>
_ _ _ XO _ _ _<br>
_ _ _ OX _ _ _<br>
_ _ _ _ _ _ _ _<br>
_ _ _ _ _ _ _ _<br>
_ _ _ _ _ _ _ _<br>
<br>
inaccessible state:
<br>
_ _ _ _ _ _ _ _<br>
_ X _ _ _ _ _ _<br>
_ _ _ _ _ _ _ _<br>
_ _ _ XO _ _ _<br>
_ _ _ OX _ _ _ <br>
_ _ _ _ _ _ _ _<br>
_ _ _ _ _ _ _ _<br>
_ _ _ _ _ _ _ _<br>

- The actions are preformed in the following series: from top left pos to down right pos.
- When we reapet the methodical run we get the same result beacuse its the same path on the tree of possiable actions.

## Answer 2
- Both heuristic evaluation functions are useing weights to interpernt meaning.
- The features are: 
    - coinParity
    - mobility
    - cornerControl
    - edgeControl
- H1:[19, 15, 7, 6] Prioritizes short-term advantages (coin parity and mobility) at the expense of strategic positioning (corners and edges).
- H2:[17, 12, 8, 19], Overprioritizes edge control, which can lead to poor decisions like securing unstable edges instead of focusing on corners.

### first state:
_ _ _ _ _ _ _ _<br>
_ _ _ _ _ _ _ _<br>
_ _ _ _ X _ _ _<br>
_ _ _ XX _ _ _<br>
_ _ _ OXO _ _<br>
_ _ _ _ _ X _ _<br>
_ _ _ _ _ _ _ _<br>
_ _ _ _ _ _ _ _<br>

- 'X' moved
- coinParity = +3
- mobility = +3
- cornerControl = 0
- EdgeControl = 0
#### results
- H1: 19 *3 + 15 *3 + 7 *0 + 6 *0 = 102
- H2: 17 *3 + 12 *3 + 8 *0 + 19 *0 = 87

### second state:
X X X X OX X X<br>
X X X X OOOO<br>
X X O X OOOO<br>
X X O O OOOO<br>
X X O O OOOO<br>
X X O O OOO _<br>
X X X X OOO X<br>
O X X O OOO X<br>

- 'X' moved
- coinParity = -7
- mobility = +1
- cornerControl = +2
- EdgeControl = +7
#### results
- H1: 19 *-7 + 15 *1 + 7 *2 + 6 *7 = -62
- H2: 17 *-7 + 12 *1 + 8 *2 + 19 *7 = 42


### third state:
_ _ _ _ _ _ _ _<br>
_ _ _ _ _ _ _ _<br>
_ _ _ _ X XX _<br>
_ _OOO _ _ _<br>
_ _ _ OX _ _ _<br>
_ _ _ _ _ _ _ _<br>
_ _ _ _ _ _ _ _<br>
_ _ _ _ _ _ _ _<br>

- 'O' moved
- coinParity = 0
- mobility = +2
- cornerControl = 0
- EdgeControl = 0
#### results
- H1: 19 *0 + 15 *2 + 7 *0 + 6 *0 = 30
- H2: 17 *0 + 12 *2 + 8 *0 + 19 *0 = 24

#### sections
- 2.A: when we reapet the simulation using h1 or h2 we get the same result, its beacuse the same path on the tree of possiable actions.
- 2.B: we can compare running h1 as the first player and h2 as the second player and vise versa.

## Answer 3
- 3.A: <br>
 P1 VS P2 <br>
 H1 VS H2 => P1 <br>
 H2 VS H1 => P2

 - 3.B: There is an importance to run H1 and H2 as p1 p2 and p2 p1 becuase it will demonstrate that no matter the player starting the game, the heuristic is better

## Answer 4
- 4.A: when we reapet the simulation we get the same result, its beacuse the same actions are being picked, nothing changing. there is no randomness in the decisions.
- 4.B: Time complexity analysis - the max depth of a game is 60. the branching factor is given as 10. at each play we calulate a decition tree of depth 2, which is 10^2 = 100. in total we get 60 * 100 = 6000 operations.
- 4.C: If for each decision we would calulate a minimax tree to the bottom, the time complexity would be O(10^60). more precisily sum(10^m for m in range(60)) operations.
- 4.D: we can use Alpha-Beta pruning
- 4.E: Yes, it's advisable to change the heuristic during the game. In the early stages, the focus can be on piece count and mobility, and in the later stages, the focus could shift to controlling edges and corners, as they provide a strategic advantage. this can be implemented easily due to depth knowledge.