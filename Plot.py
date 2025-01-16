"""
Name: Matan suliman
"""

# import dependencies
from MDP import MDP
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import TwoSlopeNorm
from typing import * # clearity of code

def plt_save(mdp: MDP, i: int, Q: int, npz_file_path: str):

    rows, cols = mdp.getGrid().getRows(), mdp.getGrid().getCols()
    utilities_mat = [[mdp.getState((row, col)).getUtility() for col in range(cols)] for row in range(rows)]
    action_mat = [[mdp.getState((row, col)).printActions('s') for col in range(cols)] for row in range(rows)]

    # Annotate goal and wall cells in the action matrix
    for state in mdp.getStates():
        row, col = state.getPos()
        if state.is_goal(): action_mat[row][col] = 'O'
        if state.is_wall(): action_mat[row][col] = 'X'

    # Define the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Name:  Matan Suliman\n\nIterations Until Convergens:  {i}')

    # Plot the utilities matrix
    Z = np.array(utilities_mat, dtype=float)
    im = ax1.imshow(Z, cmap='seismic', norm=TwoSlopeNorm(vcenter=0))
    fig.colorbar(im, ax=ax1) # Add colorbar to the utilities plot

    # Annotate the cells in the utilities matrix
    """
    for row in range(rows):
        for col in range(cols):
            value = utilities_mat[row][col]
            background_color = im.cmap(im.norm(Z[row, col]))  # Get the background color
            # Calculate luminance (perceived brightness) of the background
            r, g, b, _ = background_color  # RGBA values
            luminance = 0.299 * r + 0.587 * g + 0.114 * b  # Standard luminance formula
            text_color = "white" if luminance < 0.5 else "black"  # White text on dark background

            ax1.text(col, row, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=4)
    """
    
    # Plot the action policy table
    ax2.set_axis_off()
    ax2.table(cellText=action_mat, cellLoc='center', loc='center')  # Create the table

    # Save the combined figure
    val = {
        '2.1': 'ValueIteration_Combined',
        '2.2': f'ValueIteration_gamma{mdp.getDiscountFactor()}',
        '2.3': f'ValueIteration_p{mdp.getP()}',
        '3.1': f'PolicyIteration_Combined',
        '3.2.1': f'PolicyIteration_variation1',
        '3.2.2': f'PolicyIteration_variation2',
    }

    plt.tight_layout()  # Adjust layout to avoid overlap
    name = ''.join(npz_file_path.split('.')[:-1])
    plt.savefig(f'Results\\{name}_{val[Q]}_MatanSuliman.jpg', dpi= 500)
    plt.close()

def graph_save(mdp: MDP, x: List[int], y: List[int], Q: int, npz_file_path: str):

    plt.scatter(x, y)
    plt.plot(x, y)
    plt.xticks(x)  # Add x-tick labels
    plt.ylim(0)

    for i, val in enumerate(y):
        plt.text(x[i] + 0.1, y[i], f'{val}', fontsize=9, color='red')

    # Save the combined figure
    val = {
        '3.2.1': f'Graph_variation1',
        '3.2.2': f'Graph_variation2'
    }
    name = ''.join(npz_file_path.split('.')[:-1])
    file_name = f'Results\\{name}_PolicyIteration_{val[Q]}_MatanSuliman.jpg'
    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.savefig(file_name, dpi=300)
    plt.close()