"""
Lights Out Game Solver using Linear Algebra over GF(2)

This implementation solves the Lights Out puzzle by:
1. Creating a toggle matrix representing how each button press affects the grid
2. Solving the linear system: toggle_matrix * solution ≡ initial_state (mod 2)
3. Using Gaussian elimination over GF(2) to find the optimal solution

The game rules:
- Pressing any light toggles it and its 4 adjacent neighbors (up, down, left, right)
- Goal is to turn all lights off
- Each position should be pressed at most once in an optimal solution

Based on the mathematical approach described in:
- "Turning Lights Out with Linear Algebra" by Anderson & Feil (1998)
- YouTube explanation: https://www.youtube.com/watch?v=rQtRK-AJOGg
"""

import numpy as np
from typing import List, Optional, Tuple


def create_toggle_matrix(n: int) -> np.ndarray:
    """
    Create the toggle matrix for an n×n Lights Out game.
    
    Args:
        n (int): Size of the grid (n×n)
        
    Returns:
        np.ndarray: A (n²)×(n²) matrix where entry (i,j) is 1 if pressing button j 
                    affects position i, and 0 otherwise
    """
    size = n * n
    matrix = np.zeros((size, size), dtype=int)
    
    def get_index(row: int, col: int) -> int:
        """Convert 2D coordinates to 1D index"""
        return row * n + col
    
    def get_coords(index: int) -> Tuple[int, int]:
        """Convert 1D index to 2D coordinates"""
        return index // n, index % n
    
    # For each button position
    for button_idx in range(size):
        button_row, button_col = get_coords(button_idx)
        
        # The button affects itself
        matrix[button_idx, button_idx] = 1
        
        # The button affects its neighbors
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        for dr, dc in directions:
            neighbor_row = button_row + dr
            neighbor_col = button_col + dc
            
            # Check if neighbor is within bounds
            if 0 <= neighbor_row < n and 0 <= neighbor_col < n:
                neighbor_idx = get_index(neighbor_row, neighbor_col)
                matrix[neighbor_idx, button_idx] = 1
    
    return matrix


def gf2_gaussian_elimination(matrix: np.ndarray, vector: np.ndarray) -> Optional[np.ndarray]:
    """
    Solve the linear system Ax = b over GF(2) using Gaussian elimination.
    
    Args:
        matrix (np.ndarray): Coefficient matrix A
        vector (np.ndarray): Right-hand side vector b
        
    Returns:
        np.ndarray or None: Solution vector x, or None if no solution exists
    """
    n = matrix.shape[0]
    # Create augmented matrix
    augmented = np.hstack([matrix.copy(), vector.reshape(-1, 1)])
    
    # Forward elimination
    pivot_row = 0
    for col in range(n):
        # Find pivot
        pivot_found = False
        for row in range(pivot_row, n):
            if augmented[row, col] == 1:
                # Swap rows if needed
                if row != pivot_row:
                    augmented[[pivot_row, row]] = augmented[[row, pivot_row]]
                pivot_found = True
                break
        
        if not pivot_found:
            continue
        
        # Eliminate column
        for row in range(n):
            if row != pivot_row and augmented[row, col] == 1:
                augmented[row] = (augmented[row] + augmented[pivot_row]) % 2
        
        pivot_row += 1
    
    # Check for inconsistency
    for row in range(pivot_row, n):
        if augmented[row, n] == 1:  # 0 = 1, inconsistent
            return None
    
    # Back substitution
    solution = np.zeros(n, dtype=int)
    for row in range(min(pivot_row, n) - 1, -1, -1):
        # Find the leading 1 in this row
        leading_col = -1
        for col in range(n):
            if augmented[row, col] == 1:
                leading_col = col
                break
        
        if leading_col != -1:
            val = augmented[row, n]
            for col in range(leading_col + 1, n):
                val = (val + augmented[row, col] * solution[col]) % 2
            solution[leading_col] = val
    
    return solution


def vector_to_grid(vector: np.ndarray, n: int) -> np.ndarray:
    """Convert 1D vector to n×n grid"""
    return vector.reshape(n, n)


def grid_to_vector(grid: np.ndarray) -> np.ndarray:
    """Convert n×n grid to 1D vector"""
    return grid.flatten()


def solve_lights_out(initial_grid: List[List[int]]) -> Optional[List[List[int]]]:
    """
    Solve the Lights Out puzzle for the given initial configuration.
    
    Args:
        initial_grid (List[List[int]]): n×n list where 1 = light on, 0 = light off
        
    Returns:
        List[List[int]] or None: n×n grid showing which buttons to press 
                                (1 = press, 0 = don't press), or None if no solution exists
    """
    # Convert to numpy arrays
    grid = np.array(initial_grid, dtype=int)
    n = grid.shape[0]
    
    # Convert grid to vector
    initial_vector = grid_to_vector(grid)
    
    # Create toggle matrix
    toggle_matrix = create_toggle_matrix(n)
    
    # Solve the linear system
    solution_vector = gf2_gaussian_elimination(toggle_matrix, initial_vector)
    
    if solution_vector is None:
        return None
    
    # Convert solution back to grid
    solution_grid = vector_to_grid(solution_vector, n)
    return solution_grid.tolist()


def print_grid(grid: List[List[int]], title: str = "Grid") -> None:
    """Pretty print a grid"""
    print(f"\n{title}:")
    for row in grid:
        print(" ".join("●" if cell else "○" for cell in row))


def verify_solution(initial_grid: List[List[int]], 
                   solution_grid: List[List[int]]) -> bool:
    """
    Verify that the solution correctly turns off all lights.
    
    Args:
        initial_grid (List[List[int]]): Initial state of the lights
        solution_grid (List[List[int]]): Proposed solution (which buttons to press)
        
    Returns:
        bool: True if solution is correct, False otherwise
    """
    n = len(initial_grid)
    final_grid = [[initial_grid[i][j] for j in range(n)] for i in range(n)]
    
    # Apply each button press
    for i in range(n):
        for j in range(n):
            if solution_grid[i][j]:  # If this button is pressed
                # Toggle the button itself
                final_grid[i][j] = 1 - final_grid[i][j]
                
                # Toggle adjacent cells
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        final_grid[ni][nj] = 1 - final_grid[ni][nj]
    
    # Check if all lights are off
    return all(final_grid[i][j] == 0 for i in range(n) for j in range(n))


def count_button_presses(solution_grid: List[List[int]]) -> int:
    """Count total number of button presses in solution"""
    return sum(sum(row) for row in solution_grid)


def example_usage():
    """Demonstrate the solver with example puzzles"""
    
    print("=" * 60)
    print("LIGHTS OUT SOLVER - Linear Algebra Approach")
    print("Solving Ax ≡ b (mod 2) where:")
    print("- A is the toggle matrix")
    print("- x is the solution vector (buttons to press)")  
    print("- b is the initial state vector")
    print("=" * 60)
    
    # Example 1: Simple 3x3 puzzle
    puzzle1 = [
        [1, 0, 1],
        [0, 1, 0], 
        [1, 0, 1]
    ]
    
    print("\nExample 1: 3×3 puzzle")
    print_grid(puzzle1, "Initial state")
    
    solution1 = solve_lights_out(puzzle1)
    if solution1:
        print_grid(solution1, "Solution (buttons to press)")
        is_correct = verify_solution(puzzle1, solution1)
        print(f"Solution verified: {'✓' if is_correct else '✗'}")
        print(f"Total button presses: {count_button_presses(solution1)}")
    else:
        print("No solution exists for this configuration!")
    
    print("\n" + "-" * 40)
    
    # Example 2: 5x5 puzzle (classic Lights Out size)
    puzzle2 = [
        [1, 1, 0, 1, 1],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 1, 0, 1, 1]
    ]
    
    print("Example 2: 5×5 puzzle")
    print_grid(puzzle2, "Initial state")
    
    solution2 = solve_lights_out(puzzle2)
    if solution2:
        print_grid(solution2, "Solution (buttons to press)")
        is_correct = verify_solution(puzzle2, solution2)
        print(f"Solution verified: {'✓' if is_correct else '✗'}")
        print(f"Total button presses: {count_button_presses(solution2)}")
    else:
        print("No solution exists for this configuration!")
    
    print("\n" + "-" * 40)
    
    # Example 3: All lights on (stress test)
    puzzle3 = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    
    print("Example 3: All lights on (3×3)")
    print_grid(puzzle3, "Initial state")
    
    solution3 = solve_lights_out(puzzle3)
    if solution3:
        print_grid(solution3, "Solution (buttons to press)")
        is_correct = verify_solution(puzzle3, solution3)
        print(f"Solution verified: {'✓' if is_correct else '✗'}")
        print(f"Total button presses: {count_button_presses(solution3)}")
    else:
        print("No solution exists for this configuration!")
    
    # Mark Example: Simple 5x5 puzzle
    puzzle4 = [[
        0,
        1,
        0,
        1,
        0
      ],
      [
        0,
        0,
        1,
        1,
        1
      ],
      [
        0,
        0,
        0,
        1,
        0
      ],
      [
        0,
        1,
        1,
        0,
        1
      ],
      [
        0,
        1,
        0,
        0,
        0
      ]
    ]
    
    print("\nMarks Example: 5×5 puzzle")
    print_grid(puzzle4, "Initial state")
    
    solution4 = solve_lights_out(puzzle4)
    if solution4:
        print_grid(solution4, "Solution (buttons to press)")
        is_correct = verify_solution(puzzle4, solution4)
        print(f"Solution verified: {'✓' if is_correct else '✗'}")
        print(f"Total button presses: {count_button_presses(solution4)}")
    else:
        print("No solution exists for this configuration!")
    
    print("\n" + "-" * 40)


def solve_custom_puzzle(grid: List[List[int]]) -> None:
    """
    Solve a custom puzzle provided by the user.
    
    Args:
        grid (List[List[int]]): The initial state of the lights
    """
    print(f"\nSolving custom {len(grid)}×{len(grid[0])} puzzle...")
    print_grid(grid, "Initial state")
    
    solution = solve_lights_out(grid)
    if solution:
        print_grid(solution, "Solution (buttons to press)")
        is_correct = verify_solution(grid, solution)
        print(f"Solution verified: {'✓' if is_correct else '✗'}")
        print(f"Total button presses: {count_button_presses(solution)}")
        return solution
    else:
        print("No solution exists for this configuration!")
        return None


if __name__ == "__main__":
    try:
        example_usage()

    except ImportError as e:
        print(f"Error: {e}")
        print("This script requires numpy. Install it with: pip install numpy")
        print("Alternatively, a pure Python version can be implemented.")
