"""
Lights Out Game Solver using Linear Algebra over GF(2)

This implementation solves the Lights Out puzzle by:
1. Creating a toggle matrix representing how each button press affects the grid
2. Solving the linear system: toggle_matrix * solution ≡ initial_state (mod 2)
3. Using the high-performance gf2_lin_algebra library for optimal solving

The game rules:
- Pressing any light toggles it and its 4 adjacent neighbors (up, down, left, right)
- Goal is to turn all lights off
- Each position should be pressed at most once in an optimal solution

Based on the mathematical approach described in:
- "Turning Lights Out with Linear Algebra" by Anderson & Feil (1998)
- YouTube explanation: https://www.youtube.com/watch?v=rQtRK-AJOGg
"""

import numpy as np
import requests
import json
import uuid
import re
import time
import multiprocessing as mp
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple, Dict, Any

# High-performance GF(2) library for all grids
from gf2_lin_algebra import GF2Matrix

# Persistent HTTP session for connection pooling (reuses TCP connections)
_session = requests.Session()
_session.headers.update({"Content-Type": "application/json"})
adapter = requests.adapters.HTTPAdapter(
    pool_connections=16,
    pool_maxsize=16, 
    max_retries=requests.adapters.Retry(total=2, backoff_factor=0.1)
)
_session.mount("https://", adapter)
_session.mount("http://", adapter)

# Toggle matrix cache: maps (grid_shape_key) -> (matrix, valid_positions, pos_to_index)
_toggle_matrix_cache: Dict[str, Tuple[np.ndarray, List[Tuple[int, int]], Dict[Tuple[int, int], int]]] = {}


def _grid_shape_key(grid: List[List[int]]) -> str:
    """Create a hashable key representing the shape of a grid (which cells are valid)."""
    return "|".join(
        ",".join("1" if cell != -1 else "0" for cell in row)
        for row in grid
    )


def create_toggle_matrix_for_grid(grid: List[List[int]], use_cache: bool = True) -> Tuple[np.ndarray, List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """
    Create the toggle matrix for any grid shape (rectangular, octagon, cross, etc.).
    Optimized for speed with early rectangular grid detection.
    
    Args:
        grid (List[List[int]]): The game grid where -1 = blank cell, 0/1 = lights
        
    Returns:
        Tuple containing:
        - np.ndarray: Toggle matrix for valid positions only
        - List[Tuple[int, int]]: Valid positions (row, col) 
        - Dict[Tuple[int, int], int]: Mapping from (row, col) to matrix index
    """
    n_rows = len(grid)
    n_cols = len(grid[0]) if grid else 0
    
    # Check cache first (same grid shape = same toggle matrix)
    if use_cache:
        cache_key = _grid_shape_key(grid)
        if cache_key in _toggle_matrix_cache:
            return _toggle_matrix_cache[cache_key]
    
    # Fast path for rectangular grids (most common case)
    is_rectangular = all(len(row) == n_cols and all(cell != -1 for cell in row) for row in grid)
    
    if is_rectangular:
        # Optimized path for rectangular grids
        num_valid = n_rows * n_cols
        valid_positions = [(r, c) for r in range(n_rows) for c in range(n_cols)]
        pos_to_index = {(r, c): r * n_cols + c for r in range(n_rows) for c in range(n_cols)}
        
        # Create matrix more efficiently for rectangular case
        matrix = np.zeros((num_valid, num_valid), dtype=np.int8)  # Use int8 to save memory
        
        for r in range(n_rows):
            for c in range(n_cols):
                button_idx = r * n_cols + c
                matrix[button_idx, button_idx] = 1  # Button affects itself
                
                # Neighbors with bounds checking
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n_rows and 0 <= nc < n_cols:
                        neighbor_idx = nr * n_cols + nc
                        matrix[neighbor_idx, button_idx] = 1
        
        result = (matrix, valid_positions, pos_to_index)
        if use_cache:
            _toggle_matrix_cache[cache_key] = result
        return result
    
    # Fallback to general case for irregular grids
    valid_positions = []
    pos_to_index = {}
    
    for row in range(n_rows):
        for col in range(n_cols):
            if row < len(grid) and col < len(grid[row]) and grid[row][col] != -1:
                index = len(valid_positions)
                valid_positions.append((row, col))
                pos_to_index[(row, col)] = index
    
    num_valid = len(valid_positions)
    if num_valid == 0:
        raise ValueError("No valid positions found in grid!")
    
    # Create toggle matrix for valid positions only
    matrix = np.zeros((num_valid, num_valid), dtype=np.int8)  # Use int8
    
    for button_idx, (button_row, button_col) in enumerate(valid_positions):
        # The button affects itself
        matrix[button_idx, button_idx] = 1
        
        # The button affects its neighbors
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        for dr, dc in directions:
            neighbor_row = button_row + dr
            neighbor_col = button_col + dc
            
            # Check if neighbor exists and is valid (not -1, within bounds)
            if (0 <= neighbor_row < n_rows and 
                0 <= neighbor_col < n_cols and
                neighbor_row < len(grid) and 
                neighbor_col < len(grid[neighbor_row]) and
                grid[neighbor_row][neighbor_col] != -1):
                
                neighbor_pos = (neighbor_row, neighbor_col)
                if neighbor_pos in pos_to_index:
                    neighbor_idx = pos_to_index[neighbor_pos]
                    matrix[neighbor_idx, button_idx] = 1
    
    result = (matrix, valid_positions, pos_to_index)
    if use_cache:
        _toggle_matrix_cache[cache_key] = result
    return result


def create_toggle_matrix(n: int) -> np.ndarray:
    """
    Create the toggle matrix for an n×n rectangular Lights Out game.
    
    Args:
        n (int): Size of the grid (n×n)
        
    Returns:
        np.ndarray: A (n²)×(n²) matrix where entry (i,j) is 1 if pressing button j 
                    affects position i, and 0 otherwise
    """
    # Create a dummy rectangular grid for backward compatibility
    grid = [[0 for _ in range(n)] for _ in range(n)]
    matrix, _, _ = create_toggle_matrix_for_grid(grid)
    return matrix


def gf2_solve(toggle_matrix: np.ndarray, initial_vector: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
    """
    High-performance solver using gf2_lin_algebra library (Rust backend).
    Falls back to optimized numpy implementation for very large matrices or library errors.
    
    Args:
        toggle_matrix (np.ndarray): Toggle matrix  
        initial_vector (np.ndarray): Initial state vector
        debug (bool): Enable detailed debugging output
        
    Returns:
        np.ndarray or None: Solution vector, or None if no solution exists
    """
    
    try:
        rows, cols = toggle_matrix.shape
        
        if debug:
            print(f"🔍 DEBUG: Solving {rows}x{cols} matrix system")
            print(f"🔍 DEBUG: Matrix density: {np.count_nonzero(toggle_matrix)}/{rows*cols} = {100*np.count_nonzero(toggle_matrix)/(rows*cols):.1f}%")
        
        # For very large matrices (>500 vars), use numpy fallback immediately
        # This is much faster than converting large matrices to the library format
        if rows > 500:
            if debug:
                print(f"🔍 DEBUG: Using numpy solver for large matrix ({rows} variables)")
            print(f"   Using optimized numpy solver for large matrix ({rows} variables)")
            return gf2_numpy_solve(toggle_matrix, initial_vector, debug=debug)
        
        # For smaller matrices, try the gf2_lin_algebra library directly
        # Exception handling will catch any panics (rank-deficient, etc.)
        if debug:
            conversion_start = time.time()
        
        matrix_list = toggle_matrix.astype(bool).tolist()
        vector_list = initial_vector.astype(bool).tolist()
        
        if debug:
            conversion_time = time.time() - conversion_start
            print(f"🔍 DEBUG: Data conversion took {conversion_time:.3f}s")
            create_start = time.time()
        
        # Create GF2Matrix object
        gf2_matrix = GF2Matrix(matrix_list)
        
        if debug:
            create_time = time.time() - create_start
            print(f"🔍 DEBUG: Matrix creation took {create_time:.3f}s")
            solve_start = time.time()
        
        # Solve the linear system Ax = b over GF(2)
        solution = gf2_matrix.solve(vector_list)
        
        if debug:
            solve_time = time.time() - solve_start
            print(f"🔍 DEBUG: Library solve took {solve_time:.3f}s")
        
        if solution is None:
            if debug:
                print(f"🔍 DEBUG: Library returned no solution")
            return None
            
        return np.array(solution, dtype=int)
        
    except BaseException as e:
        # Catch BaseException to handle pyo3 PanicException (inherits BaseException, not Exception)
        error_msg = str(e)
        error_type = type(e).__name__
        
        # Handle library panics (rank-deficient matrices, etc.)
        if ("Matrix must have full rank" in error_msg or 
            "PanicException" in error_msg or 
            "PanicException" in error_type or
            "No solution" in error_msg):
            if debug:
                print(f"🔍 DEBUG: Library error/panic - falling back to numpy")
                print(f"🔍 DEBUG: Error: {error_type} - {error_msg}")
            return gf2_numpy_solve(toggle_matrix, initial_vector, debug=debug)
        
        # Re-raise KeyboardInterrupt so Ctrl+C still works
        if isinstance(e, KeyboardInterrupt):
            raise
        
        # Unexpected error - still fall back but report it
        if debug:
            print(f"🔍 DEBUG: Falling back to numpy solver due to: {error_type} - {error_msg}")
        return gf2_numpy_solve(toggle_matrix, initial_vector, debug=debug)


def gf2_numpy_solve(matrix: np.ndarray, vector: np.ndarray, debug: bool = False, timeout: int = 900) -> Optional[np.ndarray]:
    """
    Optimized numpy-based GF(2) solver for large matrices.
    Uses bitwise XOR for GF(2) arithmetic (faster than modular addition).
    Thread-safe timeout via elapsed time checks (no signals).
    """
    
    start_time = time.time()
    
    try:
        
        n = matrix.shape[0]
        if debug:
            print(f"🔍 DEBUG: Numpy solver starting with {n}x{n} matrix")
        
        # Create augmented matrix using uint8 for XOR compatibility
        augmented = np.hstack([matrix.copy().astype(np.uint8), vector.reshape(-1, 1).astype(np.uint8)])
        
        if debug:
            print(f"🔍 DEBUG: Augmented matrix created, starting elimination")
        
        # Forward elimination with numpy optimizations
        pivot_row = 0
        for col in range(n):
            # Thread-safe timeout check (every 100 columns to avoid overhead)
            if timeout > 0 and col % 100 == 0 and col > 0:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    print(f"⏰ Solver timed out after {elapsed:.1f}s")
                    return None
            
            if debug and col % 500 == 0 and col > 0:  # Less frequent debug output
                elapsed = time.time() - start_time
                print(f"🔍 DEBUG: Column {col}/{n} ({100*col/n:.1f}%) - {elapsed:.1f}s elapsed")
            
            # Find pivot efficiently
            pivot_candidates = np.where(augmented[pivot_row:, col] == 1)[0]
            if len(pivot_candidates) == 0:
                continue
            
            # Get first pivot
            pivot_idx = pivot_candidates[0] + pivot_row
            
            # Swap rows if needed
            if pivot_idx != pivot_row:
                augmented[[pivot_row, pivot_idx]] = augmented[[pivot_idx, pivot_row]]
            
            # Eliminate column efficiently using numpy broadcasting
            other_rows = np.where(augmented[:, col] == 1)[0]
            other_rows = other_rows[other_rows != pivot_row]
            
            if len(other_rows) > 0:
                augmented[other_rows] ^= augmented[pivot_row]  # XOR = addition in GF(2)
            
            pivot_row += 1
        
        if debug:
            elimination_time = time.time() - start_time
            print(f"🔍 DEBUG: Forward elimination completed in {elimination_time:.3f}s")
        
        # Check for inconsistency
        inconsistent_rows = 0
        for row in range(pivot_row, n):
            if augmented[row, n] == 1:
                inconsistent_rows += 1
        
        if inconsistent_rows > 0:
            if debug:
                print(f"🔍 DEBUG: Found {inconsistent_rows} inconsistent rows - no solution")
            return None
        
        if debug:
            print(f"🔍 DEBUG: System is consistent, starting back substitution")
            backsubst_start = time.time()
        
        # Back substitution
        solution = np.zeros(n, dtype=int)
        for row in range(min(pivot_row, n) - 1, -1, -1):
            # Find leading 1
            leading_cols = np.where(augmented[row, :n] == 1)[0]
            if len(leading_cols) > 0:
                leading_col = leading_cols[0]
                # XOR-based back substitution
                val = augmented[row, n] ^ (np.dot(augmented[row, leading_col + 1:n], solution[leading_col + 1:n]) & 1)
                solution[leading_col] = val
        
        if debug:
            backsubst_time = time.time() - backsubst_start
            total_time = time.time() - start_time
            print(f"🔍 DEBUG: Back substitution in {backsubst_time:.3f}s")
            print(f"🔍 DEBUG: Numpy solve total time: {total_time:.3f}s")
        
        return solution
        
    except Exception as e:
        print(f"Error in numpy solver: {e}")
        return None


def estimate_solve_time(num_variables: int) -> str:
    """Estimate solving time based on grid size with gf2_lin_algebra"""
    if num_variables <= 100:
        return "< 0.1 second"
    elif num_variables <= 625:  # 25x25
        return "< 0.5 seconds" 
    elif num_variables <= 2500:  # 50x50
        return "1-2 seconds"
    elif num_variables <= 10000:  # 100x100
        return "2-5 seconds"
    else:
        return "5-15 seconds"


def vector_to_grid(vector: np.ndarray, n: int) -> np.ndarray:
    """Convert 1D vector to n×n grid"""
    return vector.reshape(n, n)


def grid_to_vector(grid: np.ndarray) -> np.ndarray:
    """Convert n×n grid to 1D vector"""
    return grid.flatten()


def solve_lights_out(initial_grid: List[List[int]], debug: bool = False, competition_mode: bool = False) -> Optional[List[List[int]]]:
    """
    Solve the Lights Out puzzle for any grid shape (rectangular, octagon, cross, etc.).
    
    Args:
        initial_grid (List[List[int]]): n×m grid where 1 = light on, 0 = light off, -1 = blank cell
        debug (bool): Enable detailed debugging output
        competition_mode (bool): Enable speed optimizations for competition (less output)
        
    Returns:
        List[List[int]] or None: Grid showing which buttons to press 
                                (1 = press, 0 = don't press, -1 = blank), 
                                or None if no solution exists
    """
    
    if debug and not competition_mode:
        start_time = time.time()
        print(f"🔍 DEBUG: Starting solve for {len(initial_grid)}x{len(initial_grid[0]) if initial_grid else 0} grid")
    
    # Handle irregular grids (octagon, cross, etc.)
    try:
        if debug and not competition_mode:
            matrix_start = time.time()
        toggle_matrix, valid_positions, pos_to_index = create_toggle_matrix_for_grid(initial_grid)
        if debug and not competition_mode:
            matrix_time = time.time() - matrix_start
            print(f"🔍 DEBUG: Toggle matrix created in {matrix_time:.3f}s")
    except ValueError as e:
        print(f"Grid error: {e}")
        return None
    
    num_variables = len(valid_positions)
    
    if debug and not competition_mode:
        estimated_time = estimate_solve_time(num_variables)
        print(f"🔍 DEBUG: Matrix size: {toggle_matrix.shape}")
        print(f"🔍 DEBUG: Variables: {num_variables}")
        print(f"🔍 DEBUG: Estimated time: {estimated_time}")
    
    # Performance info only for very large grids
    if num_variables > 1000 and not competition_mode:
        estimated_time = estimate_solve_time(num_variables)
        print(f"⚠️  Large grid detected: {num_variables} variables")
        print(f"   Estimated solve time: {estimated_time}")
    
    # Create initial state vector for valid positions only
    if debug and not competition_mode:
        vector_start = time.time()
    initial_vector = np.zeros(num_variables, dtype=np.int8)  # Use int8
    for i, (row, col) in enumerate(valid_positions):
        initial_vector[i] = initial_grid[row][col]
    
    if debug and not competition_mode:
        vector_time = time.time() - vector_start
        print(f"🔍 DEBUG: Initial vector created in {vector_time:.3f}s")
        print(f"🔍 DEBUG: Lights currently on: {np.sum(initial_vector)} / {num_variables}")
    
    # Use solver with minimal output in competition mode
    if not competition_mode:
        print(f"🚀 Using gf2_lin_algebra solver ({num_variables} variables)")
    
    if debug and not competition_mode:
        solve_start = time.time()
    solution_vector = gf2_solve(toggle_matrix, initial_vector, debug=(debug and not competition_mode))
    if debug and not competition_mode:
        solve_time = time.time() - solve_start
        print(f"🔍 DEBUG: Solving completed in {solve_time:.3f}s")
    
    if solution_vector is None:
        if debug and not competition_mode:
            print(f"🔍 DEBUG: No solution exists for this grid configuration")
        if not competition_mode:
            print("❌ No solution exists for this puzzle configuration")
        return None
    
    if debug and not competition_mode:
        button_presses = np.sum(solution_vector)
        print(f"🔍 DEBUG: Solution requires {button_presses} button presses")
        grid_start = time.time()
    
    # Convert solution back to grid format (optimized)
    n_rows = len(initial_grid)
    solution_grid = [[-1 if initial_grid[i][j] == -1 else 0 for j in range(len(initial_grid[i]))] 
                     for i in range(n_rows)]
    
    # Fill in the solution for valid positions
    for i, (row, col) in enumerate(valid_positions):
        solution_grid[row][col] = int(solution_vector[i])  # Ensure int type
    
    if debug and not competition_mode:
        grid_time = time.time() - grid_start
        total_time = time.time() - start_time
        print(f"🔍 DEBUG: Grid conversion in {grid_time:.3f}s")
        print(f"🔍 DEBUG: Total solve time: {total_time:.3f}s")
    
    return solution_grid


def print_grid(grid: List[List[int]], title: str = "Grid", force: bool = False) -> None:
    """Pretty print a grid, handling irregular shapes (octagon, cross, etc.)"""
    if not force:  # Skip printing during competition for speed
        return
    print(f"\n{title}:")
    for row in grid:
        display_row = []
        for cell in row:
            if cell == -1:
                display_row.append(" ")  # Blank cell
            elif cell == 1:
                display_row.append("●")  # On/Press
            else:
                display_row.append("○")  # Off/Don't press
        print(" ".join(display_row))


def verify_solution(initial_grid: List[List[int]], 
                   solution_grid: List[List[int]]) -> bool:
    """
    Verify that the solution correctly turns off all lights.
    Supports irregular grids (octagon, cross, etc.) with -1 blank cells.
    
    Args:
        initial_grid (List[List[int]]): Initial state (1=on, 0=off, -1=blank)
        solution_grid (List[List[int]]): Proposed solution (1=press, 0=don't press, -1=blank)
        
    Returns:
        bool: True if solution is correct, False otherwise
    """
    n_rows = len(initial_grid)
    n_cols = len(initial_grid[0]) if initial_grid else 0
    
    # Create final grid starting with initial state
    final_grid = [[initial_grid[i][j] if j < len(initial_grid[i]) else -1 
                   for j in range(max(len(row) for row in initial_grid))] 
                  for i in range(n_rows)]
    
    # Apply each button press
    for i in range(n_rows):
        for j in range(len(initial_grid[i]) if i < len(initial_grid) else 0):
            # Skip blank cells and cells that can't be pressed
            if (i >= len(solution_grid) or j >= len(solution_grid[i]) or 
                solution_grid[i][j] == -1 or initial_grid[i][j] == -1):
                continue
                
            if solution_grid[i][j] == 1:  # If this button is pressed
                # Toggle the button itself (if it's not blank)
                if final_grid[i][j] != -1:
                    final_grid[i][j] = 1 - final_grid[i][j]
                
                # Toggle adjacent cells
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < n_rows and 0 <= nj < len(initial_grid[ni] if ni < len(initial_grid) else []) and
                        initial_grid[ni][nj] != -1):  # Don't toggle blank cells
                        final_grid[ni][nj] = 1 - final_grid[ni][nj]
    
    # Check if all valid lights are off
    for i in range(n_rows):
        for j in range(len(initial_grid[i]) if i < len(initial_grid) else 0):
            if initial_grid[i][j] != -1:  # Only check non-blank cells
                if final_grid[i][j] != 0:
                    return False
    
    return True


def count_button_presses(solution_grid: List[List[int]]) -> int:
    """Count total number of button presses in solution, ignoring blank cells"""
    total = 0
    for row in solution_grid:
        for cell in row:
            if cell == 1:  # Only count actual button presses (not -1 blank cells)
                total += 1
    return total


def validate_team_id(team_id: str) -> str:
    """
    Validate and fix team ID format. Returns a valid UUID.
    
    Args:
        team_id (str): Input team ID (might be invalid format)
        
    Returns:
        str: Valid UUID string
    """
    # Check if it's already a valid UUID
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    
    if uuid_pattern.match(team_id):
        return team_id
    
    # If not valid UUID format, provide feedback and use default
    print(f"⚠️  Invalid team ID format: '{team_id}'")
    print(f"   Team IDs must be UUIDs like: 9faa6787-3b95-419d-8e56-28a22ea025eb")
    default_team = "9faa6787-3b95-419d-8e56-28a22ea025eb"
    print(f"   Using default team ID: {default_team}")
    return default_team


# API Integration Functions
API_BASE_URL = "https://planetrandall.com/lightsout"

def create_new_game(team_id: str, game_type: str = "simple_5x5") -> Optional[Dict[str, Any]]:
    """
    Create a new Lights Out game via API.
    
    Args:
        team_id (str): UUID identifying the team
        game_type (str): Type of game (simple_5x5, random_5x5, etc.)
        
    Returns:
        Dict containing game data or None if error
    """
    try:
        response = _session.post(
            f"{API_BASE_URL}/api/games",
            json={"teamId": team_id, "gameType": game_type}
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            print(f"✓ Game created successfully!")
            return data["data"]
        else:
            print(f"✗ Failed to create game: {data.get('error', 'Unknown error')}")
            return None
            
    except requests.RequestException as e:
        print(f"✗ API Error creating game: {e}")
        return None


def get_game_by_id(game_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific game by ID via API.
    
    Args:
        game_id (str): UUID of the game
        
    Returns:
        Dict containing game data or None if error
    """
    try:
        response = _session.get(f"{API_BASE_URL}/api/games/{game_id}")
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            return data["data"]
        else:
            print(f"✗ Failed to get game: {data.get('error', 'Unknown error')}")
            return None
            
    except requests.RequestException as e:
        print(f"✗ API Error fetching game: {e}")
        return None


def submit_solution_to_api(game_id: str, team_id: str, solution_grid: List[List[int]], debug: bool = False) -> bool:
    """
    Submit solution to API and check if it was accepted.
    Handles irregular grids by only submitting moves for valid positions (not -1 blank cells).
    
    Args:
        game_id (str): UUID of the game
        team_id (str): UUID of the team  
        solution_grid (List[List[int]]): Grid where 1 = press button, 0 = don't press, -1 = blank
        debug (bool): Enable detailed debugging
        
    Returns:
        bool: True if solution was accepted, False otherwise
    """
    # Convert solution grid to API move format (only valid positions)
    moves = []
    for row in range(len(solution_grid)):
        for col in range(len(solution_grid[row])):
            # Only include button presses on valid cells (not -1 blank cells)
            if (col < len(solution_grid[row]) and 
                solution_grid[row][col] == 1):
                moves.append({"row": row, "col": col})
    
    if debug:
        print(f"🔍 DEBUG: Submitting {len(moves)} moves to API")
        if len(moves) <= 20:  # Show moves if not too many
            print(f"🔍 DEBUG: Moves: {moves}")
    
    payload = {"gameId": game_id, "teamId": team_id, "moves": moves}
    
    try:
        if debug:
            print(f"🔍 DEBUG: POST to {API_BASE_URL}/api/games/solution")
            print(f"🔍 DEBUG: Payload: {payload}")
            
        response = _session.post(
            f"{API_BASE_URL}/api/games/solution",
            json=payload
        )
        
        if debug:
            print(f"🔍 DEBUG: Response status: {response.status_code}")
            print(f"🔍 DEBUG: Response text: {response.text}")
            
        response.raise_for_status()
        data = response.json()
        
        success = data.get("success", False)
        message = data.get("message", "No message provided")
        
        if success:
            move_count = data.get("data", {}).get("moveCount", len(moves))
            print(f"✓ Solution accepted! {message}")
            print(f"  Move count: {move_count}")
            return True
        else:
            print(f"✗ Solution rejected: {message}")
            return False
            
    except requests.RequestException as e:
        print(f"✗ API Error submitting solution: {e}")
        if debug:
            print(f"🔍 DEBUG: Full error details: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"🔍 DEBUG: Response status: {e.response.status_code}")
                print(f"🔍 DEBUG: Response text: {e.response.text}")
        return False


def get_competition_details(competition_id: str) -> Optional[Dict[str, Any]]:
    """
    Get competition details including total games.
    
    Args:
        competition_id (str): UUID of the competition
        
    Returns:
        Dict containing competition data or None if error
    """
    try:
        response = _session.get(f"{API_BASE_URL}/api/competitions/{competition_id}")
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            return data["data"]
        else:
            print(f"✗ Failed to get competition: {data.get('error', 'Unknown error')}")
            return None
            
    except requests.RequestException as e:
        print(f"✗ API Error fetching competition: {e}")
        return None


def get_competition_game(competition_id: str, game_number: int) -> Optional[Dict[str, Any]]:
    """
    Get a specific game from a competition.
    
    Args:
        competition_id (str): UUID of the competition
        game_number (int): Game number (1-100)
        
    Returns:
        Dict containing game data or None if error
    """
    try:
        response = _session.get(f"{API_BASE_URL}/api/competitions/{competition_id}/games/{game_number}")
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            return data["data"]
        else:
            print(f"✗ Failed to get competition game {game_number}: {data.get('error', 'Unknown error')}")
            return None
            
    except requests.RequestException as e:
        print(f"✗ API Error fetching competition game {game_number}: {e}")
        return None


def submit_competition_solution(competition_id: str, game_number: int, team_id: str, solution_grid: List[List[int]], debug: bool = False) -> bool:
    """
    Submit solution for a competition game.
    Handles irregular grids by only submitting moves for valid positions (not -1 blank cells).
    
    Args:
        competition_id (str): UUID of the competition
        game_number (int): Game number (1-100)
        team_id (str): UUID of the team
        solution_grid (List[List[int]]): Grid where 1 = press button, 0 = don't press, -1 = blank
        debug (bool): Enable detailed debugging
        
    Returns:
        bool: True if solution was accepted, False otherwise
    """
    # Convert solution grid to API move format (only valid positions)
    moves = []
    for row in range(len(solution_grid)):
        for col in range(len(solution_grid[row])):
            # Only include button presses on valid cells (not -1 blank cells)
            if (col < len(solution_grid[row]) and 
                solution_grid[row][col] == 1):
                moves.append({"row": row, "col": col})
    
    if debug:
        print(f"🔍 DEBUG: Submitting {len(moves)} moves for competition game {game_number}")
    
    payload = {"teamId": team_id, "moves": moves}
    
    try:
        if debug:
            print(f"🔍 DEBUG: POST to {API_BASE_URL}/api/competitions/{competition_id}/games/{game_number}/solution")
            print(f"🔍 DEBUG: Payload: {payload}")
            
        response = _session.post(
            f"{API_BASE_URL}/api/competitions/{competition_id}/games/{game_number}/solution",
            json=payload
        )
        
        if debug:
            print(f"🔍 DEBUG: Response status: {response.status_code}")
            if not response.ok:
                print(f"🔍 DEBUG: Response text: {response.text}")
                
        response.raise_for_status()
        data = response.json()
        
        success = data.get("success", False)
        message = data.get("message", "No message provided")
        
        if success:
            print(f"✓ Competition solution accepted! {message}")
            return True
        else:
            print(f"✗ Competition solution rejected: {message}")
            return False
            
    except requests.RequestException as e:
        print(f"✗ API Error submitting competition solution: {e}")
        if debug:
            print(f"🔍 DEBUG: Full error details: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"🔍 DEBUG: Response status: {e.response.status_code}")
                print(f"🔍 DEBUG: Response text: {e.response.text}")
        return False


def solve_single_competition_game(args):
    """
    Solve a single competition game (for parallel processing).
    
    Args:
        args: Tuple of (competition_id, game_number, team_id, debug)
        
    Returns:
        Dict containing game result
    """
    competition_id, game_number, team_id, debug = args
    
    result = {
        "game_number": game_number, 
        "success": False, 
        "moves": 0, 
        "error": None,
        "status": "unknown",
        "game_type": "unknown"
    }
    
    try:
        # Get game data
        game_data = get_competition_game(competition_id, game_number)
        if not game_data:
            result["error"] = "Failed to fetch game data"
            result["status"] = "api_error"
            return result
        
        result["game_type"] = game_data.get('gameType', 'Unknown')
        grid = game_data["grid"]
        
        # Solve the puzzle (fast mode)
        solution_grid = solve_lights_out(grid, debug=False, competition_mode=True)
        if not solution_grid:
            result["status"] = "impossible"
            result["error"] = "Mathematically unsolvable"
            return result
        
        # Count moves and submit
        move_count = count_button_presses(solution_grid)
        result["moves"] = move_count
        
        # Submit to API
        success = submit_competition_solution(competition_id, game_number, team_id, solution_grid, debug=False)
        
        if success:
            result["status"] = "accepted"
            result["success"] = True
        else:
            result["status"] = "rejected"
            result["error"] = "API rejected solution"
            
        return result
        
    except BaseException as e:
        # Catch BaseException to handle pyo3 PanicException
        if isinstance(e, KeyboardInterrupt):
            raise
        result["error"] = f"Exception: {str(e)}"
        result["status"] = "error"
        return result


def solve_competition_parallel(competition_id: str, team_id: str = "9faa6787-3b95-419d-8e56-28a22ea025eb", 
                              game_range: Optional[Tuple[int, int]] = None, debug: bool = False,
                              max_workers: Optional[int] = None) -> Dict[str, Any]:
    """
    PARALLEL version of solve_competition for maximum speed.
    Uses ThreadPoolExecutor (optimal for I/O-bound HTTP work) with connection pooling.
    
    Args:
        competition_id (str): UUID of the competition
        team_id (str): Team ID for API requests
        game_range (Optional[Tuple[int, int]]): Range of games to solve
        debug (bool): Enable debugging (recommended: False for max speed)
        max_workers (Optional[int]): Number of parallel threads (default: 2x CPU count, capped at 20)
        
    Returns:
        Dict with comprehensive results
    """
    print(f"\n{'='*70}")
    print("🚀 PARALLEL LIGHTS OUT COMPETITION SOLVER")
    print(f"{'='*70}")
    print(f"Competition ID: {competition_id}")
    print(f"Team ID: {team_id}")
    
    # Get competition details
    competition_data = get_competition_details(competition_id)
    if not competition_data:
        return {"success": False, "error": "Failed to fetch competition details"}
    
    total_games = competition_data.get("totalGames", 100)
    competition_name = competition_data.get("name", "Unknown Competition")
    
    # Determine worker count — more threads for I/O-bound work
    if max_workers is None:
        max_workers = min(mp.cpu_count() * 2, 20)  # 2x CPU cores, cap at 20
    
    print(f"Competition: {competition_name}")
    print(f"Total Games: {total_games}")
    print(f"🔥 Parallel Workers: {max_workers}")
    
    # Determine game range
    if game_range:
        start_game, end_game = game_range
        start_game = max(1, start_game)
        end_game = min(total_games, end_game)
        print(f"Solving games {start_game} to {end_game}")
    else:
        start_game, end_game = 1, total_games
        print(f"Solving ALL games (1 to {total_games})")
    
    # Prepare arguments for parallel processing
    game_args = [(competition_id, game_num, team_id, debug) 
                 for game_num in range(start_game, end_game + 1)]
    
    # Initialize results tracking
    results = {
        "competition_id": competition_id,
        "team_id": team_id,
        "total_attempted": len(game_args),
        "successful_submissions": 0,
        "failed_submissions": 0,
        "impossible_games": 0,
        "api_errors": 0,
        "total_moves_submitted": 0,
        "game_details": [],
        "impossible_game_numbers": [],
        "successful_game_numbers": [],
        "failed_game_numbers": [],
        "game_type_breakdown": {
            "simple_9x9": {"attempted": 0, "successful": 0, "impossible": 0},
            "octagon_15": {"attempted": 0, "successful": 0, "impossible": 0},
            "cross_15": {"attempted": 0, "successful": 0, "impossible": 0}
        }
    }
    
    print(f"\n{'-'*50}")
    print("🚀 Starting parallel processing...")
    print(f"{'-'*50}")
    
    start_time = time.time()
    
    # Execute games in parallel using threads (optimal for I/O-bound API work)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all games
        future_to_game = {executor.submit(solve_single_competition_game, args): args[1] 
                         for args in game_args}
        
        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_game):
            game_number = future_to_game[future]
            completed += 1
            
            try:
                result = future.result()
                results["game_details"].append(result)
                
                # Update counters
                if result["status"] == "accepted":
                    results["successful_submissions"] += 1
                    results["successful_game_numbers"].append(game_number)
                    results["total_moves_submitted"] += result["moves"]
                    print(f"✅ Game {game_number} - SUCCESS! ({result['moves']} moves) [{completed}/{len(game_args)}]")
                elif result["status"] == "impossible":
                    results["impossible_games"] += 1
                    results["impossible_game_numbers"].append(game_number)
                    print(f"❌ Game {game_number} - IMPOSSIBLE [{completed}/{len(game_args)}]")
                elif result["status"] == "rejected":
                    results["failed_submissions"] += 1
                    results["failed_game_numbers"].append(game_number)
                    print(f"❌ Game {game_number} - REJECTED [{completed}/{len(game_args)}]")
                else:
                    results["api_errors"] += 1
                    print(f"💥 Game {game_number} - API ERROR: {result.get('error', 'Unknown')} [{completed}/{len(game_args)}]")
                
                # Update game type breakdown
                game_type = result.get("game_type", "unknown")
                if game_type in results["game_type_breakdown"]:
                    results["game_type_breakdown"][game_type]["attempted"] += 1
                    if result["status"] == "accepted":
                        results["game_type_breakdown"][game_type]["successful"] += 1
                    elif result["status"] == "impossible":
                        results["game_type_breakdown"][game_type]["impossible"] += 1
                
                # Progress update every 25% completion
                if completed % max(1, len(game_args) // 4) == 0:
                    progress = (completed / len(game_args)) * 100
                    elapsed = time.time() - start_time
                    print(f"🚀 Progress: {progress:.0f}% - {results['successful_submissions']} successes - {elapsed:.1f}s")
                    
            except Exception as exc:
                print(f"💥 Game {game_number} generated an exception: {exc}")
                results["api_errors"] += 1
                results["game_details"].append({
                    "game_number": game_number,
                    "success": False,
                    "error": f"Exception: {str(exc)}",
                    "status": "error"
                })
    
    elapsed_time = time.time() - start_time
    
    # Print comprehensive final summary
    print(f"\n{'='*70}")
    print("🏆 PARALLEL COMPETITION RESULTS")
    print(f"{'='*70}")
    print(f"Competition: {competition_name}")
    print(f"⏱️  Total Time: {elapsed_time:.1f} seconds")
    print(f"🚀 Parallel Workers: {max_workers}")
    print(f"Games Attempted: {results['total_attempted']}")
    print(f"✅ Successful Submissions: {results['successful_submissions']}")
    print(f"❌ Failed Submissions: {results['failed_submissions']}")
    print(f"🚫 Impossible Games: {results['impossible_games']}")
    print(f"💥 API Errors: {results['api_errors']}")
    print(f"📊 Total Moves Submitted: {results['total_moves_submitted']}")
    
    if results['successful_submissions'] > 0:
        avg_moves = results['total_moves_submitted'] / results['successful_submissions']
        print(f"📈 Average Moves per Success: {avg_moves:.1f}")
        throughput = results['successful_submissions'] / elapsed_time * 60
        print(f"🎯 Solve Rate: {throughput:.1f} games/minute")
    
    success_rate = (results['successful_submissions'] / results['total_attempted']) * 100 if results['total_attempted'] > 0 else 0
    print(f"🔥 Success Rate: {success_rate:.1f}%")
    
    return results


def solve_competition(competition_id: str, team_id: str = "9faa6787-3b95-419d-8e56-28a22ea025eb", 
                     game_range: Optional[Tuple[int, int]] = None, debug: bool = False, 
                     parallel: bool = True, max_workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Solve all games in a competition with optional parallel processing.
    
    Args:
        competition_id (str): UUID of the competition
        team_id (str): Team ID for API requests
        game_range (Optional[Tuple[int, int]]): Range of games to solve
        debug (bool): Enable debugging
        parallel (bool): Use parallel processing for maximum speed
        max_workers (Optional[int]): Number of parallel processes
        
    Returns:
        Dict with comprehensive results
    """
    if parallel:
        return solve_competition_parallel(competition_id, team_id, game_range, debug, max_workers)
    else:
        return solve_competition_sequential(competition_id, team_id, game_range, debug)


def solve_competition_sequential(competition_id: str, team_id: str = "9faa6787-3b95-419d-8e56-28a22ea025eb", 
                     game_range: Optional[Tuple[int, int]] = None, debug: bool = False) -> Dict[str, Any]:
    """
    Sequential version of solve_competition for compatibility.
    
    Args:
        competition_id (str): UUID of the competition
        team_id (str): Team ID for API requests (must be valid UUID)
        game_range (Optional[Tuple[int, int]]): Range of games to solve (start, end), or None for all 100
        debug (bool): Enable detailed debugging output
        
    Returns:
        Dict with comprehensive summary of results
    """
    print(f"\n{'='*70}")
    print("LIGHTS OUT COMPETITION SOLVER")
    print(f"{'='*70}")
    print(f"Competition ID: {competition_id}")
    print(f"Team ID: {team_id}")
    
    # Get competition details
    competition_data = get_competition_details(competition_id)
    if not competition_data:
        return {"success": False, "error": "Failed to fetch competition details"}
    
    total_games = competition_data.get("totalGames", 100)
    competition_name = competition_data.get("name", "Unknown Competition")
    
    print(f"Competition: {competition_name}")
    print(f"Total Games: {total_games}")
    
    # Determine game range
    if game_range:
        start_game, end_game = game_range
        start_game = max(1, start_game)
        end_game = min(total_games, end_game)
        print(f"Solving games {start_game} to {end_game}")
    else:
        start_game, end_game = 1, total_games
        print(f"Solving ALL games (1 to {total_games})")
    
    # Track results with detailed categorization
    results = {
        "competition_id": competition_id,
        "team_id": team_id,
        "total_attempted": 0,
        "successful_submissions": 0,     # Solutions accepted by API
        "failed_submissions": 0,        # Solutions rejected by API (wrong)
        "impossible_games": 0,          # Games with no mathematical solution
        "already_submitted": 0,         # Games already solved (409 error)
        "api_errors": 0,               # Other API errors (404, etc.)
        "total_moves_submitted": 0,     # Sum of all successful move counts
        "game_details": [],            # Detailed results for each game
        "impossible_game_numbers": [],  # List of impossible game numbers
        "successful_game_numbers": [], # List of successfully solved games
        "failed_game_numbers": [],     # List of games with wrong solutions
        "game_type_breakdown": {       # Results by game type
            "simple_9x9": {"attempted": 0, "successful": 0, "impossible": 0},
            "octagon_15": {"attempted": 0, "successful": 0, "impossible": 0},
            "cross_15": {"attempted": 0, "successful": 0, "impossible": 0}
        }
    }
    
    print(f"\n{'-'*50}")
    print("Starting to solve games...")
    print(f"{'-'*50}")
    
    # Solve each game in the range
    for game_number in range(start_game, end_game + 1):
        print(f"\n🎮 Game {game_number}/{total_games}")
        print(f"{'='*30}")
        
        results["total_attempted"] += 1
        game_result = {"game_number": game_number, "success": False, "moves": 0, "error": None}
        
        # Get game data
        game_data = get_competition_game(competition_id, game_number)
        if not game_data:
            game_result["error"] = "Failed to fetch game data"
            results["api_errors"] += 1
            results["game_details"].append(game_result)
            continue
        
        game_type = game_data.get('gameType', 'Unknown')
        print(f"Game Type: {game_type}")
        print(f"Grid Size: {game_data['size']}×{game_data['size']}")
        
        # Check for irregular grid shapes (minimal logging)
        grid = game_data["grid"]
        has_blanks = any(-1 in row for row in grid)
        if has_blanks and debug:
            valid_cells = sum(1 for row in grid for cell in row if cell != -1)
            print(f"⚠️  Irregular {game_type}: {valid_cells} valid cells")
        
        # Solve the puzzle - FAST MODE (no printing, no verification)
        solution_grid = solve_lights_out(grid, debug=debug, competition_mode=True)
        if not solution_grid:
            print(f"❌ Game {game_number} - Mathematically impossible")
            game_result["status"] = "impossible"
            game_result["error"] = "Mathematically unsolvable"
            results["impossible_games"] += 1
            results["impossible_game_numbers"].append(game_number)
            if game_type in results["game_type_breakdown"]:
                results["game_type_breakdown"][game_type]["impossible"] += 1
            results["game_details"].append(game_result)
            continue
        
        # Count moves and submit immediately (no local verification - trust solver)
        move_count = count_button_presses(solution_grid)
        game_result["moves"] = move_count
        
        # Submit to API
        success = submit_competition_solution(competition_id, game_number, team_id, solution_grid, debug=debug)
        
        if success:
            print(f"✅ Game {game_number} - SUCCESS! ({move_count} moves)")
            game_result["status"] = "accepted"
            game_result["success"] = True
            results["successful_submissions"] += 1
            results["successful_game_numbers"].append(game_number)
            results["total_moves_submitted"] += move_count
            if game_type in results["game_type_breakdown"]:
                results["game_type_breakdown"][game_type]["successful"] += 1
        else:
            print(f"❌ Game {game_number} - REJECTED")
            game_result["status"] = "rejected"
            game_result["error"] = "API rejected solution"
            results["failed_submissions"] += 1
            results["failed_game_numbers"].append(game_number)
        
        results["game_details"].append(game_result)
        
        # Progress update every 10 games
        if game_number % 10 == 0:
            progress = (game_number - start_game + 1) / (end_game - start_game + 1) * 100
            print(f"   Progress: {progress:.0f}% - {results['successful_submissions']} successes")
    
    # Print comprehensive final summary
    print(f"\n{'='*70}")
    print("🏆 COMPETITION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Competition: {competition_name}")
    print(f"Games Attempted: {results['total_attempted']}")
    print(f"✅ Successful Submissions: {results['successful_submissions']}")
    print(f"❌ Failed Submissions: {results['failed_submissions']}")
    print(f"🚫 Impossible Games: {results['impossible_games']}")
    print(f"⚠️  Already Submitted: {results['already_submitted']}")
    print(f"💥 API Errors: {results['api_errors']}")
    print(f"📊 Total Moves Submitted: {results['total_moves_submitted']}")
    
    if results['successful_submissions'] > 0:
        avg_moves = results['total_moves_submitted'] / results['successful_submissions']
        print(f"📈 Average Moves per Success: {avg_moves:.1f}")
    
    success_rate = (results['successful_submissions'] / results['total_attempted']) * 100 if results['total_attempted'] > 0 else 0
    print(f"🎯 Success Rate: {success_rate:.1f}%")
    
    return results


def solve_api_game(game_id: str = None, team_id: str = "9faa6787-3b95-419d-8e56-28a22ea025eb", 
                   create_new: bool = False, game_type: str = "simple_5x5", debug: bool = False) -> bool:
    """
    Complete workflow: get/create game, solve it, and submit solution.
    
    Args:
        game_id (str, optional): Specific game ID to solve, or None to create new game
        team_id (str): Team ID for API requests
        create_new (bool): Whether to create a new game instead of using game_id
        game_type (str): Type of game to create if create_new=True
        
    Returns:
        bool: True if successfully solved and submitted
    """
    print(f"\n{'='*60}")
    print("LIGHTS OUT API SOLVER")
    print(f"{'='*60}")
    
    # Step 1: Get or create game
    if create_new or game_id is None:
        print("Creating new game...")
        game_data = create_new_game(team_id, game_type)
        if not game_data:
            return False
        game_id = game_data["gameId"]
        print(f"Game ID: {game_id}")
    else:
        print(f"Fetching game: {game_id}")
        game_data = get_game_by_id(game_id)
        if not game_data:
            return False
    
    # Step 2: Display game info
    print(f"Game Type: {game_data.get('gameType', 'Unknown')}")
    print(f"Grid Size: {game_data['size']}×{game_data['size']}")
    print(f"Already solved: {game_data.get('solved', False)}")
    
    # Step 3: Solve the puzzle
    grid = game_data["grid"]
    print_grid(grid, "Initial state")
    
    solution_grid = solve_lights_out(grid, debug=debug)
    if not solution_grid:
        print("✗ No solution exists for this configuration!")
        return False
    
    print_grid(solution_grid, "Solution (buttons to press)")
    
    # Step 4: Verify solution locally first
    is_correct = verify_solution(grid, solution_grid)
    move_count = count_button_presses(solution_grid)
    
    if not is_correct:
        print("✗ Local verification failed!")
        return False
        
    print(f"✓ Local verification passed")
    print(f"Total button presses: {move_count}")
    
    # Step 5: Submit to API
    print(f"\nSubmitting solution to API...")
    success = submit_solution_to_api(game_id, team_id, solution_grid, debug=debug)
    
    return success


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
    import sys
    
    try:
        # Check command line arguments for API usage
        if len(sys.argv) > 1:
            if sys.argv[1] == "--create-game":
                # Create and solve a new game
                team_id = sys.argv[2] if len(sys.argv) > 2 else "9faa6787-3b95-419d-8e56-28a22ea025eb"
                game_type = sys.argv[3] if len(sys.argv) > 3 else "simple_5x5"
                solve_api_game(create_new=True, team_id=team_id, game_type=game_type)
                
            elif sys.argv[1] == "--solve-game":
                # Solve existing game by ID
                if len(sys.argv) < 3:
                    print("Usage: python Lights_out.py --solve-game GAME_ID [TEAM_ID] [--debug]")
                    sys.exit(1)
                game_id = sys.argv[2]
                
                # Check for debug flag
                debug = "--debug" in sys.argv
                if debug:
                    print("🔍 Debug mode enabled")
                
                raw_team_id = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "--debug" else "9faa6787-3b95-419d-8e56-28a22ea025eb"
                team_id = validate_team_id(raw_team_id)
                solve_api_game(game_id=game_id, team_id=team_id, debug=debug)
                
            elif sys.argv[1] == "--hackday":
                # Simple hackday command - just needs competition ID
                if len(sys.argv) < 3:
                    print("Usage: python Lights_out.py --hackday COMPETITION_ID [--debug] [--sequential] [--workers N]")
                    print("")
                    print("This will solve all 100 games in the competition using your default team ID.")
                    print("Options:")
                    print("  --debug      Enable detailed debug output")
                    print("  --sequential Use sequential processing (default: parallel)")
                    print("  --workers N  Set number of parallel workers (default: CPU count)")
                    print("At the end, you'll get a comprehensive report of all results.")
                    sys.exit(1)
                    
                competition_id = sys.argv[2]
                debug = "--debug" in sys.argv
                sequential = "--sequential" in sys.argv
                parallel = not sequential
                team_id = "9faa6787-3b95-419d-8e56-28a22ea025eb"  # Your team ID
                
                # Parse worker count
                max_workers = None
                try:
                    if "--workers" in sys.argv:
                        worker_idx = sys.argv.index("--workers") + 1
                        if worker_idx < len(sys.argv):
                            max_workers = int(sys.argv[worker_idx])
                            print(f"🔥 Custom worker count: {max_workers}")
                except (ValueError, IndexError):
                    print("⚠️  Invalid worker count, using default")
                
                print("\n🚀 HACKDAY COMPETITION SOLVER STARTING!")
                if parallel:
                    print("🔥 PARALLEL MODE - Maximum Performance")
                    if max_workers:
                        print(f"🔥 Workers: {max_workers}")
                    else:
                        print(f"🔥 Workers: {mp.cpu_count()} (auto-detect)")
                else:
                    print("⚠️  Sequential mode (slower)")
                print(f"Competition ID: {competition_id}")
                print(f"Team ID: {team_id}")
                print("This will attempt all 100 games in the competition...\n")
                
                results = solve_competition(competition_id, team_id, debug=debug, 
                                          parallel=parallel, max_workers=max_workers)
                
                print("\n🎯 HACKDAY COMPETITION COMPLETE!")
                print("Check the summary above for your results.")
                
            elif sys.argv[1] == "--solve-competition":
                # Solve all games in a competition
                if len(sys.argv) < 3:
                    print("Usage: python Lights_out.py --solve-competition COMPETITION_ID [TEAM_ID] [START_GAME] [END_GAME] [--debug] [--sequential] [--workers N]")
                    sys.exit(1)
                competition_id = sys.argv[2]
                
                # Check for debug and parallel flags
                debug = "--debug" in sys.argv
                sequential = "--sequential" in sys.argv
                parallel = not sequential
                if debug:
                    print("🔍 Debug mode enabled")
                if parallel:
                    print("🚀 Parallel mode enabled")
                
                # Parse worker count
                max_workers = None
                try:
                    if "--workers" in sys.argv:
                        worker_idx = sys.argv.index("--workers") + 1
                        if worker_idx < len(sys.argv):
                            max_workers = int(sys.argv[worker_idx])
                            print(f"🔥 Custom worker count: {max_workers}")
                except (ValueError, IndexError):
                    print("⚠️  Invalid worker count, using default")

                raw_team_id = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "9faa6787-3b95-419d-8e56-28a22ea025eb"
                team_id = validate_team_id(raw_team_id)
                
                # Optional game range
                game_range = None
                if len(sys.argv) >= 6:
                    try:
                        # Find numeric arguments (skipping flags)
                        numeric_args = [arg for arg in sys.argv[3:] if not arg.startswith("--") and arg.isdigit()]
                        if len(numeric_args) >= 2:
                            start_game = int(numeric_args[0])
                            end_game = int(numeric_args[1])
                            game_range = (start_game, end_game)
                            print(f"Solving games {start_game} to {end_game}")
                    except ValueError:
                        print("Invalid game range. Using all games.")
                
                solve_competition(competition_id, team_id, game_range, debug=debug, 
                                parallel=parallel, max_workers=max_workers)
                
            elif sys.argv[1] == "--help":
                print("Lights Out Solver - Usage:")
                print("  python Lights_out.py                    # Run local examples")
                print("  python Lights_out.py --hackday COMP_ID [OPTIONS]           # HACKDAY: Solve all 100 games")
                print("  python Lights_out.py --create-game [TEAM_ID] [GAME_TYPE]  # Create & solve new game")
                print("  python Lights_out.py --solve-game GAME_ID [TEAM_ID] [--debug]      # Solve existing game")
                print("  python Lights_out.py --solve-competition COMP_ID [TEAM_ID] [START] [END] [OPTIONS]  # Solve competition")  
                print("  python Lights_out.py --test-performance # Test solver performance")
                print("  python Lights_out.py --help             # Show this help")
                print("")
                print("🎯 FOR HACKDAY: Use --hackday COMPETITION_ID")
                print("")
                print("🚀 PARALLEL PROCESSING OPTIONS:")
                print("  --sequential   Use sequential processing (slower, default is parallel)")
                print("  --workers N    Set number of parallel workers (default: CPU count)")
                print("  --debug        Enable detailed debug output (reduces performance)")
                print("")
                print("Game types: simple_5x5, random_5x5, simple_9x9, random_9x9")
                print("Default team ID: 9faa6787-3b95-419d-8e56-28a22ea025eb")
                print("")
                print("Competition examples:")
                print("  python Lights_out.py --hackday abc123-def456-789ghi               # FAST: Parallel mode (recommended)")
                print("  python Lights_out.py --hackday abc123-def456-789ghi --workers 8   # Custom worker count")
                print("  python Lights_out.py --hackday abc123-def456-789ghi --sequential  # Sequential mode")
                print("  python Lights_out.py --hackday abc123-def456-789ghi --debug       # With debug (slower)")
                print("  python Lights_out.py --solve-competition abc123-def456-789ghi --workers 16    # 16 parallel workers")
                print("  python Lights_out.py --solve-competition abc123-def456-789ghi TEAM_ID 1 10    # Games 1-10 only")
                print("")
                print("🔥 PERFORMANCE TIPS:")
                print(f"  • Default parallel workers: {mp.cpu_count()} (your CPU cores)")
                print("  • More workers = faster solving (up to ~16 workers for this API)")
                print("  • Use --sequential only if you have issues with parallel mode")
                print("  • Avoid --debug during competitions for maximum speed")
                print("")
                print("Debug mode:")
                print("  Add --debug flag for detailed solving information and performance timing")
                print("")
                print("Performance:")
                print("  Optimized for all grid sizes using gf2-lin-algebra with parallel processing")
                
            elif sys.argv[1] == "--test-performance":
                # Test performance with different grid sizes                
                print("PERFORMANCE TEST - Solver Speed Comparison")
                print("="*50)
                
                test_sizes = [5, 10, 15, 20, 30, 50, 75, 100]
                
                for size in test_sizes:
                    print(f"\nTesting {size}×{size} grid...")
                    
                    # Create test grid (all lights on)
                    test_grid = [[1 for _ in range(size)] for _ in range(size)]
                    
                    start_time = time.time()
                    solution = solve_lights_out(test_grid)
                    end_time = time.time()
                    
                    solve_time = end_time - start_time
                    num_vars = size * size
                    
                    if solution:
                        moves = count_button_presses(solution)
                        print(f"  ✓ Solved in {solve_time:.3f}s ({num_vars} variables, {moves} moves)")
                    else:
                        print(f"  ✗ No solution found in {solve_time:.3f}s")
                
                print(f"\n🚀 Using gf2-lin-algebra for optimal performance on all grid sizes!")
        else:
            # Default: run local examples
            example_usage()
            
            print("\n" + "="*60)
            print("API USAGE EXAMPLES:")
            print("="*60)
            print("Create and solve a new game:")
            print("  python Lights_out.py --create-game")
            print("")
            print("Solve existing game by ID:")
            print("  python Lights_out.py --solve-game YOUR_GAME_ID")
            print("")
            print("Solve entire competition:")
            print("  python Lights_out.py --solve-competition COMPETITION_ID")
            print("")
            print("Solve specific games in competition (e.g., games 1-10):")
            print("  python Lights_out.py --solve-competition COMP_ID TEAM_ID 1 10")
            print("")
            print("Use --help for more options")
            
    except ImportError as e:
        print(f"Error: {e}")
        print("Required packages: numpy, requests")
        print("Install with: pip install numpy requests")
        print("")
        print("High-performance solving ready for all grid sizes!")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
