# Lights Out Solver - API Integration

A mathematical solver for the Lights Out puzzle game using linear algebra over GF(2). Supports both local solving and integration with the Lights Out API.

## Overview

This solver uses Gaussian elimination in GF(2) to find optimal solutions to Lights Out puzzles by solving the linear system: **Toggle_Matrix × Solution_Vector ≡ Initial_State (mod 2)**

**✨ Supports Multiple Grid Shapes:**
- ✅ **Rectangular grids** (3×3, 5×5, 9×9, etc.)
- ✅ **Octagon grids** (octagon_15, octagon_30, etc.)  
- ✅ **Cross-shaped grids** (cross_15, cross_100, etc.)
- ✅ **Any irregular shape** with blank cells (-1 values)

## Requirements

```bash
pip install numpy requests gf2-lin-algebra
```

- **numpy** — matrix operations and linear algebra
- **requests** — HTTP communication with the Lights Out API
- **gf2-lin-algebra** — high-performance GF(2) solver (Rust backend) used for grids up to 500 variables; the solver falls back to an optimised numpy implementation for larger matrices

## Usage

### 1. Local Examples (Default)
Run the script without arguments to see local puzzle examples:

```bash
python Lights_out.py
```

This will demonstrate the solver with:
- 3×3 puzzle examples
- 5×5 puzzle examples  
- Solution verification
- Button press counting

### 2. API Integration

#### Create and Solve New Game
Create a new random game via the API and solve it:

```bash
# Basic usage (uses default team ID and simple_5x5)
python Lights_out.py --create-game

# With custom team ID
python Lights_out.py --create-game YOUR_TEAM_ID

# With custom team ID and game type
python Lights_out.py --create-game YOUR_TEAM_ID random_9x9
```

**Parameters:**
- `TEAM_ID` (optional): Your team UUID. Default: `9faa6787-3b95-419d-8e56-28a22ea025eb`
- `GAME_TYPE` (optional): Type of game to create. Default: `simple_5x5`

**Available Game Types:**
- `simple_5x5` - Guaranteed solvable 5×5 grid
- `random_5x5` - Random 5×5 grid (may not be solvable)
- `simple_9x9` - Guaranteed solvable 9×9 grid  
- `random_9x9` - Random 9×9 grid (may not be solvable)

#### Solve Existing Game
Solve a specific game by providing its game ID:

```bash
# Basic usage (uses default team ID)
python Lights_out.py --solve-game GAME_ID

# With custom team ID
python Lights_out.py --solve-game GAME_ID YOUR_TEAM_ID

# With debug output
python Lights_out.py --solve-game GAME_ID --debug
```

**Parameters:**
- `GAME_ID` (required): The UUID of the game to solve
- `TEAM_ID` (optional): Your team UUID for solution submission
- `--debug` (optional): Enable detailed solver diagnostics and timing

**Example:**
```bash
python Lights_out.py --solve-game 23371ece-5f16-43dd-bf80-5fca6b139748
```

#### HackDay Quick Command
Simplified command to solve all games in a competition. Uses parallel processing by default for maximum speed:

```bash
# Solve all 100 games (parallel, default workers)
python Lights_out.py --hackday COMPETITION_ID

# Custom number of parallel workers
python Lights_out.py --hackday COMPETITION_ID --workers 8

# Sequential mode (slower, use if parallel causes issues)
python Lights_out.py --hackday COMPETITION_ID --sequential

# With debug output (reduces performance)
python Lights_out.py --hackday COMPETITION_ID --debug
```

**Parameters:**
- `COMPETITION_ID` (required): The UUID of the competition
- `--workers N` (optional): Number of parallel threads (default: 2× CPU cores, capped at 20)
- `--sequential` (optional): Disable parallel processing and solve games one at a time
- `--debug` (optional): Enable detailed debug output (slower)

#### Solve Competition
Solve all games in a competition with more control over team ID and game range:

```bash
# Solve all games in competition (parallel by default)
python Lights_out.py --solve-competition COMPETITION_ID

# With custom team ID
python Lights_out.py --solve-competition COMPETITION_ID YOUR_TEAM_ID

# Solve specific range of games (e.g., games 1-10 only)
python Lights_out.py --solve-competition COMPETITION_ID YOUR_TEAM_ID 1 10

# With 16 parallel workers
python Lights_out.py --solve-competition COMPETITION_ID --workers 16

# Sequential mode
python Lights_out.py --solve-competition COMPETITION_ID --sequential
```

**Parameters:**
- `COMPETITION_ID` (required): The UUID of the competition
- `TEAM_ID` (optional): Your team UUID for solution submission
- `START_GAME` (optional): First game number to solve (default: 1)
- `END_GAME` (optional): Last game number to solve (default: all games)
- `--workers N` (optional): Number of parallel threads
- `--sequential` (optional): Use sequential processing instead of parallel
- `--debug` (optional): Enable detailed debug output

### 3. Parallelisation

Competition solving uses **parallel processing by default** to maximise throughput. Games are fetched, solved, and submitted concurrently using a thread pool, which is optimal for the I/O-bound HTTP work involved.

| Option | Behaviour |
|---|---|
| *(default)* | Parallel with `2 × CPU cores` threads (capped at 20) |
| `--workers N` | Set a custom number of parallel threads |
| `--sequential` | Process games one at a time |

The solver also uses **HTTP connection pooling** (persistent sessions with up to 16 pooled connections) to avoid the overhead of establishing a new TCP connection for every API call.

**Performance tips:**
- More workers generally means faster solving (up to ~16 workers for this API)
- Use `--sequential` only if you encounter issues with parallel mode
- Avoid `--debug` during competitions for maximum speed

### 4. Test Solver Performance
Benchmark the solver across different grid sizes:

```bash
python Lights_out.py --test-performance
```

This tests grids from 5×5 up to 100×100 and reports solve times.

### 5. Help
Show all available commands and options:

```bash
python Lights_out.py --help
```

**Examples:**
```bash
# Solve all 100 games in a competition
python Lights_out.py --solve-competition abc123-def456-789ghi

# Solve first 20 games only
python Lights_out.py --solve-competition abc123-def456-789ghi my-team-id 1 20

# Solve games 50-60 for testing
python Lights_out.py --solve-competition abc123-def456-789ghi my-team-id 50 60
```

#### Help
Display usage information:

```bash
python Lights_out.py --help
```

## API Configuration

The solver connects to: `https://planetrandall.com/lightsout/api/games`

**API Endpoints Used:**
- `POST /api/games` - Create new game
- `GET /api/games/{gameId}` - Retrieve game state
- `POST /api/games/solution` - Submit solution
- `GET /api/competitions/{competitionId}` - Get competition details
- `GET /api/competitions/{competitionId}/games/{gameNumber}` - Get competition game
- `POST /api/competitions/{competitionId}/games/{gameNumber}/solution` - Submit competition solution

## Output Format

### Local Examples
```
Initial state:
● ○ ●
○ ● ○  
● ○ ●

Solution (buttons to press):
○ ● ○
● ○ ●
○ ● ○

Solution verified: ✓
Total button presses: 4
```

### Irregular Grid Examples (Octagon/Cross)
```
Initial state (octagon_15):
    ● ○ ●
  ● ○ ● ○ ●
● ○ ● ○ ● ○ ●
  ● ○ ● ○ ●  
    ● ○ ●

Solution (buttons to press):
       ○
  ○ ● ○ ○ ●
○   ○ ● ○   ○
  ○ ○ ● ○ ○
       ○

⚠️ Irregular grid detected (octagon_15 shape)
Valid cells: 37/49
Solution verified: ✓
Total button presses: 12
```

**Legend:**
- ● = Light ON / Press button
- ○ = Light OFF / Don't press  
- (space) = Blank cell (cannot be pressed)

### API Integration
```
============================================================
LIGHTS OUT API SOLVER
============================================================
Creating new game...
✓ Game created successfully!
Game ID: abc123-def456-789ghi
Game Type: simple_5x5
Grid Size: 5×5
Already solved: False

Initial state:
● ○ ● ○ ○
○ ● ○ ● ○
● ○ ● ○ ●
○ ● ○ ● ○
○ ○ ● ○ ●

Solution (buttons to press):
○ ● ○ ○ ○
● ○ ● ○ ●
○ ● ○ ● ○
○ ○ ● ○ ○
○ ○ ○ ● ○

✓ Local verification passed
Total button presses: 9

Submitting solution to API...
✓ Solution accepted! Solution is correct!
Move count: 9
```

### Competition Solving
```
======================================================================
LIGHTS OUT COMPETITION SOLVER
======================================================================
Competition ID: abc123-def456-789ghi
Team ID: 550e8400-e29b-41d4-a716-446655440000
Competition: Hackathon Championship 2026
Total Games: 100
Solving ALL games (1 to 100)

--------------------------------------------------
Starting to solve games...
--------------------------------------------------

🎮 Game 1/100
==============================
Game Type: simple_9x9
Grid Size: 9×9

[Game solving details...]

✓ Game 1 - Local verification passed
  Move count: 15
🎉 Game 1 - SUCCESS!

🎮 Game 2/100  
==============================
Game Type: octagon_15
Grid Size: 15×15
⚠️  Irregular grid detected (octagon_15 shape)
   Valid cells: 181/225

[Octagon grid display with spaces for blank cells...]

✓ Game 2 - Local verification passed
  Move count: 23
🎉 Game 2 - SUCCESS!

🎮 Game 3/100
==============================
[continues for all games...]

======================================================================
COMPETITION RESULTS SUMMARY
======================================================================
Competition: Hackathon Championship 2026
Games Attempted: 100
✅ Successful Solves: 97
❌ Failed Solves: 2
🚫 API Rejections: 1
🔒 Unsolvable Games: 0
📊 Total Moves: 1847
📈 Average Moves per Solved Game: 19.0
🎯 Success Rate: 97.0%
```

## How It Works

1. **Grid Analysis**: Identifies valid positions (excluding -1 blank cells in irregular grids)
2. **Toggle Matrix Generation**: Creates a matrix representing how each button press affects only valid positions
3. **Linear System**: Formulates the puzzle as Ax ≡ b (mod 2) where:
   - A = toggle matrix (for valid positions only)
   - x = solution vector (buttons to press)
   - b = initial state vector (for valid positions only)
4. **GF(2) Gaussian Elimination**: Solves the system using XOR operations
5. **Solution Mapping**: Maps solution back to original grid shape (including -1 placeholders)
6. **Verification**: Validates the solution locally before API submission
7. **API Submission**: Converts solution grid to move format (excluding blank cells)

## Error Handling

- **Missing Dependencies**: Script checks for numpy/requests and provides install instructions
- **API Connectivity**: Handles network errors and invalid responses
- **Invalid Game IDs**: Provides clear error messages for non-existent games
- **Unsolvable Puzzles**: Detects and reports when no solution exists

## Examples

### Quick Start
```bash
# See local examples
python Lights_out.py

# Create and solve a new API game
python Lights_out.py --create-game

# Get help
python Lights_out.py --help
```

### Advanced Usage
```bash
# Create specific game type with your team ID
python Lights_out.py --create-game 12345678-1234-1234-1234-123456789abc random_9x9

# Solve someone else's game
python Lights_out.py --solve-game 87654321-4321-4321-4321-cba987654321 my-team-id

# Solve entire competition
python Lights_out.py --solve-competition comp123-abc456-def789 my-team-id

# Solve subset of competition for testing (games 1-5)
python Lights_out.py --solve-competition comp123-abc456-def789 my-team-id 1 5

# Solve specific range in middle of competition (games 50-75)
python Lights_out.py --solve-competition comp123-abc456-def789 my-team-id 50 75
```

## Mathematical Background

Based on "Turning Lights Out with Linear Algebra" by Anderson & Feil (1998), this solver treats each puzzle as a system of linear equations in GF(2). The approach guarantees finding optimal solutions when they exist and can detect unsolvable configurations.