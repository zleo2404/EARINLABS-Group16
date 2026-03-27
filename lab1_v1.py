from collections import deque
import time
import os

def bfs(maze, start, finish):
    """
    Breadth-first search

    Parameters:
    - maze: The 2D matrix that represents the maze with 0 represents emptry space and 1 represents a wall
    - start: A tuple with the coordinates of starting position
    - finish: A tuple with the coordinates of finishing position

    Returns:
    - Number of steps from start to finish, equals -1 if the path is not found
    - Viz - anything required for step-by-step vizualization
    
    """
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    #the queue contains tuples of (current_position, current_steps)
    queue = deque([(start, 0)])
    visited = set([start])
    visited_order = [] #for tracking the order

    while queue:
        current, steps = queue.popleft() #popleft so we can get always the element in the first position
        visited_order.append(current)

        if current == finish:
            viz = {'maze': maze, 'start': start, 'finish': finish, 'visited_order': visited_order, 'name': 'BFS'} #dictionary that contains evrything for the visualization
            return steps, viz

        r, c = current
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            #check that we are inside the maze and it's not a wall
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), steps + 1))

    viz = {'maze': maze, 'start': start, 'finish': finish, 'visited_order': visited_order, 'name': 'BFS'}
    return -1, viz

def dfs(maze, start, finish):
    """
    Depth-first search

    Parameters:
    - maze: The 2D matrix that represents the maze with 0 represents emptry space and 1 represents a wall
    - start: A tuple with the coordinates of starting position
    - finish: A tuple with the coordinates of finishing position

    Returns:
    - Number of steps from start to finish, equals -1 if the path is not found
    - Viz - everything required for step-by-step vizualization
    
    """
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    #the stack is made as the same concept as the queue
    stack = [(start, 0)]
    visited = set([start])
    visited_order = []

    while stack:
        current, steps = stack.pop() #pop because we need to explore all the newest nodes first
        visited_order.append(current)

        if current == finish:
            viz = {'maze': maze, 'start': start, 'finish': finish, 'visited_order': visited_order, 'name': 'DFS'}
            return steps, viz

        r, c = current
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    stack.append(((nr, nc), steps + 1))

    viz = {'maze': maze, 'start': start, 'finish': finish, 'visited_order': visited_order, 'name': 'DFS'}
    return -1, viz

def vizualize(viz):
    """
    Vizualization function. Shows step by step the work of the search algorithm

    Parameters:
    - viz: Everything required for vizualization
    """
    maze = viz['maze']
    start = viz['start']
    finish = viz['finish']
    visited_order = viz['visited_order']
    algo_name = viz['name']
    
    rows = len(maze)
    cols = len(maze[0])
    
    current_visited = set()
    
    print(f"\nStarting {algo_name} visualization in 2 seconds...")
    time.sleep(2)
    
    for step, current in enumerate(visited_order):
        current_visited.add(current)
        
        #using cls for cleaning the console
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"--- Algorithm: {algo_name} | Step: {step + 1} ---")
        
        for r in range(rows):
            row_str = ""
            for c in range(cols):
                #check the current coordintes for printing the correct symbol
                if (r, c) == start:
                    row_str += " S "
                elif (r, c) == finish:
                    row_str += " E "
                elif (r, c) == current:
                    row_str += " @ " #head of the search
                elif (r, c) in current_visited:
                    row_str += " . " #previously visited cells
                elif maze[r][c] == 1:
                    row_str += "###"
                else:
                    row_str += "   " #unvisited space
            print(row_str)
        
        time.sleep(0.5)
        
    print(f"{algo_name} Visualization Complete.\n")


# Example usage:
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0]
]

start_position = (0, 0)
finish_position = (4, 0)

num_steps_bfs, viz_bfs = bfs(maze, start_position, finish_position)
num_steps_dfs, viz_dfs = dfs(maze, start_position, finish_position)

# Print number of steps in path
if num_steps_bfs != -1:
    print(f"Path from {start_position} to {finish_position} using bfs is {num_steps_bfs} steps.")
else:
    print(f"No path from {start_position} to {finish_position} exists.")

if num_steps_dfs != -1:
    print(f"Path from {start_position} to {finish_position} using dfs is {num_steps_dfs} steps.")
else:
    print(f"No path from {start_position} to {finish_position} exists.")

# Vizualize algorithm step-by-step even if the path was not found
vizualize(viz_bfs)
vizualize(viz_dfs)

#-----------------------------------------------------------------------------------------------------------------
# Other tests 
def run_custom_tests():
    test_cases = {
        "Open Space (Shows BFS expanding everywhere, DFS going straight)": {
            "maze": [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            "start": (0, 0),
            "finish": (3, 4)
        },
        "Long Corridor (Snake)": {
            "maze": [
                [0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0]
            ],
            "start": (0, 0),
            "finish": (0, 2)
        },
        "No Path Available (Trapped)": {
            "maze": [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            "start": (0, 0),
            "finish": (3, 4)
        }
    }

    for name, data in test_cases.items():
        print(f"\n=== TEST: {name} ===")
        b_steps, b_viz = bfs(data["maze"], data["start"], data["finish"])
        d_steps, d_viz = dfs(data["maze"], data["start"], data["finish"])
        print(f"BFS Steps: {b_steps} (Visited {len(b_viz['visited_order'])} cells)")
        print(f"DFS Steps: {d_steps} (Visited {len(d_viz['visited_order'])} cells)")

        # vizualize(b_viz)
        # vizualize(d_viz)

run_custom_tests()