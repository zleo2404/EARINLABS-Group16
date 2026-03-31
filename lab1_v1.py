from collections import deque
import os
import time


def build_viz(maze, start, finish, visited_order, path, name):
    return {
        "maze": maze,
        "start": start,
        "finish": finish,
        "visited_order": visited_order,
        "path": path,
        "name": name,
    }


# Helper
def get_path(parent, finish):
    path = []
    node = finish
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def validate_position(position, rows, cols):
    if not isinstance(position, (tuple, list)) or len(position) != 2:
        return False

    row, col = position
    if not isinstance(row, int) or not isinstance(col, int):
        return False

    return 0 <= row < rows and 0 <= col < cols


def validate_maze_input(maze, start, finish):
    if not isinstance(maze, list) or len(maze) == 0:
        return False, 0, 0

    if not all(isinstance(row, list) for row in maze):
        return False, 0, 0

    cols = len(maze[0])
    if cols == 0:
        return False, 0, 0

    if any(len(row) != cols for row in maze):
        return False, 0, 0

    rows = len(maze)
    if not validate_position(start, rows, cols):
        return False, 0, 0

    if not validate_position(finish, rows, cols):
        return False, 0, 0

    return True, rows, cols


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
    is_valid_input, rows, cols = validate_maze_input(maze, start, finish)
    if not is_valid_input:
        return -1, build_viz(maze, start, finish, [], [], "BFS")

    if maze[start[0]][start[1]] == 1 or maze[finish[0]][finish[1]] == 1:
        return -1, build_viz(maze, start, finish, [], [], "BFS")

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # the queue contains tuples of (current_position, current_steps)
    queue = deque([(start, 0)])
    visited = {start}
    visited_order = []
    parent = {start: None}

    while queue:
        current, steps = (
            queue.popleft()
        )  # popleft so we can always get the element in the first position
        visited_order.append(current)

        if current == finish:
            path = get_path(parent, finish)
            return len(path) - 1, build_viz(
                maze, start, finish, visited_order, path, "BFS"
            )

        r, c = current
        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            # check that we are inside the maze and it's not a wall
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), steps + 1))
                    parent[(nr, nc)] = current

    return -1, build_viz(maze, start, finish, visited_order, [], "BFS")


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
    is_valid_input, rows, cols = validate_maze_input(maze, start, finish)
    if not is_valid_input:
        return -1, build_viz(maze, start, finish, [], [], "DFS")

    if maze[start[0]][start[1]] == 1 or maze[finish[0]][finish[1]] == 1:
        return -1, build_viz(maze, start, finish, [], [], "DFS")

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    stack = [(start, 0)]
    visited = {start}
    visited_order = []
    parent = {start: None}

    while stack:
        current, steps = (
            stack.pop()
        )  # pop because we need to explore all the newest nodes first.
        visited_order.append(current)

        if current == finish:
            path = get_path(parent, finish)
            return len(path) - 1, build_viz(
                maze, start, finish, visited_order, path, "DFS"
            )

        r, c = current
        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    stack.append(((nr, nc), steps + 1))
                    parent[(nr, nc)] = current

    return -1, build_viz(maze, start, finish, visited_order, [], "DFS")


def vizualize(viz):
    """
    Vizualization function. Shows step by step the work of the search algorithm

    Parameters:
    - viz: Everything required for vizualization
    """
    maze = viz["maze"]
    start = viz["start"]
    finish = viz["finish"]
    visited_order = viz["visited_order"]
    path = viz["path"]
    algo_name = viz["name"]

    valid_shape, rows, cols = validate_maze_input(maze, start, finish)
    if not valid_shape:
        print(f"\n--- Algorithm: {algo_name} ---")
        print("Invalid or empty maze.")
        print(f"{algo_name} Visualization Complete.\n")
        return

    current_visited = set()

    print(f"\nStarting {algo_name} visualization in 2 seconds...")
    time.sleep(2)

    for step, current in enumerate(visited_order):
        current_visited.add(current)

        # using cls for cleaning the console
        os.system("cls" if os.name == "nt" else "clear")
        print(f"--- Algorithm: {algo_name} | Step: {step + 1} ---")

        for r in range(rows):
            row_str = ""
            for c in range(cols):
                # check the current coordinates to print the correct symbol
                if (r, c) == start:
                    row_str += " S "
                elif (r, c) == finish:
                    row_str += " E "
                elif (r, c) == current:
                    row_str += " @ "  # head of the search
                elif (r, c) in current_visited:
                    row_str += " . "  # previously visited cells
                elif maze[r][c] == 1:
                    row_str += "###"
                else:
                    row_str += "   "  # unvisited space
            print(row_str)
        time.sleep(0.5)

    if path:
        os.system("cls" if os.name == "nt" else "clear")
        print(f"--- Algorithm: {algo_name} | FINAL PATH ---")
        for r in range(rows):
            row_str = ""
            for c in range(cols):
                if (r, c) == start:
                    row_str += " S "
                elif (r, c) == finish:
                    row_str += " E "
                elif (r, c) in path:
                    row_str += " * "
                elif maze[r][c] == 1:
                    row_str += "###"
                else:
                    row_str += "   "
            print(row_str)
        time.sleep(3)
    print(f"{algo_name} Visualization Complete.\n")


def run_example():
    maze = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ]

    start_position = (0, 0)
    finish_position = (4, 0)

    num_steps_bfs, viz_bfs = bfs(maze, start_position, finish_position)
    num_steps_dfs, viz_dfs = dfs(maze, start_position, finish_position)

    if num_steps_bfs != -1:
        print(
            f"Path from {start_position} to {finish_position} using bfs is {num_steps_bfs} steps."
        )
    else:
        print(f"No path from {start_position} to {finish_position} exists using bfs.")

    if num_steps_dfs != -1:
        print(
            f"Path from {start_position} to {finish_position} using dfs is {num_steps_dfs} steps."
        )
    else:
        print(f"No path from {start_position} to {finish_position} exists using dfs.")

    # Vizualize algorithm step-by-step even if the path was not found
    vizualize(viz_bfs)
    vizualize(viz_dfs)


def run_custom_tests(show_visualization=False):
    test_cases = {
        "Complex maze": {
            "maze": [
                [0, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1],
                [0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0],
            ],
            "start": (0, 0),
            "finish": (1, 3),
        },
        "Open Space": {
            "maze": [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            "start": (0, 0),
            "finish": (3, 4),
        },
        "No Path Available (Trapped)": {
            "maze": [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            "start": (0, 0),
            "finish": (3, 4),
        },
        "Start and End are the same": {
            "maze": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            "start": (0, 0),
            "finish": (0, 0),
        },
        "Start inside a wall": {
            "maze": [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
            "start": (0, 0),
            "finish": (2, 2),
        },
        "Finish inside a wall": {
            "maze": [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
            "start": (0, 0),
            "finish": (2, 2),
        },
        "1x1 Open Maze": {
            "maze": [[0]],
            "start": (0, 0),
            "finish": (0, 0),
        },
        "1x1 Blocked Maze": {
            "maze": [[1]],
            "start": (0, 0),
            "finish": (0, 0),
        },
        "Disconnected Maze": {
            "maze": [[0, 1, 1], [1, 1, 1], [1, 1, 0]],
            "start": (0, 0),
            "finish": (2, 2),
        },
        "BFS Shorter Than DFS": {
            "maze": [
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            "start": (0, 0),
            "finish": (3, 4),
        },
        "Empty Maze": {
            "maze": [],
            "start": (0, 0),
            "finish": (0, 0),
        },
        "Non-Rectangular Maze": {
            "maze": [[0, 0, 0], [0, 1], [0, 0, 0]],
            "start": (0, 0),
            "finish": (2, 2),
        },
        "Out-of-Bounds Start": {
            "maze": [[0, 0], [0, 0]],
            "start": (2, 0),
            "finish": (1, 1),
        },
        "Negative Coordinate": {
            "maze": [[0, 0], [0, 0]],
            "start": (-1, 0),
            "finish": (1, 1),
        },
    }

    for name, data in test_cases.items():
        print(f"\n=== TEST: {name} ===")
        b_steps, b_viz = bfs(data["maze"], data["start"], data["finish"])
        d_steps, d_viz = dfs(data["maze"], data["start"], data["finish"])
        print(f"BFS Steps: {b_steps} (Visited {len(b_viz['visited_order'])} cells)")
        print(f"DFS Steps: {d_steps} (Visited {len(d_viz['visited_order'])} cells)")

        if name == "BFS Shorter Than DFS":
            print(
                f"Comparison result for current direction order: BFS = {b_steps}, DFS = {d_steps}"
            )
            if b_steps != -1 and d_steps != -1 and d_steps > b_steps:
                print("This test shows BFS finding a shorter path than DFS.")
            else:
                print(
                    "This maze does not show DFS taking a longer path with the current direction order."
                )

        if show_visualization:
            vizualize(b_viz)
            vizualize(d_viz)


if __name__ == "__main__":
    run_example()
    run_custom_tests()
