from collections import deque


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
    rows, cols = len(maze), len(maze[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    visited = set()
    visited.add(start)
    parent = {start: None}
    queue = deque([start])
    visited_order = []

    while queue:
        current = queue.popleft()
        visited_order.append(current)

        if current == finish:
            path = []
            node = finish
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()

            viz = {"maze": maze, "visited_order": visited_order, "path": path, "name": "BFS"}
            return len(path) - 1, viz

        r, c = current
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                parent[(nr, nc)] = current
                queue.append((nr, nc))

    viz = {"maze": maze, "visited_order": visited_order, "path": [], "name": "BFS"}
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
    rows, cols = len(maze), len(maze[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    visited = set()
    parent = {start: None}
    stack = [start]
    visited_order = []

    while stack:
        current = stack.pop()

        if current in visited:
            continue
        visited.add(current)
        visited_order.append(current)

        if current == finish:
            path = []
            node = finish
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()

            viz = {"maze": maze, "visited_order": visited_order, "path": path, "name": "DFS"}
            return len(path) - 1, viz

        r, c = current
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0 and (nr, nc) not in visited:
                parent[(nr, nc)] = current
                stack.append((nr, nc))

    viz = {"maze": maze, "visited_order": visited_order, "path": [], "name": "DFS"}
    return -1, viz


def vizualize(viz):
    """
    Vizualization function. Shows step by step the work of the search algorithm

    Parameters:
    - viz: Everything required for vizualization
    """
    maze = viz["maze"]
    visited_order = viz["visited_order"]
    path = viz["path"]
    name = viz["name"]
    rows, cols = len(maze), len(maze[0])
    path_set = set(path)
    visited_set = set(visited_order)

    print(f"\n{'='*40}")
    print(f" {name} Visualization")
    print(f"{'='*40}")

    # Print exploration order
    print(f"\nVisited {len(visited_order)} cells in this order:")
    for step, cell in enumerate(visited_order):
        print(f"  Step {step}: {cell}")

    # Print maze grid with markers
    print(f"\nFinal state (. = empty, # = wall, * = visited, P = path, S = start, F = finish):")
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            pos = (r, c)
            if path and pos == path[0]:
                row_str += "S "
            elif path and pos == path[-1]:
                row_str += "F "
            elif pos in path_set:
                row_str += "P "
            elif pos in visited_set:
                row_str += "* "
            elif maze[r][c] == 1:
                row_str += "# "
            else:
                row_str += ". "
        print(f"  {row_str}")

    if path:
        print(f"\nPath ({len(path) - 1} steps): {' -> '.join(str(p) for p in path)}")
    else:
        print("\nNo path found!")


def run_test(name, maze, start, finish):
    """Helper to run and display a test case."""
    print(f"\n{'#'*50}")
    print(f" TEST: {name}")
    print(f"{'#'*50}")

    num_steps_bfs, viz_bfs = bfs(maze, start, finish)
    num_steps_dfs, viz_dfs = dfs(maze, start, finish)

    if num_steps_bfs != -1:
        print(f"BFS: path from {start} to {finish} is {num_steps_bfs} steps.")
    else:
        print(f"BFS: no path from {start} to {finish}.")

    if num_steps_dfs != -1:
        print(f"DFS: path from {start} to {finish} is {num_steps_dfs} steps.")
    else:
        print(f"DFS: no path from {start} to {finish}.")

    vizualize(viz_bfs)
    vizualize(viz_dfs)


# Test 1: Normal case (from template)
maze1 = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0]
]
run_test("Normal maze", maze1, (0, 0), (4, 4))

# Test 2: No path exists
maze2 = [
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 0]
]
run_test("No path (blocked start)", maze2, (0, 0), (2, 2))

# Test 3: Start equals finish
maze3 = [
    [0, 0],
    [0, 0]
]
run_test("Start == Finish", maze3, (0, 0), (0, 0))

# Test 4: 1x1 maze
maze4 = [[0]]
run_test("1x1 maze", maze4, (0, 0), (0, 0))

# Test 5: Larger complex maze
maze5 = [
    [0, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0]
]
run_test("Complex maze", maze5, (0, 0), (6, 6))
