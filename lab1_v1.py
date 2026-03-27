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
    visited_order = viz["visited_order"]
    path = viz["path"]
    name = viz["name"]

    # Simple text listing of visited cells
    print(f"\n--- {name} ---")
    print(f"Visited {len(visited_order)} cells in this order:")
    for step, cell in enumerate(visited_order):
        print(f"  Step {step}: {cell}")

    if path:
        print(f"Path ({len(path) - 1} steps): {' -> '.join(str(p) for p in path)}")
    else:
        print("No path found!")


# Example usage:
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0]
]

start_position = (0, 0)
finish_position = (4, 4)

num_steps_bfs, viz_bfs = bfs(maze, start_position, finish_position)
num_steps_dfs, viz_dfs = dfs(maze, start_position, finish_position)

if num_steps_bfs != -1:
    print(f"Path from {start_position} to {finish_position} using bfs is {num_steps_bfs} steps.")
else:
    print(f"No path from {start_position} to {finish_position} exists.")

if num_steps_dfs != -1:
    print(f"Path from {start_position} to {finish_position} using dfs is {num_steps_dfs} steps.")
else:
    print(f"No path from {start_position} to {finish_position} exists.")

vizualize(viz_bfs)
vizualize(viz_dfs)
