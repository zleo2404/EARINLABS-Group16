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
    queue = deque([start])
    visited_order = []

    # Explore the maze level by level using a queue
    while queue:
        current = queue.popleft()
        visited_order.append(current)

        if current == finish:
            # TODO: reconstruct actual path length
            return len(visited_order), visited_order

        r, c = current
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))

    return -1, visited_order


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
    # TODO: implement DFS
    pass


def vizualize(viz):
    """
    Vizualization function. Shows step by step the work of the search algorithm

    Parameters:
    - viz: Everything required for vizualization
    """
    # TODO: implement visualization
    pass


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

print(f"BFS explored {len(viz_bfs)} cells")
if num_steps_bfs != -1:
    print(f"Found finish after visiting {num_steps_bfs} cells (not path length yet)")
else:
    print("No path found")
