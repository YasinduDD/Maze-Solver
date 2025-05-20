# Group 4 - 3 Bits
# Group Members -   20/ENG/034 - Dissanayake S.D.A.Y.D.
#                   20/ENG/043 - Fernando W.N.R.
#                   20/ENG/157 - Waththegama K.S.

import numpy as np
import random
from collections import deque
import pygame
import time
import heapq
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# Maze generation and solving functions


# Function to generate a maze using a randomized Prim's algorithm
def generate_maze(size):
    maze = np.ones((size, size), dtype=int)
    start = (random.randint(0, size - 1), random.randint(0, size - 1))
    maze[start] = 0
    walls = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        x, y = start[0] + dx, start[1] + dy
        if 0 <= x < size and 0 <= y < size:
            walls.append((x, y))
    while walls:
        wall = random.choice(walls)
        x, y = wall
        visited_neighbors = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and maze[nx, ny] == 0:
                visited_neighbors += 1
        if visited_neighbors == 1:
            maze[x, y] = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size and maze[nx, ny] == 1:
                    walls.append((nx, ny))
        walls.remove(wall)
    return maze


# Breadth First Search algorithm
def bfs(maze, start, end, draw_function=None):
    h, w = maze.shape
    queue = deque([(start[0], start[1], [])])
    visited = set()
    parent = {}

    while queue:
        if draw_function:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

        y, x, path = queue.popleft()

        if (y, x) == end:
            return path + [(y, x)]

        if (y, x) in visited:
            continue

        visited.add((y, x))
        parent[(y, x)] = path[-1] if path else None

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)] + (
            [(-1, -1), (-1, 1), (1, -1), (1, 1)] if (draw_function == None) else []
        ):
            ny, nx = y + dy, x + dx
            if (
                0 <= ny < h
                and 0 <= nx < w
                and maze[ny, nx] == 0
                and (ny, nx) not in visited
            ):
                queue.append((ny, nx, path + [(y, x)]))
                if draw_function:
                    draw_function(maze, visited, parent, start, end, (y, x))
                    time.sleep(0.05)
    return None


# Dijkstra's algorithm
def dijkstra(maze, start, end, draw_function=None):
    h, w = maze.shape
    pq = [(0, start[0], start[1], [])]  # (distance, y, x, path)
    distances = {(start[0], start[1]): 0}
    parent = {}

    while pq:
        if draw_function:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

        current_dist, y, x, path = heapq.heappop(pq)

        if (y, x) == end:
            return path + [(y, x)]

        if (y, x) in distances and current_dist > distances[(y, x)]:
            continue

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)] + (
            [(-1, -1), (-1, 1), (1, -1), (1, 1)] if (draw_function == None) else []
        ):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and maze[ny, nx] == 0:
                new_dist = current_dist + 1  # Assuming uniform cost
                if (ny, nx) not in distances or new_dist < distances[(ny, nx)]:
                    distances[(ny, nx)] = new_dist
                    heapq.heappush(pq, (new_dist, ny, nx, path + [(y, x)]))
                    parent[(ny, nx)] = (y, x)
                    if draw_function:
                        draw_function(
                            maze, set(distances.keys()), parent, start, end, (y, x)
                        )
                        time.sleep(0.05)

    return None


# Heuristic function for A* algorithm (Manhattan distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# A* algorithm
def astar(maze, start, end, draw_function=None):
    h, w = maze.shape
    open_set = [(0, start[0], start[1], [])]  # (f_score, y, x, path)
    came_from = {}
    g_score = {(start[0], start[1]): 0}
    f_score = {(start[0], start[1]): heuristic(start, end)}

    while open_set:
        if draw_function:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

        current_f, y, x, path = heapq.heappop(open_set)

        if (y, x) == end:
            return path + [(y, x)]

        if (y, x) in g_score and current_f > f_score[(y, x)]:
            continue

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)] + (
            [(-1, -1), (-1, 1), (1, -1), (1, 1)] if (draw_function == None) else []
        ):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and maze[ny, nx] == 0:
                tentative_g_score = g_score[(y, x)] + 1
                if (ny, nx) not in g_score or tentative_g_score < g_score[(ny, nx)]:
                    came_from[(ny, nx)] = (y, x)
                    g_score[(ny, nx)] = tentative_g_score
                    f_score[(ny, nx)] = tentative_g_score + heuristic((ny, nx), end)
                    heapq.heappush(
                        open_set, (f_score[(ny, nx)], ny, nx, path + [(y, x)])
                    )
                    if draw_function:
                        draw_function(
                            maze, set(g_score.keys()), came_from, start, end, (y, x)
                        )
                        time.sleep(0.05)

    return None


# Function to draw the maze using Pygame
def draw_maze(maze, explored=None, parent=None, start=None, end=None, current=None):
    size = len(maze)
    cell_size = 30
    screen_size = size * cell_size
    screen = pygame.display.get_surface()
    colors = {
        "wall": (0, 0, 0),
        "path": (255, 255, 255),
        "start": (0, 255, 0),
        "end": (255, 0, 0),
        "explored": (173, 216, 230),
        "current": (255, 165, 0),
        "solution": (0, 0, 255),
    }
    screen.fill((255, 255, 255))
    for x in range(size):
        for y in range(size):
            rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
            if maze[x, y] == 1:
                pygame.draw.rect(screen, colors["wall"], rect)
            else:
                pygame.draw.rect(screen, colors["path"], rect)
    if explored:
        for x, y in explored:
            rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, colors["explored"], rect)
    if current:
        rect = pygame.Rect(
            current[1] * cell_size, current[0] * cell_size, cell_size, cell_size
        )
        pygame.draw.rect(screen, colors["current"], rect)
    if start:
        start_rect = pygame.Rect(
            start[1] * cell_size, start[0] * cell_size, cell_size, cell_size
        )
        pygame.draw.rect(screen, colors["start"], start_rect)
    if end:
        end_rect = pygame.Rect(
            end[1] * cell_size, end[0] * cell_size, cell_size, cell_size
        )
        pygame.draw.rect(screen, colors["end"], end_rect)
    pygame.display.flip()


# Function to get the cell clicked by the user
def get_clicked_cell(mouse_pos, cell_size):
    x, y = mouse_pos
    row = y // cell_size
    col = x // cell_size
    return (row, col)


# Function to crop the image to the maze boundaries
def crop_to_maze_boundaries(img):
    if img.ndim == 3:
        gray = img[:, :, 0]
    else:
        gray = img.copy()
    binary = gray < np.max(gray) / 2
    ys, xs = np.where(binary)
    if ys.size == 0 or xs.size == 0:
        print("No walls detected, returning original image.")
        return img
    min_y, max_y = np.min(ys), np.max(ys)
    min_x, max_x = np.min(xs), np.max(xs)
    cropped_img = img[min_y:max_y, min_x:max_x]
    return cropped_img, min_x, min_y


# Function to find the nearest path to a given point
def find_nearest_path(y, x, mapT, search_radius=20):
    h, w = mapT.shape
    for r in range(search_radius):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mapT[ny, nx] == 0:
                    return ny, nx
    return None, None


# Function to solve the maze using the selected algorithm
def solve_image_maze(mapT, start, end, algorithm):
    if algorithm == "BFS":
        return bfs(mapT, start, end)
    elif algorithm == "Dijkstra":
        return dijkstra(mapT, start, end)
    elif algorithm == "A*":
        return astar(mapT, start, end)
    else:
        raise ValueError("Invalid algorithm selected.")


def main():
    print("Select an option:")
    print("1. Use a pre-defined maze")
    print("2. Input an image of a maze")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        print("Select the size of the maze:")
        print("1. 9x9")
        print("2. 16x16")
        print("3. 21x21")
        size_choice = input("Enter your choice (1, 2, or 3): ")
        if size_choice == "1":
            size = 9
        elif size_choice == "2":
            size = 16
        elif size_choice == "3":
            size = 21
        else:
            print("Invalid choice. Defaulting to 16x16.")
            size = 16
        maze = generate_maze(size)
        pygame.init()
        cell_size = 30
        screen_size = size * cell_size
        screen = pygame.display.set_mode((screen_size, screen_size))
        pygame.display.set_caption("Maze Solver")
        draw_maze(maze)
        print("Click on the maze to set the START point (green).")
        start = None
        while start is None:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    row, col = get_clicked_cell(mouse_pos, cell_size)
                    if maze[row, col] == 0:
                        start = (row, col)
                        draw_maze(maze, start=start)
                        print(f"Start point set at: {start}")
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
        print("Click on the maze to set the END point (red).")
        end = None
        while end is None:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    row, col = get_clicked_cell(mouse_pos, cell_size)
                    if maze[row, col] == 0 and (row, col) != start:
                        end = (row, col)
                        draw_maze(maze, start=start, end=end)
                        print(f"End point set at: {end}")
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
        print("Select the algorithm to solve the maze:")
        print("1. Breadth-First Search (BFS)")
        print("2. Dijkstra's Algorithm")
        print("3. A* Search")
        algorithm_choice = input("Enter your choice (1, 2, or 3): ")
        start_time = time.time()
        if algorithm_choice == "1":
            print("Solving with BFS...")
            path = bfs(maze, start, end, draw_maze)
        elif algorithm_choice == "2":
            print("Solving with Dijkstra's Algorithm...")
            path = dijkstra(maze, start, end, draw_maze)
        elif algorithm_choice == "3":
            print("Solving with A*...")
            path = astar(maze, start, end, draw_maze)
        else:
            print("Invalid choice. Defaulting to BFS.")
            path = bfs(maze, start, end, draw_maze)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print("Press R to reset")
        draw_maze(maze, set(), {}, start, end)
        for x, y in path:
            rect = pygame.Rect(y * 30, x * 30, 30, 30)
            pygame.draw.rect(screen, (0, 0, 255), rect)
            pygame.display.flip()
            time.sleep(0.1)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset the maze when 'R' is pressed
                        maze = generate_maze(size)
                        start = None
                        end = None
                        draw_maze(maze)
                        print("Maze reset. Click to set new START and END points.")
                        while start is None:
                            for event in pygame.event.get():
                                if event.type == pygame.MOUSEBUTTONDOWN:
                                    mouse_pos = pygame.mouse.get_pos()
                                    row, col = get_clicked_cell(mouse_pos, cell_size)
                                    if maze[row, col] == 0:
                                        start = (row, col)
                                        draw_maze(maze, start=start)
                                        print(f"Start point set at: {start}")
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    exit()
                        while end is None:
                            for event in pygame.event.get():
                                if event.type == pygame.MOUSEBUTTONDOWN:
                                    mouse_pos = pygame.mouse.get_pos()
                                    row, col = get_clicked_cell(mouse_pos, cell_size)
                                    if maze[row, col] == 0 and (row, col) != start:
                                        end = (row, col)
                                        draw_maze(maze, start=start, end=end)
                                        print(f"End point set at: {end}")
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    exit()
                        print("Select the algorithm to solve the maze:")
                        print("1. Breadth-First Search (BFS)")
                        print("2. Dijkstra's Algorithm")
                        print("3. A* Search")
                        algorithm_choice = input("Enter your choice (1, 2, or 3): ")
                        start_time = time.time()
                        if algorithm_choice == "1":
                            print("Solving with BFS...")
                            path = bfs(maze, start, end, draw_maze)
                        elif algorithm_choice == "2":
                            print("Solving with Dijkstra's Algorithm...")
                            path = dijkstra(maze, start, end, draw_maze)
                        elif algorithm_choice == "3":
                            print("Solving with A*...")
                            path = astar(maze, start, end, draw_maze)
                        else:
                            print("Invalid choice. Defaulting to BFS.")
                            path = bfs(maze, start, end, draw_maze)
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"Time taken: {elapsed_time:.2f} seconds")
                        print("Press R to reset")
                        draw_maze(maze, set(), {}, start, end)
                        for x, y in path:
                            rect = pygame.Rect(y * 30, x * 30, 30, 30)
                            pygame.draw.rect(screen, (0, 0, 255), rect)
                            pygame.display.flip()
                            time.sleep(0.1)
        pygame.quit()
    elif choice == "2":
        img_name = input("Enter the path to the maze image (Relative Path): ")
        rgb_img = plt.imread(img_name)
        cropped_img, offset_x, offset_y = crop_to_maze_boundaries(rgb_img)
        plt.figure(figsize=(14, 14))
        plt.imshow(cropped_img)
        plt.title("Click on the start and end points")
        points = plt.ginput(2)
        plt.close()
        (x0, y0), (x1, y1) = points
        x0, y0, x1, y1 = (
            int(x0) + offset_x,
            int(y0) + offset_y,
            int(x1) + offset_x,
            int(y1) + offset_y,
        )
        if cropped_img.ndim == 3:
            thr_img = cropped_img[:, :, 0] > np.max(cropped_img[:, :, 0]) / 2
        else:
            thr_img = cropped_img > np.max(cropped_img) / 2
        skeleton = skeletonize(thr_img)
        mapT = ~skeleton
        y0, x0 = find_nearest_path(y0, x0, mapT)
        y1, x1 = find_nearest_path(y1, x1, mapT)
        if y0 is None or x0 is None or y1 is None or x1 is None:
            print("Could not find valid start or end points on the skeletonized path.")
            exit()
        print("Select the algorithm to solve the maze:")
        print("1. Breadth-First Search (BFS)")
        print("2. Dijkstra's Algorithm")
        print("3. A* Search")
        algorithm_choice = input("Enter your choice (1, 2, or 3): ")
        start_time = time.time()
        if algorithm_choice == "1":
            print("Solving with BFS...")
            path = solve_image_maze(mapT, (y0, x0), (y1, x1), "BFS")
        elif algorithm_choice == "2":
            print("Solving with Dijkstra's Algorithm...")
            path = solve_image_maze(mapT, (y0, x0), (y1, x1), "Dijkstra")
        elif algorithm_choice == "3":
            print("Solving with A*...")
            path = solve_image_maze(mapT, (y0, x0), (y1, x1), "A*")
        else:
            print("Invalid choice. Defaulting to BFS.")
            path = solve_image_maze(mapT, (y0, x0), (y1, x1), "BFS")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.2f} seconds")
        if path:
            path_y, path_x = zip(*path)
            print("Path found.")
        else:
            print("No valid path found!")
            path_x, path_y = [], []
        plt.figure(figsize=(14, 14))
        plt.imshow(cropped_img)
        plt.plot(x0, y0, "go", markersize=14, label="Start")
        plt.plot(x1, y1, "ro", markersize=14, label="End")
        if path:
            plt.plot(path_x, path_y, "r-", linewidth=3, label="Path")
        plt.legend()
        plt.show()
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
