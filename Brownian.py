# brownian_tree_dla.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Parameters
GRID_SIZE = 301            
NUM_PARTICLES = 3000       
SPAWN_RADIUS = GRID_SIZE//2 - 5
KILL_RADIUS = GRID_SIZE//2 + 5
MAX_STEPS_PER_PARTICLE = 20000
FRAME_EVERY = 50           # update animation every N stuck particles

def random_spawn(center, radius):
    """Return integer (x,y) spawn coordinate on a circle of given radius."""
    theta = random.random() * 2 * np.pi
    x = int(center + radius * np.cos(theta))
    y = int(center + radius * np.sin(theta))
    return x, y

def in_bounds(x, y, size):
    return 0 <= x < size and 0 <= y < size

def neighbors8(x, y):
    # 8-neighborhood
    return [(x-1,y-1),(x-1,y),(x-1,y+1),
            (x,y-1),           (x,y+1),
            (x+1,y-1),(x+1,y),(x+1,y+1)]

def main():
    size = GRID_SIZE
    grid = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    # seed
    grid[center, center] = 1

    stuck_count = 1
    frames = []  # snapshots for animation (store arrays)
    print("Starting DLA: grid size", size, "num particles", NUM_PARTICLES)

    while stuck_count < NUM_PARTICLES:
        # spawn
        x, y = random_spawn(center, SPAWN_RADIUS)
        steps = 0

        while True:
            # random 8-direction step (Brownian)
            dx, dy = random.choice([(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)])
            x += dx
            y += dy
            steps += 1

            # if out of bounds or too far from center, respawn or kill this walker
            dx_c = x - center
            dy_c = y - center
            dist_sq = dx_c*dx_c + dy_c*dy_c
            if not in_bounds(x, y, size) or dist_sq > (KILL_RADIUS**2):
                # respawn
                x, y = random_spawn(center, SPAWN_RADIUS)
                steps = 0
                continue

            # check adjacency to cluster
            touching = False
            for nx, ny in neighbors8(x, y):
                if in_bounds(nx, ny, size) and grid[nx, ny]:
                    touching = True
                    break

            if touching:
                grid[x, y] = 1
                stuck_count += 1
                # optionally expand spawn/killing radii slowly (not necessary)
                if stuck_count % FRAME_EVERY == 0:
                    frames.append(grid.copy())
                    print(f"Stuck: {stuck_count}/{NUM_PARTICLES}")
                break

            if steps > MAX_STEPS_PER_PARTICLE:
                # give up and respawn
                x, y = random_spawn(center, SPAWN_RADIUS)
                steps = 0

    # ensure final frame captured
    frames.append(grid.copy())
    print("Finished. Creating animation...")

    # plotting / animation
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title("Brownian Tree (DLA)")
    ax.axis('off')
    im = ax.imshow(frames[0], cmap='inferno', origin='lower')

    def update(i):
        im.set_data(frames[i])
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
    plt.show()

    # If you want to save the animation uncomment the next line (requires ffmpeg)
    # ani.save('brownian_tree.mp4', dpi=150, writer='ffmpeg')

if __name__ == "__main__":
    main()
