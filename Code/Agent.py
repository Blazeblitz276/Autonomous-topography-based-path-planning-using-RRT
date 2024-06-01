import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from RRT import RRT_Exp

class Agent:
    # Initialize the agent with starting position, step size for the movement, and iterations for RRT
    def __init__(self, start_position, step_size=1, iterations=5000, discovery_radius = 4):
        self.position = np.array(start_position, dtype=float)  # Current position of the agent
        self.orientation = 0  # Current orientation of the agent in radians
        self.velocity = 0  # Current linear velocity of the agent
        self.angular_velocity = 0  # Current angular velocity of the agent
        self.discovery_radius = discovery_radius  # Radius within which the agent discovers the terrain
        self.step_size = step_size  # Step size for each movement the agent makes
        self.iterations = iterations  # Number of iterations for the RRT algorithm

    # Update the velocity of the agent, ensuring it doesn't go below 0 or above a max velocity
    def change_velocity(self, dv):
        self.velocity += dv
        self.velocity = min(max(self.velocity, 0), 3)

    # Update the angular velocity of the agent, ensuring it stays within -pi to pi range
    def change_angular_velocity(self, dw):
        self.angular_velocity += dw
        self.angular_velocity = np.clip(self.angular_velocity, -np.pi, np.pi)

    # Move the agent based on its current velocity and orientation
    def move(self, dt=1):
        self.orientation += self.angular_velocity * dt
        dx = self.velocity * np.cos(self.orientation) * dt
        dy = self.velocity * np.sin(self.orientation) * dt
        self.position += np.array([dx, dy])

def initialize_terrain_and_cost(terrain_gen):
    terrain = terrain_gen.generate_terrain_with_trees()[0]
    cost_matrix = np.full(terrain.shape, np.inf)
    return terrain, cost_matrix

def update_explored_area_and_cost(agent, terrain, cost_matrix):
    x, y = np.round(agent.position).astype(int)
    for i in range(x - agent.discovery_radius, x + agent.discovery_radius + 1):
        for j in range(y - agent.discovery_radius, y + agent.discovery_radius + 1):
            if 0 <= i < terrain.shape[0] and 0 <= j < terrain.shape[1]:
                distance = np.sqrt((x - i)**2 + (y - j)**2)
                if distance <= agent.discovery_radius:
                    cost_matrix[i, j] = min(cost_matrix[i, j], terrain[i, j])

def on_key(event, agent, terrain, cost_matrix, ax, exploration_finished, start_end_points, rrt_popup_shown):
        if not exploration_finished[0]:
            if event.key == 'up':
                agent.change_velocity(0.5)  # Increase velocity
            elif event.key == 'down':
                agent.change_velocity(-0.5)  # Decrease velocity
            elif event.key == 'left':
                agent.change_angular_velocity(-0.1)  # Turn left
            elif event.key == 'right':
                agent.change_angular_velocity(0.1)  # Turn right
            elif event.key == 'x':
               tk.Tk().withdraw()  # Prevents an empty tkinter window from appearing
               messagebox.showinfo("Exploration Ended", "Exploration has ended. Now select two locations.")
               exploration_finished[0] = True
            agent.move()
            update_explored_area_and_cost(agent, terrain, cost_matrix)
            redraw_plot(ax, cost_matrix, agent)
            
def redraw_plot(ax, cost_matrix, agent):
    ax.clear()
    ax.imshow(cost_matrix, cmap='terrain', origin='lower')
    ax.scatter(agent.position[1], agent.position[0], c='red', s=40)
    plt.draw()

def on_mouse_click(event, start_end_points, ax, cost_matrix, exploration_finished, rrt_popup_shown, agent, terrain):
    if event.inaxes == ax and exploration_finished[0] and len(start_end_points) < 2:
        click_point = (int(event.ydata), int(event.xdata))
        if is_valid_click(cost_matrix, click_point):
            add_click_point(start_end_points, click_point, ax)
            if len(start_end_points) == 2:
                tk.Tk().withdraw()
                messagebox.showinfo("Points Selected", "Two points have been selected. Click OK to start RRT.")
                rrt_popup_shown[0] = True
                ani = run_rrt_and_visualize(agent, cost_matrix, start_end_points,ax)
                ani.save('Exploration.gif', fps=10)

def is_valid_click(cost_matrix, click_point):
    return cost_matrix[click_point] != np.inf

def add_click_point(start_end_points, click_point, ax):
    start_end_points.append(click_point)
    ax.scatter(*click_point[::-1], c='yellow', s=50)
    plt.draw()
    
def run_rrt_and_visualize(agent, cost_matrix, start_end_points, ax):
    start_point, end_point = start_end_points
    rrt_exp = RRT_Exp(cost_matrix, start_point, end_point, agent)
    rrt_exp.expand_tree(iterations=agent.iterations)
    path = rrt_exp.find_path()

    # Plot the RRT tree directly onto the existing axes
    rrt_exp.plot_tree(ax)

    # Visualize the agent's path and store the animation object
    ani = rrt_exp.visualize_agent_traversal(path, ax)

    # Show the plot with the RRT tree and the agent's path
    plt.show()

    # Return the animation object to keep it alive
    return ani
            





