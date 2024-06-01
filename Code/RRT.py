import numpy as np
import matplotlib.patches as patches
import matplotlib.animation as animation

class RRT_Exp:
    """
    Class for Rapidly-exploring Random Trees (RRT) exploration.
    """
    
    def __init__(self, terrain_matrix, start, goal, agent):
        """
        Initializes the RRT exploration algorithm.
        
        Parameters:
        - terrain_matrix: A matrix representing the terrain and its costs.
        - start: Starting point (x, y) for the RRT.
        - goal: Goal point (x, y) for the RRT.
        - agent: The agent object which will explore the terrain.
        """
        self.terrain_matrix = terrain_matrix  # The cost matrix of the terrain.
        self.start = start[::-1]  # Start point in (y, x) format to match matrix indexing.
        self.goal = goal[::-1]  # Goal point in (y, x) format to match matrix indexing.
        self.agent = agent  # The agent performing the RRT.
        self.tree = [(start, None)]  # Tree initialized with the start position.

    def is_collision_free(self, point):
        """
        Checks if a point is collision-free (i.e., traversable).
        
        Parameters:
        - point: The point (x, y) to check for collision.
        
        Returns:
        - Boolean indicating whether the point is free of collisions.
        """
        y, x = point
        return self.terrain_matrix[int(x), int(y)] != np.inf  # Check if the point is not an obstacle.

    def sample_free_space(self):
        while True:
            point = np.random.uniform(0, self.terrain_matrix.shape[0], 2)
            if self.is_collision_free(point):
                return point

    def nearest_vertex_index(self, point):
        closest_index = None
        min_dist = np.inf
        for index, (vertex, _) in enumerate(self.tree):
            dist = np.linalg.norm(np.array(vertex) - np.array(point))
            if dist < min_dist:
                closest_index = index
                min_dist = dist
        return closest_index

    def steer(self, from_point, to_point):
        direction = np.array(to_point) - np.array(from_point)
        distance = np.linalg.norm(direction)
        step_size = min(distance, self.agent.step_size)
        direction = direction / distance
        new_point = np.array(from_point) + step_size * direction
        return new_point.tolist()

    def expand_tree(self, iterations=10000, goal_bias=0.1):
        goal_region = 3  # Defines a region around the goal
        for _ in range(iterations):
            random_point = self.sample_free_space()
            # Add a bias to sample towards the goal
            if np.random.rand() < goal_bias:
                random_point = self.goal
            nearest_index = self.nearest_vertex_index(random_point)
            new_point = self.steer(self.tree[nearest_index][0], random_point)

            if self.is_collision_free(new_point):
                self.tree.append((new_point, nearest_index))
                
                if np.linalg.norm(np.array(new_point) - np.array(self.goal)) < goal_region:
                    if self.is_collision_free(self.goal):
                        self.tree.append((self.goal, len(self.tree) - 1))
                        return  # Goal reached

    def find_path(self):
        path = []
        current_index = len(self.tree) - 1
        # print("start",self.tree[current_index])
        while current_index is not None:
            current_node = self.tree[current_index]
            path.append(current_node[0])
            current_index = current_node[1]
        # print("end",self.tree[0])
        return path[::-1]

    def plot_tree(self, ax):
        for point, parent_index in self.tree:
            if parent_index is not None:
                parent_point = self.tree[parent_index][0]
                ax.plot([point[0], parent_point[0]], [point[1], parent_point[1]], 'b-')

        # ax.plot(*zip(*self.tree[::-1]), marker='o', color='red', linestyle='None')
        ax.imshow(self.terrain_matrix, cmap='terrain', origin='lower')
        ax.set_aspect('equal', adjustable='box')

    def visualize_agent_traversal(self, path, ax):
        # Create a circle patch representing the agent at the starting location
        agent_patch = patches.Circle(path[0][::-1], radius=self.agent.discovery_radius, color='red', alpha=0.5)
        agent_patch = patches.Circle(path[-1][::-1], radius=self.agent.discovery_radius, color='blue', alpha=0.5)
        ax.add_patch(agent_patch)
        
        # Create a line object for the optimal path, initially empty
        optimal_path_line, = ax.plot([], [], 'm-', linewidth=2)

        # Update function to animate the agent's movement and draw the optimal path
        def update(frame):
            # Draw the optimal path up to the current frame
            optimal_path_line.set_data([p[0] for p in path[:frame+1]], [p[1] for p in path[:frame+1]])

            # Move the agent along the path
            x, y = path[frame] # Reverse for correct axis order
            agent_patch.set_center((x, y))
            
            return agent_patch, optimal_path_line

        # Create the animation object
        ani = animation.FuncAnimation(fig=ax.figure, func=update, frames=len(path), blit=True, interval=100)
        
        # Return the animation object to prevent it from being garbage collected
        return ani
