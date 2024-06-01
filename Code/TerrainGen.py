import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class TerrainGenerator:
    def __init__(self, seed=None, max_elevation=50, min_elevation=-50, num_trees=30):
        self.seed = seed
        self.max_elevation = max_elevation
        self.min_elevation = min_elevation
        self.num_trees = num_trees
        self.terrain = None
        np.random.seed(self.seed)

    def generate_noise(self, shape, scales):
        noise = np.zeros(shape)
        for scale in scales:
            noise += gaussian_filter(np.random.randn(*shape), scale)
        noise -= noise.min()
        noise /= noise.max()
        return noise

    def create_terrain(self):
        # Create base terrain
        terrain = np.zeros((200, 200))

        # Generate noise for peaks and valleys
        peaks_valleys_noise = self.generate_noise((200, 200), [10, 15, 20])

        # Apply elevations to the noise
        terrain += peaks_valleys_noise * (self.max_elevation - self.min_elevation) + self.min_elevation

        # Smooth the terrain for gradual gradients
        terrain = gaussian_filter(terrain, sigma=3)

        return terrain

    def add_trees(self, terrain):
        tree_positions = np.random.choice(200*200, self.num_trees, replace=False)
        tree_coordinates = np.array(np.unravel_index(tree_positions, (200, 200))).T
        for x, y in tree_coordinates:
            terrain[x, y] = self.max_elevation
        return terrain, tree_coordinates

    def generate_terrain_with_trees(self):
        self.terrain = self.create_terrain()
        self.terrain, tree_coordinates = self.add_trees(self.terrain)
        return self.terrain, tree_coordinates

    def plot_terrain(self, terrain, tree_coordinates):
        plt.figure(figsize=(10, 10))
        plt.contourf(terrain, cmap='terrain', levels=100)
        plt.colorbar(label='Terrain Elevation')
        plt.scatter(tree_coordinates[:, 1], tree_coordinates[:, 0], color='green', marker='+', s=50)
        plt.title('Colored Terrain Contours with Trees')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def get_terrain_matrix(self):
        return self.terrain