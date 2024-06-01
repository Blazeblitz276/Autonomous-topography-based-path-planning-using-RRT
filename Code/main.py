from TerrainGen import TerrainGenerator
from Agent import Agent,initialize_terrain_and_cost,plt,on_key,on_mouse_click


if __name__ == "__main__":
    terrain_gen = TerrainGenerator(seed=125)
    terrain, cost_matrix = initialize_terrain_and_cost(terrain_gen)
    agent = Agent(start_position=(1, 1), step_size=2, iterations=5000, discovery_radius= 4)
    exploration_finished = [False]
    start_end_points = []
    rrt_popup_shown = [False]

    # Create the first plot for exploration
    fig_explore, ax_explore = plt.subplots()
    fig_explore.colorbar(plt.cm.ScalarMappable( cmap='terrain'), ax=ax_explore)
    ax_explore.imshow(terrain, cmap='terrain', origin='lower')
    fig_explore.canvas.mpl_connect('key_press_event', lambda event: on_key(event, agent, terrain, cost_matrix, ax_explore, exploration_finished, start_end_points, rrt_popup_shown))
    fig_explore.canvas.mpl_connect('button_press_event', lambda event: on_mouse_click(event, start_end_points, ax_explore, cost_matrix, exploration_finished, rrt_popup_shown, agent, terrain))
    plt.show()
    