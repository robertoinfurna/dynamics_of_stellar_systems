import matplotlib.pyplot as plt
import numpy as np



def view(galaxy, r_max, t=[0], orbit_indexes=None, figsize=(20, 7), time_conversion_factor = 0.067069):
    """
    Visualizes the state of a galaxy system at specified times using 3D scatter plots.
    
    This function generates 3D plots of the galaxy's stars at different time points.
    It also supports visualizing the orbits of specific stars by plotting their trajectories.

    Parameters:
    - galaxy (Galaxy): The galaxy object containing the system of stars and their state over time.
    - r_max (float): The maximum range for the x, y, and z axes in the plot (in parsecs).
    - t (list of floats): A list of times at which to visualize the galaxy system. Default is [0], meaning the initial state.
    - orbit_indexes (list of int, optional): A list of star indices whose orbits will be tracked and plotted. If None, no orbits are plotted.
    - figsize (tuple of int, optional): The figure size for the plot. Default is (20, 7) for a wide view.
    
    Returns:
    None (displays the plots in the current figure).

    This function creates a series of 3D scatter plots of stars' positions in the galaxy at different times.
    If `orbit_indexes` are provided, the function will also plot the orbits of the corresponding stars up to the specified times.

    The time points are plotted across multiple subplots, with up to three time points per row.
    """
         
    n_rows = (len(t) + 3 - 1) // 3
    fig, ax = plt.subplots(n_rows,3,figsize=(20,7 * n_rows), subplot_kw={"projection": "3d"})
    #if len(t) == 1:
    #    ax = [ax]
    ax = ax.flatten()
    
    for i,time in enumerate(t):
    
        if time > galaxy.t[-1]: time = galaxy.t[-1] 
        frame_t = np.digitize(time,galaxy.t)-1
    
        ax[i].set(xlim3d=(-r_max, r_max), ylim3d=(-r_max, r_max), zlim3d=(-r_max, r_max),
               xlabel='X [parsec]', ylabel='Y [parsec]', zlabel='Z [parsec]')
        ax[i].set_aspect("auto")  
        ax[i].set_title(f"t = {time:.3f} i.u. ({time * time_conversion_factor:.3f} Myr)", fontsize=10)
    
        P_x = [star.x[0] for star in galaxy.system[frame_t]]  # X-coordinates
        P_y = [star.x[1] for star in galaxy.system[frame_t]]  # Y-coordinates
        P_z = [star.x[2] for star in galaxy.system[frame_t]]  # Z-coordinates
        
        ax[i].scatter(P_x, P_y, P_z, c='black', s=0.5)
    
        if orbit_indexes:
            frame_t_orbits = np.digitize(time, galaxy.t_orbits) - 1 

            for j, index in enumerate(orbit_indexes):

                orbits_x = [pos[0] for pos in galaxy.orbits[j][:frame_t_orbits + 1]]
                orbits_y = [pos[1] for pos in galaxy.orbits[j][:frame_t_orbits + 1]]
                orbits_z = [pos[2] for pos in galaxy.orbits[j][:frame_t_orbits + 1]]
                
                ax[i].plot(orbits_x, orbits_y, orbits_z, label=f"Orbit {index}")
                ax[i].scatter(orbits_x[-1], orbits_y[-1], orbits_z[-1], s=10)

    for j in range(len(t), len(ax)):
        fig.delaxes(ax[j])
    plt.show()
    
    
