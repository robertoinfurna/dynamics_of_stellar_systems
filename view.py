import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
import os


def view(galaxy, r_max, t=[0], figsize=(20, 7), time_conversion_factor=14.910128):
    n_rows = (len(t) + 3 - 1) // 3
    fig, ax = plt.subplots(n_rows, 3, figsize=(20, 7 * n_rows), subplot_kw={"projection": "3d"})
    ax = ax.flatten()
    
    for i, time in enumerate(t):
    
        if time > galaxy.t[-1]: 
            time = galaxy.t[-1] 
        frame_t = np.digitize(time, galaxy.t) - 1
    
        ax[i].set(
            xlim3d=(-r_max, r_max), 
            ylim3d=(-r_max, r_max), 
            zlim3d=(-r_max, r_max),
            xlabel='X [parsec]', 
            ylabel='Y [parsec]', 
            zlabel='Z [parsec]'
        )
        ax[i].set_aspect("auto")  
        ax[i].set_title(f"t = {time:.3f} i.u. ({time * time_conversion_factor:.3f} Myr)", fontsize=10)
    
        P_x = [star.x[0] for star in galaxy.system[frame_t]]  # X-coordinates
        P_y = [star.x[1] for star in galaxy.system[frame_t]]  # Y-coordinates
        P_z = [star.x[2] for star in galaxy.system[frame_t]]  # Z-coordinates
        
        ax[i].scatter(P_x, P_y, P_z, c='black', s=0.5)
        
        if hasattr(galaxy, 'orbits') and galaxy.orbits:
            frame_t_orbits = np.digitize(time, galaxy.t_orbits) - 1
    
            for j in range(len(galaxy.orbits)):
                orbits_x = [pos[0] for pos in galaxy.orbits[j][:frame_t_orbits+1]] 
                orbits_y = [pos[1] for pos in galaxy.orbits[j][:frame_t_orbits+1]]
                orbits_z = [pos[2] for pos in galaxy.orbits[j][:frame_t_orbits+1]]
                
                ax[i].plot(orbits_x, orbits_y, orbits_z)
                ax[i].scatter(orbits_x[-1], orbits_y[-1], orbits_z[-1], s=10)

    for j in range(len(t), len(ax)):
        fig.delaxes(ax[j])
        
    plt.show()
    
    

def show_animation(galaxy, r_max, time_conversion_factor=14.910128, speed = 600, marker_size = 0.2, filename=None):    


    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(-r_max, r_max)
    ax.set_ylim3d(-r_max, r_max)
    ax.set_zlim3d(-r_max, r_max)
    ax.set_xlabel('X [parsec]')
    ax.set_ylabel('Y [parsec]')
    ax.set_zlabel('Z [parsec]')
    ax.set_aspect("equal")


    star_points = [ax.scatter(galaxy.system[0][i].x[0], 
                              galaxy.system[0][i].x[1], 
                              galaxy.system[0][i].x[2], 
                              color="black", s=marker_size) 
                   for i in range(galaxy.N)]
    
    orbit_end_markers = []
    if hasattr(galaxy, 'orbits'):
        orbits = [ax.plot([], [], [])[0] for _ in galaxy.orbits]
        orbit_end_markers = [ax.scatter([], [], [], s=50) for _ in galaxy.orbits]  # Markers at end of orbits
                    
    def animate(frame): 
        t = galaxy.t[frame]
        plt.title("t = %.2f i.u. (%.2f Myr)" % (t, t * time_conversion_factor), fontsize=20) 

        for i in range(galaxy.N): 
            star_points[i]._offsets3d = (galaxy.system[frame][i].x[0:1], 
                                         galaxy.system[frame][i].x[1:2], 
                                         galaxy.system[frame][i].x[2:])

        if hasattr(galaxy, 'orbits'):
            for i in range(len(galaxy.orbits)):
                frame_t_orbits = np.digitize(t, galaxy.t_orbits) - 1
                orbits_x = [pos[0] for pos in galaxy.orbits[i][:frame_t_orbits+1]]
                orbits_y = [pos[1] for pos in galaxy.orbits[i][:frame_t_orbits+1]]
                orbits_z = [pos[2] for pos in galaxy.orbits[i][:frame_t_orbits+1]]
                
                orbits[i].set_data(orbits_x, orbits_y)
                orbits[i].set_3d_properties(orbits_z)

                if len(orbits_x) > 0:
                    orbit_end_markers[i]._offsets3d = (orbits_x[-1:], orbits_y[-1:], orbits_z[-1:])
                    orbit_end_markers[i].set_color(orbits[i].get_color())
                    orbit_end_markers[i].set_sizes([50])  
                    
                    
    anim = animation.FuncAnimation(fig, animate, frames=len(galaxy.system) - 1, interval=speed)
    display(HTML(anim.to_jshtml()))

    if filename is not None:
        if os.path.exists(filename):
            os.remove(filename)
        anim.save(filename, writer='pillow')  # save as gif
        #anim.save(filename, writer='ffmpeg', fps=30)  #  save as HTML5 video

    plt.close()
    
from sympy import sin, cos, pi
from matplotlib import cm
import matplotlib.colors as mcolors

def sky_projection(
    galaxy,
    time=0,

    i=0,                # inclination angle in degrees
    alpha=0,            # rotation angle in the plane of the sky in degrees
    gamma=0,            # position angle of the line of nodes in degrees
    distance=1000,      # distance to the galaxy in parsecs
    angular_resolution=10,   # angular resolution in arcseconds
    FOV=60,                # field of view in arcmin
    sensitivity=1e-17,      # sensitivity limit in erg/s/cm^2/arcsec^2 (e.g., HST sensitivity)
    saturation=1e-10,       # saturation limit in erg/s/cm^2/arcsec^2
    scale='linear',
    color_scale_inf=1e-17, # can be set manually. By default to sensitivity
    color_scale_sup=1e-10, # can be set manually. By default to saturation
    exp_time=900,           # exposure time in seconds
    spectral_resolution=2000, # spectral resolution (dimensionless)
    ):
 
    if time > galaxy.t[-1]: time=galaxy.t[-1] 
    frame = np.digitize(time,galaxy.t)-1

    # project in the plane of the sky

    beta = np.pi/2-i
    LOS = np.array([cos(alpha)*cos(beta),sin(alpha)*cos(beta),-sin(beta)])
    sky_plane_x = np.array([cos(alpha)*sin(beta)*sin(gamma) + sin(alpha)*cos(gamma),sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma),cos(beta)*sin(gamma)])
    sky_plane_y = np.array([cos(alpha)*sin(beta)*cos(gamma) + sin(alpha)*sin(gamma),sin(alpha)*sin(beta)*cos(gamma) - cos(alpha)*sin(gamma),cos(beta)*cos(gamma)])
    
    star_sky_x = [] 
    star_sky_y = [] 
    v_LOS = []
    for star in galaxy.system[frame]:  
        star_sky_x.append(np.dot(star.x,sky_plane_x))
        star_sky_y.append(np.dot(star.x,sky_plane_y))
        v_LOS.append(np.dot(star.v,LOS))     
    
    # bins
    distance_to_angular_size = 3437.75 / distance    # converts distances in angular separation in the sky, in arcmin
    r_max = FOV/2 / distance_to_angular_size
    dr = angular_resolution / distance_to_angular_size
    print(f"R covered by FOV is {r_max:.2f} pc, dr is {dr:.2f} pc")
    
    bins = [np.arange(-r_max, r_max+dr, dr), np.arange(-r_max, r_max+dr, dr)] 
    LOS_v_matrix = [[ [] for _ in range(len(bins[0]))] for _ in range(len(bins[1]))]  
    
    # build LOS velocity dispersion matrix
    
    for k in range(galaxy.N):
        i,j = np.digitize(star_sky_x[k],bins[0]), np.digitize(star_sky_y[k],bins[1]) 
        if i != 0 and j != 0 and i != len(bins[0]) and j != len(bins[1]): 
            LOS_v_matrix[i-1][j-1].append(float(v_LOS[k]))
      
    LOS_v_mean = np.zeros((len(bins[1])-1,len(bins[0])-1)) 
    LOS_v_dispersion = np.zeros((len(bins[1])-1,len(bins[0])-1)) 
    LOS_v_disp_error = np.zeros((len(bins[1])-1,len(bins[0])-1))
    for i in range(len(bins[1])-1):
        for j in range(len(bins[0])-1):
            N_bin = len(LOS_v_matrix[j][i]) 
            LOS_v_mean[i][j] = np.mean(LOS_v_matrix[j][i])  if N_bin > 0 else 0
            LOS_v_dispersion[i][j] = np.std(LOS_v_matrix[j][i])  if N_bin > 1 else 0
            LOS_v_disp_error[i][j] = LOS_v_dispersion[i][j]/np.sqrt(2*(N_bin-1)) if N_bin > 1 else 0

    
        
    bins = np.array(bins) * distance_to_angular_size
    star_sky_x = np.array(star_sky_x) * distance_to_angular_size
    star_sky_y = np.array(star_sky_y) * distance_to_angular_size
    
    # limits
    mass_light_ratio = 1
    L_sun = 1e33 # erg / s luminosity of sun (in some band)
    L_star = L_sun * galaxy.m / mass_light_ratio # erg / s (mass light ratio in M_sun/L_sun, adimensional)
    distance_cm = distance / 3.24078e-19  # distance in cm
    flux_star = L_star / (4*np.pi * distance_cm**2) # erg / s / cm^2
    SB_star = flux_star / (angular_resolution**2) # erg / s / cm^2 / arcmin^2
    
    n_min = sensitivity / SB_star # minimum number of stars (of 1 solar mass) that must fall in a bin to give a non zero signal
    n_max = saturation / SB_star

    print("Signal above threshold if more than %.1f stars/bin" %(n_min))
    print("Saturation point when > %1f stars/bin" %(n_max))

    if color_scale_inf > sensitivity:
        n_min = color_scale_inf  / SB_star
    if color_scale_sup < saturation:
        n_max = color_scale_sup  / SB_star

    
    velocity_conversion = 15.248335 # converts velocity from internal units to km/s
    deltav_min = 2.99792458 * 10**5 / velocity_conversion / spectral_resolution 

    print(f"Minimum velocity detectable with spectral resolution {spectral_resolution:.1f} is {deltav_min*velocity_conversion:.2f} km/s")
    

    # plots 

    fig, axes = plt.subplots(1, 3, figsize=(12,4),layout="constrained") 
    fig.suptitle(f"Angular resolution = {angular_resolution:.2f} arcsec, FOV = {FOV:.2f} arcmin, sensitivity = {sensitivity:.2e} erg/s/cm^2/arcsec^2 \n"
    	f"saturation point = {saturation:.2e} erg/s/cm^2/arcsec^2, spectral resolution = {spectral_resolution:.0f}, exposure time = {exp_time:.2f} s")

    
    for i in range(3):
        axes[i].set_aspect('equal') 
        axes[i].set_xlabel("ra [arcmin]") 
        axes[i].set_ylabel("dec [arcmin]")
  
    
    # surface brightness
    
    cmap = cm.rainbow
    cmap.set_under('w')  
    if scale == 'linear':
        norm = mcolors.Normalize(vmin = n_min,vmax = n_max)
    if scale == 'log':
        norm=mcolors.LogNorm(vmin=n_min,vmax=n_max)
    S = axes[0].hist2d(star_sky_x, star_sky_y, bins=bins,cmap=cmap,norm=norm) 
    N = S[0]  

    cbar_sb = fig.colorbar(S[3], ax=axes[0], location='bottom', label="surface brightness [erg/s/cm^2/arcsec$^2$]") 
    cbar_sb.set_ticks(np.linspace(n_min, n_max, 5))
    tick_labels = [f'{SB_star * n:.2e}' for n in np.linspace(n_min, n_max, 5)]
    cbar_sb.set_ticklabels(tick_labels)
    for label in cbar_sb.ax.get_xticklabels():
        label.set_rotation(90)
    
    masked_LOS_v_mean = np.where((N < n_min) | (np.abs(LOS_v_mean) < deltav_min), np.nan, LOS_v_mean) * velocity_conversion            #delta_v_min
    cmap = cm.bwr 
    cmap.set_bad(color='yellow')  
    
    V_mean = axes[1].imshow(masked_LOS_v_mean, interpolation='none', extent=[-FOV/2, FOV/2,-FOV/2, FOV/2],origin='lower',cmap=cmap)  
    fig.colorbar(V_mean, ax=axes[1], location='bottom', label="$v_\parallel [km/s]$") 
    
    # velocity dispersion

    masked_LOS_v_dispersion = np.where((N < n_min) | (LOS_v_dispersion < deltav_min), np.nan, LOS_v_dispersion) * velocity_conversion     
    cmap = cm.viridis
    cmap.set_bad(color='white')  
    
    V_disp = axes[2].imshow(masked_LOS_v_dispersion, interpolation='none', extent=[-FOV/2, FOV/2,-FOV/2, FOV/2], origin='lower',cmap=cmap,vmin=deltav_min, vmax = 100*deltav_min)  
    cbar_V_disp = fig.colorbar(V_disp, ax=axes[2], location='bottom', label="$\sigma_\parallel [km/s]$")    
    
    
 
 
    
    
    
    
