import numpy as np
import matplotlib.pyplot as plt

# Constants
nx, ny = 400, 100  # size of the domain
ly = ny-1  # height of the domain
cx, cy, r = nx//4, ny//2, ny//9  # coordinates and radius of the cylinder
uLB = 0.04  # velocity in lattice units
nulb = 0.04  # kinematic viscosity in lattice units
omega = 1.0 / (3*nulb+0.5)  # relaxation parameter

# Lattice constants
v = np.array([ [ 1,  1], [ 1,  0], [ 1, -1], [ 0,  1], [ 0,  0],
               [ 0, -1], [-1,  1], [-1,  0], [-1, -1] ])  # velocities
t = 1/36 * np.ones(9)  # weights
t[[3,4,5]] = 1/9
t[0] = 4/9

def macroscopic(fin):
    rho = np.sum(fin, axis=0)
    u = np.dot(v.transpose(), fin.transpose((1,0,2))) / rho
    return rho, u

def equilibrium(rho, u):
    cu = 3.0 * np.tensordot(v, u, axes=([1],[0]))
    feq = np.zeros((9,nx,ny))
    for i in range(9):
        feq[i,:,:] = rho*t[i]*(1 + cu[i] + 0.5*cu[i]**2 - 1.5*np.sum(u**2,0))
    return feq

# Initialization
fin = equilibrium(1, np.array([uLB, 0]))

def simulate_flow(fin, nx, ny, uLB, omega, v, cx, cy, r):
    # Create a mask for the cylinder
    Y, X = np.ogrid[:ny, :nx]
    mask = (X - cx)**2 + (Y - cy)**2 < r**2

    # Main loop
    for time in range(2):
        fin[[0,1,2]], fin[[8,7,6]] = fin[[8,7,6]], fin[[0,1,2]]

        rho, u = macroscopic(fin)
        feq = equilibrium(rho, u)
        fout = fin - omega * (fin - feq)
        for i in range(9):
            fout[i, np.where((v[i,0]*u[0]+v[i,1]*u[1]) < 0)] = fin[i, np.where((v[i,0]*u[0]+v[i,1]*u[1]) < 0)]
        for i in range(9):
            fin[i,:,:] = np.roll(np.roll(fout[i,:,:], v[i,0], axis=0), v[i,1], axis=1 )
        fin[:,0,:] = equilibrium(1, np.array([uLB, 0]))[:,0,:]
        fin[:,nx-1,:] = fin[:,nx-2,:]

        # Update the distribution functions inside the cylinder
        for i in range(9):
            fin[i][mask.T] = equilibrium(1, np.array([0, 0]))[i][mask.T]

        # Check for stability: stop the simulation if any NaN values are detectedI apologize for the incomplete response. Here is the complete version of the code:
        if np.isnan(fin).any():
            print(f"Unstable simulation at timestep {time}. Stopping simulation.")
            break

    return fin

fin = simulate_flow(fin, nx, ny, uLB, omega, v, cx, cy, r)

rho, u = macroscopic(fin)

def plot_flow(u, rho):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Compute pressure from density
    pressure = rho / 3.0

    # Plot pressure as a heatmap
    heatmap = ax.imshow(pressure.transpose(), cmap='viridis', origin='lower', extent=(0, nx, 0, ny))

    # Add a colorbar for the pressure
    cbar = fig.colorbar(heatmap)
    cbar.set_label('Pressure')

    # Plot velocity as streamlines
    ax.streamplot(np.arange(nx), np.arange(ny), u[0].transpose(), u[1].transpose(), color='white')

    # Set the aspect ratio of the plot
    ax.set_aspect('equal')

    # Show the plot
    plt.show()

# Call the function to plot the flow field
plot_flow(u, rho)
