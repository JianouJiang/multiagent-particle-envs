import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import griddata


# Set parameters
nx, ny = 51, 51  # number of grid points in x and y direction
lx, ly = 1.0, 1.0  # domain dimensions
dx, dy = lx/(nx-1), ly/(ny-1)  # grid spacing
nt = 2000  # number of time steps
rho = 1.0  # fluid density
mu = 0.1  # Dynamic viscosity of water in kg/(m·s)
nu = mu / rho  # Kinematic viscosity in m^2/s
kappa_S = 2e2  # Stretching stiffness in N/m
kappa_B = 0*1e2 # Bending stiffness in N·m, 1e-2 to 1e2
dt = 0.000001  # initial time step size
t  = 1 # total time
g =  -9.81 # gravity
CFL = 0.01
alpha = 0.01 # 1.4e-7  # thermal diffusivity of water
Q = 0  # no heat source
Cp = 4186  # specific heat capacity of water
beta = 0.0 # thermal expansion coefficient
T0 = 0.0 # reference temperature

X, Y = np.meshgrid(np.linspace(0, lx, nx), np.linspace(0, ly, ny))


# Initialize fields
u = np.ones((ny, nx)) 
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
T = np.zeros((ny, nx)) 
b = np.zeros((ny, nx))

# Define the lid velocity and temperature
u_top = 0.0
T_top = 0.0
# Create a Gaussian function for the bottom boundary
x = np.linspace(-1.0, 1.0, nx)
gaussian_flux = np.exp(-5 * x**2) 

def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    return b

def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)
    pn = p.copy()
    
    for q in range(nt):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1,1:-1])

        # Lid-driven cavity B.C
        '''
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
        p[-1, :] = 0         # p = 0 at y = 2
        '''
        # Transmissive boundary conditions for p
        p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[-1, :] = p[-2, :]  # dp/dy = 0 at y = 2
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0

        return p

def Navier_Stokes(nt, u, v, dt, dx, dy, p, rho, nu, T, alpha, Q, Cp, beta, T0):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    Tn = np.empty_like(T)
    b = np.zeros((ny, nx))

    list_of_cylinders = [(0.5,0.5,0.1)] # , (0.6,0.5,0.1)]
    # Define the geometries and get the initial Lagrangian coordinates and dr
    geometries, dr = define_geometry(dx, dy, list_of_cylinders)
    geometries_prev = geometries.copy()  # Store the previous configuration of the geometries
    geometries_B = geometries.copy()   # Store the desired configuration of the geometries, we assume the initial state is the desired state
    
    ti = 0
    n = 0
    while n<nt and ti<t:
        ti = ti + dt
        n = n + 1
        print("dt:"+str(dt)+"ti"+str(ti)+" n:"+str(n)+" nt:"+str(nt))
        un = u.copy()
        vn = v.copy()
        Tn = T.copy()

        # Calculate the total force on each geometry
        F_total = calculate_total_force_lagrangian(geometries, geometries_prev, geometries_B, u, v, dt, mu, kappa_S, kappa_B, dr)

        # Spread the force from the Lagrangian points to the Eulerian grid
        fx, fy = spreading_function(F_total, geometries, dx, dy, nx, ny)

        # Update the previous configuration of the geometries
        geometries_prev = geometries.copy()

        # Update the geometries based on the velocity field
        geometries = update_geometries(geometries, u, v, dt, dx, dy)


        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) +
                         beta * (T[1:-1, 1:-1] - T0) * dt + 
                         fx[1:-1, 1:-1] * dt)
        print("long term:")
        long_term = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) +
                         beta * (T[1:-1, 1:-1] - T0) * dt)
        print(long_term.max())
        print(long_term.min())
        print("fx[1:-1, 1:-1] * dt: ")
        short_term = fx[1:-1, 1:-1] * dt
        print(short_term.max())
        print(short_term.min())

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                       (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                       (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                       (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))+ 
                        beta * (T[1:-1, 1:-1] - T0) * dt +
                        fy[1:-1, 1:-1] * dt
                        ) 

        # Add the energy equation
        T[1:-1, 1:-1] = (Tn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * 
                         (Tn[1:-1, 1:-1] - Tn[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy * 
                         (Tn[1:-1, 1:-1] - Tn[0:-2, 1:-1]) +
                         alpha * (dt / dx**2 * 
                         (Tn[1:-1, 2:] - 2 * Tn[1:-1, 1:-1] + Tn[1:-1, 0:-2]) +
                         dt / dy**2 * 
                         (Tn[2:, 1:-1] - 2 * Tn[1:-1, 1:-1] + Tn[0:-2, 1:-1])) +
                         Q / (rho * Cp) * dt)

        # Lid driven cavity Boundary conditions
        '''
        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = u_top  # lid velocity
        v[0, :] = 0
        v[-1, :]=0
        v[:, 0] = 0
        v[:, -1] = 0
        '''
        # Transmissive boundary conditions for u, v
        u[0, :] = u[1, :]
        u[:, -1] = u[:, -2]
        u[-1, :] = u[-2, :]
        v[0, :] = v[1, :]
        v[-1, :]= v[-2, :]
        v[:, -1] = v[:, -2]
        # Fixed x-direction velocity at the left boundary
        u[:, 0] = 1 + gaussian_flux*0.1 #1
        v[:, 0] = 0

        # Set the temperature boundary conditions
        T[0, :] = 0
        T[:, 0] = T[1, :] + dy * gaussian_flux
        T[:, -1] = 0
        T[-1, :] = T_top  # lid temperature


        u_max = np.max(np.abs(u))
        v_max = np.max(np.abs(v))
        dt = CFL * min(dx / u_max, dy / v_max, dx**2 / (4*alpha), dy**2 / (4*alpha))  # calculate dt based on modified CFL condition

        # plot_results(X, Y, u, v, p, T, fx, geometries)

    return u, v, p, T, fx, geometries

# define geometry in lagrangian, for ibm
def define_geometry(dx, dy, list_of_cylinders):
    geometries = []
    dr = []
    
    for cylinder in list_of_cylinders:
        center_x, center_y, radius = cylinder
        circumference = 2 * np.pi * radius
        num_points = int(circumference / min(dx, dy))
        theta = np.linspace(0, 2*np.pi, num_points)
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)
        
        geometries.append((x, y))
        dr.append(circumference / num_points)
    
    return geometries, dr



def calculate_bending_force_lagrangian(X, X_B, dr, kappa_B):
    # Calculate the second derivative of X with respect to q
    d2x_dq2 = np.gradient(np.gradient(X[0], dr), dr)
    d2y_dq2 = np.gradient(np.gradient(X[1], dr), dr)

    # Calculate the second derivative of X_B with respect to q
    d2x_B_dq2 = np.gradient(np.gradient(X_B[0], dr), dr)
    d2y_B_dq2 = np.gradient(np.gradient(X_B[1], dr), dr)

    # Calculate the bending force
    Fb = -kappa_B * (np.array([d2x_dq2, d2y_dq2]) - np.array([d2x_B_dq2, d2y_B_dq2]))

    return Fb



def calculate_compressing_force_lagrangian(X, kappa_S, dr):
    # Calculate the first derivative of X with respect to q
    dx_dq = np.gradient(X[0], dr)
    dy_dq = np.gradient(X[1], dr)

    # Calculate the magnitude of the first derivative
    mag_dX_dq = np.sqrt(dx_dq**2 + dy_dq**2)

    # Calculate the stretching force
    Fs = kappa_S * ((mag_dX_dq - 1) * np.array([dx_dq, dy_dq]) / mag_dX_dq)

    return Fs




def calculate_motion_force_lagrangian(X, X_prev, u, v, dt, mu):
    # Calculate the velocity of the boundary points
    dx_dt = (X[0] - X_prev[0]) / dt
    dy_dt = (X[1] - X_prev[1]) / dt

    # Define the points where the velocity is known
    y_grid, x_grid = np.mgrid[0:u.shape[0], 0:u.shape[1]]
    points = np.array([x_grid.ravel(), y_grid.ravel()]).T

    # Define the points where you want to interpolate the velocity
    xi = np.array([X[0].ravel(), X[1].ravel()]).T

    # Interpolate the velocity of the fluid at the boundary points
    u_interp = griddata(points, u.ravel(), xi, method='cubic').reshape(X[0].shape)
    v_interp = griddata(points, v.ravel(), xi, method='cubic').reshape(X[1].shape)

    # Calculate the motion force
    Fv = mu * (np.array([dx_dt, dy_dt]) - np.array([u_interp, v_interp]) ) / (dt * mu) * 0 #  / (dt * mu) so that the force will enhance no-slip B.C

    return Fv


def calculate_gravitational_force_lagrangian(X, rho, g):
    # X is the array of Lagrangian points
    # rho is the density of the material at each Lagrangian point
    # g is the gravitational acceleration

    # Initialize the gravitational force array
    Fg = np.zeros_like(X)

    # Calculate the gravitational force at each Lagrangian point
    Fg[0] = -rho * g # Force in the x-direction
    Fg[1] =  np.zeros_like(X[0])  # No force in the y-direction

    return Fg




def calculate_total_force_lagrangian(geometries, geometries_prev, geometries_B, u, v, dt, mu, kappa_S, kappa_B, dr_list):
    # geometries is a list of tuples, where each tuple contains the x and y coordinates of the points defining the boundary of one object
    # geometries_prev is the previous configuration of the boundary points
    # geometries_B is the desired configuration of the boundary points
    # u and v are the velocity fields of the fluid
    # dt is the time step
    # mu is the dynamic viscosity of the fluid
    # kappa_S is the stretching stiffness constant
    # kappa_B is the bending stiffness constant
    # dr is the spacing between the points in the q parameter

    F_total = []  # Total forces

    for (X, X_prev, X_B, dr) in zip(geometries, geometries_prev, geometries_B, dr_list):
        # Calculate the bending force
        Fb = calculate_bending_force_lagrangian(X, X_B, kappa_B, dr)

        # Calculate the stretching force
        Fs = calculate_compressing_force_lagrangian(X, kappa_S, dr)

        # Calculate the motion force
        Fv = calculate_motion_force_lagrangian(X, X_prev, u, v, dt, mu)

        # Calculate the gravity force for the geometry only!
        Fg = calculate_gravitational_force_lagrangian(X, rho, g)

        # Calculate the total force
        # F_total.append(np.array(Fb) + np.array(Fs) + np.array(Fv))
        print("Fs.max and min:")
        print(Fs.max())
        print(Fs.min())
        F_total.append(np.array(Fv)+ np.array(Fg))

    return F_total

'''
def spreading_function(F_total, geometries, dx, dy, nx, ny):
    # F_total is a list of arrays, where each array contains the total forces for one object
    # geometries is a list of tuples, where each tuple contains the x and y coordinates of the points defining the boundary of one object
    # dx and dy are the grid spacings in the x and y directions
    # nx and ny are the number of grid points in the x and y directions
    fx = np.zeros((ny, nx))
    fy = np.zeros((ny, nx))

    for F, (x, y) in zip(F_total, geometries):
        # Transpose the force array
        F = F.T

        # Calculate the indices of the Eulerian grid points that are closest to each Lagrangian point
        i = np.minimum(np.floor(x / dx).astype(int), nx - 1)
        j = np.minimum(np.floor(y / dy).astype(int), ny - 1)

        # Calculate the distances from the Lagrangian points to the Eulerian grid points
        dx1 = x - i * dx
        dy1 = y - j * dy
        dx2 = dx - dx1
        dy2 = dy - dy1

        # Calculate the weights for the spreading
        w1 = dx2 * dy2 / (dx * dy)
        w2 = dx1 * dy2 / (dx * dy)
        w3 = dx2 * dy1 / (dx * dy)
        w4 = dx1 * dy1 / (dx * dy)

        # Spread the force from the Lagrangian points to the Eulerian grid points
        for idx in range(len(i)):
            fx[j[idx], i[idx]] += F[idx, 0] * w1[idx]
            fx[j[idx], i[idx]+1] += F[idx, 0] * w2[idx]
            fx[j[idx]+1, i[idx]] += F[idx, 0] * w3[idx]
            fx[j[idx]+1, i[idx]+1] += F[idx, 0] * w4[idx]
            fy[j[idx], i[idx]] += F[idx, 1] * w1[idx]
            fy[j[idx], i[idx]+1] += F[idx, 1] * w2[idx]
            fy[j[idx]+1, i[idx]] += F[idx, 1] * w3[idx]
            fy[j[idx]+1, i[idx]+1] += F[idx, 1] * w4[idx]

    return fx, fy
'''
def spreading_function(F_total, geometries, dx, dy, nx, ny, sigma=1.5):
    fx = np.zeros((ny, nx))
    fy = np.zeros((ny, nx))

    for F, (x, y) in zip(F_total, geometries):
        # Transpose the force array
        F = F.T

        # Calculate the indices of the Eulerian grid points that are closest to each Lagrangian point
        i = np.minimum(np.floor(x / dx).astype(int), nx - 1)
        j = np.minimum(np.floor(y / dy).astype(int), ny - 1)

        # Calculate the distances from the Lagrangian points to the Eulerian grid points
        dx1 = x - i * dx
        dy1 = y - j * dy

        # Define the Gaussian kernel
        kernel = lambda r, sigma: np.exp(-0.5 * (r / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

        # Spread the force from the Lagrangian points to the Eulerian grid points
        for idx in range(len(i)):
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    if 0 <= i[idx]+di < nx and 0 <= j[idx]+dj < ny:
                        weight_x = kernel(dx1[idx] - di*dx, sigma)
                        weight_y = kernel(dy1[idx] - dj*dy, sigma)
                        weight = weight_x * weight_y
                        fx[j[idx]+dj, i[idx]+di] += F[idx, 0] * weight
                        fy[j[idx]+dj, i[idx]+di] += F[idx, 1] * weight

    return fx, fy





def update_geometries(geometries, u, v, dt, dx, dy):
    updated_geometries = []

    # Create a grid of points in the Eulerian frame
    y = np.arange(0, u.shape[0]*dy, dy)
    x = np.arange(0, u.shape[1]*dx, dx)
    points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    for X in geometries:
        x, y = X

        # Interpolate the velocity of the fluid at the boundary points
        u_interp = griddata(points, u.ravel(), (x, y), method='cubic')
        v_interp = griddata(points, v.ravel(), (x, y), method='cubic')

        # Update the coordinates of the boundary points based on the interpolated velocity
        x_new = x + u_interp * dt 
        y_new = y + v_interp * dt 

        # Store the updated coordinates
        updated_geometries.append((x_new, y_new))

    return updated_geometries

# This function is not used currently!
def update_geometries_with_inertia(geometries, u, v, dt, dx, dy, mass, old_velocity):
    # geometries is a list of tuples, where each tuple contains the x and y coordinates of the points defining the boundary of one object
    # u and v are the velocity fields of the fluid
    # dt is the time step
    # dx and dy are the grid spacings in the x and y directions
    # mass is the mass of the cylinder
    # old_velocity is the velocity of the cylinder at the previous time step

    updated_geometries = []  # Updated geometries
    new_velocity = np.zeros_like(old_velocity)

    for i, X in enumerate(geometries):
        x, y = X

        # Interpolate the velocity of the fluid at the boundary points
        u_interp = griddata(points, u.ravel(), (x, y), method='cubic')
        v_interp = griddata(points, v.ravel(), (x, y), method='cubic')

        # Calculate the acceleration due to the fluid forces
        acceleration_x = u_interp / mass
        acceleration_y = v_interp / mass

        # Update the velocity of the boundary points based on the acceleration
        new_velocity[i, 0] = old_velocity[i, 0] + acceleration_x * dt
        new_velocity[i, 1] = old_velocity[i, 1] + acceleration_y * dt

        # Update the coordinates of the boundary points based on the velocity
        x_new = x + new_velocity[i, 0] * dt
        y_new = y + new_velocity[i, 1] * dt 

        # Store the updated coordinates
        updated_geometries.append((x_new, y_new))

    return updated_geometries, new_velocity



def plot_results(X, Y, u, v, p, T, F, geometries):
    plt.figure(figsize=(11,7), dpi=100)

    # Change the contour plot to a heatmap
    plt.imshow(u, extent=[0, lx, 0, ly], origin='lower', cmap='viridis')#, vmin=-1.5, vmax=1.5)
    plt.colorbar(label='Pressure')

    # Plot the geometries
    for geometry in geometries:
        x, y = geometry
        plt.plot(x, y, 'r-', linewidth=2)

    # Adjust the streamplot to match the heatmap
    plt.streamplot(X, Y, u, v, color='k')

    plt.title('Pressure Heatmap with Velocity Streamlines and Geometries')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    print(u.max())
    print(v.max())


# Run the solver
u, v, p, T, F, geometries = Navier_Stokes(nt, u, v, dt, dx, dy, p, rho, nu, T, alpha, Q, Cp, beta, T0)


# Plot the results

plot_results(X, Y, u, v, p, T, F, geometries)




