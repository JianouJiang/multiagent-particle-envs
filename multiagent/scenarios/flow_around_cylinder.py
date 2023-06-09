import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
from collections import defaultdict
from scipy.signal import convolve2d
import os



"""

Modified flow past multiple cylinders for an isothermal fluid by Philip Mocz.
This code acts as a base for further development of MARL.

"""
# Simulation parameters
X                      = 1.2    # X length [m]
Y                      = 0.4    # Y length [m]
Nx                     = 300     # resolution x-dir
Ny                     = 100     # resolution y-dir
dx                     = X / Nx # cell size in x-dir
dy                     = Y / Ny # cell size in y-dir
rho0                   = 100    # average density
tau                    = 0.6    # collision timescale
T                      = 11      # total running time [s]
Nt                     = 11   # number of timesteps
dt                     = dx # T/Nt   # time step
nu                     = (2*tau - 1) / 6  # kinematic viscosity
initial_ux             = 1.8
characteristic_length  = Ny/9 * 2
Re                     = initial_ux * characteristic_length / nu / 50
plotRealTime = True # switch on for plotting as the simulation goes along
#circles = [(Nx/4, Ny/2, characteristic_length/2),  (Nx/3, Ny/2, characteristic_length/2)    ]


def read_stopit_file():
    with open('stopit.txt', 'r') as file:
        value = file.read().strip()  # strip is used to remove leading/trailing whitespaces
        if value == '1':
            return 1
        else:
            return 0

def boundary_force_distribution(force_field):
    # Initialize forces
    force_north = np.array([0.0, 0.0])
    force_south = np.array([0.0, 0.0])
    force_west = np.array([0.0, 0.0])
    force_east = np.array([0.0, 0.0])
    
    # Find boundaries
    non_zero_indices = np.nonzero(force_field)
    min_i, max_i = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
    min_j, max_j = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
    
    # Calculate forces
    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            if np.any(force_field[i][j] != 0):
                if i == min_i:
                    force_north += force_field[i][j]
                elif i == max_i:
                    force_south += force_field[i][j]
                if j == min_j:
                    force_west += force_field[i][j]
                elif j == max_j:
                    force_east += force_field[i][j]
    
    return [force_north, force_south, force_west, force_east]


def surface_vector(boundary_mask, dA=0.1):

    dA = np.sqrt(dx**2 + dy**2)  # Define the cell area

    # Define the kernel for convolution to calculate gradient
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Perform convolution operation to calculate gradient at each cell
    gradient_x = convolve2d(boundary_mask, kernel, mode='same', boundary='symm')
    gradient_y = convolve2d(boundary_mask, np.transpose(kernel), mode='same', boundary='symm')

    # Calculate the magnitude of the gradient (assumed to be the cell area)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    magnitude[boundary_mask == 0] = 0  # Set magnitude to 0 for non-boundary cells

    # Normalize the gradient to get the direction vector
    direction_x = np.where(magnitude != 0, gradient_x / magnitude, 0)
    direction_y = np.where(magnitude != 0, gradient_y / magnitude, 0)

    # Multiply the direction by the cell area (magnitude) to get the surface vector
    surface_vector_array = np.dstack((direction_x * magnitude * dA, direction_y * magnitude * dA))

    return surface_vector_array


def compute_viscous_drag_force(ux, uy, boundary_mask, surface_vector_mask):
    # Compute the velocity gradients using central difference
    duy_dx = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / (2 * dx)
    dux_dy = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / (2 * dy)

    # Compute the viscous stress tensor
    tau_xx = 2 * nu * (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) / (2 * dx)
    tau_yy = 2 * nu * (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) / (2 * dy)
    tau_xy = nu * (duy_dx + dux_dy)

    # Compute the viscous drag force
    F_drag_visc_x = tau_xx * surface_vector_mask[:,:,0] + tau_xy * surface_vector_mask[:,:,1]
    F_drag_visc_y = tau_xy * surface_vector_mask[:,:,0] + tau_yy * surface_vector_mask[:,:,1]

    # Combine them
    F_drag_visc = np.stack((F_drag_visc_x, F_drag_visc_y), axis=2)

    # Sum up the drag force over all cells
    total_drag_force_x = np.sum(F_drag_visc_x[boundary_mask == 1])
    total_drag_force_y = np.sum(F_drag_visc_y[boundary_mask == 1])

    return total_drag_force_x, total_drag_force_y, F_drag_visc

# TODO: this is WRONG, need to modify later...
def compute_drag(ux, uy, boundary_masks, rho=rho0, eta=nu):
    dA = np.sqrt(dx**2+dy**2) * np.sqrt(dx**2+dy**2)
    drag_dict = defaultdict(float)
    drag_mask_dict = defaultdict(float)
    for idx, boundary_mask in enumerate(boundary_masks):

        # direction_mask = mark_velocity_direction(ux, uy, boundary_mask)

        surface_vector_mask  = surface_vector(boundary_mask, dA)
      
        # convert it to m/s in real world scale from lattice boltzmann scale
        ux = ux * dx/dt
        uy = uy * dy/dt
        # Total drag = Pressure Drag + viscous drag

        # Compute the velocity magnitude
        V = np.sqrt(ux**2 + uy**2)

        

        # Compute the dynamic pressure: delta_p = 1/2 * rho * (v_front^2 - v_back^2), F_d = delta_p * A
        P_dynamic = 0.5 * rho * V**2 

        '''
        plt.cla()
        
        plt.imshow(P_dynamic, cmap='bwr')

        plt.clim(P_dynamic.min(), P_dynamic.max())
        ax = plt.gca()
        plt.show()
        '''

        # Compute the drag force for each cell
        F_drag_pres = P_dynamic[:,:,np.newaxis] * surface_vector_mask

        total_horizontal_force = np.sum(F_drag_pres[:,:,0])
        total_vertical_force = np.sum(F_drag_pres[:,:,1])


        # Sum up the drag force for the boundary cells
        Drag_pres = np.sqrt(total_horizontal_force**2 + total_vertical_force**2) # np.sum(F_drag_pres[boundary_mask])


        # Compute viscous drag
        total_drag_visc_x, total_drag_visc_y, F_drag_visc = compute_viscous_drag_force(ux, uy, boundary_mask, surface_vector_mask)
        Drag_visc = np.sqrt(total_drag_visc_x ** 2 + total_drag_visc_y ** 2)

        # Compute total drag
        drag_dict[idx] = Drag_pres + Drag_visc

        # Compute drag arrays
        #drag_mask_dict[idx] = [F_drag_pres, F_drag_visc] 

        drag_mask_dict[idx] = [x + y for x, y in zip(boundary_force_distribution(F_drag_pres), boundary_force_distribution(F_drag_visc))] # boundary_force_distribution returns [force_north, force_south, force_west, force_east] 
        
        

    return drag_dict, drag_mask_dict


def get_boundary_values(mask):

    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]])
    
    conv_mask = convolve(mask, kernel, mode='constant', cval=0.0)
    boundary_mask = np.logical_and(conv_mask > 0, mask == 0)
    
    return boundary_mask.astype(int)


def create_cylinders(circles):
    cylinders = []
    for x, y, r in circles:
        cylinder = np.zeros((Ny, Nx), dtype=bool)
        y_coords, x_coords = np.ogrid[:Ny, :Nx]
        mask = (x_coords - x)**2 + (y_coords - y)**2 <= r**2
        cylinder[mask] = True
        cylinders.append(cylinder)

    return cylinders


def create_cylinder_as_one(circles):
    cylinders = np.zeros((Ny, Nx), dtype=bool)
    for x, y, r in circles:
      y_coords, x_coords = np.ogrid[:Ny, :Nx]
      cylinder = (x_coords - x)**2 + (y_coords - y)**2 <= r**2
      cylinders[cylinder] = True

    return cylinders

# Function to check if any cylinders overlap
def check_overlap(circles):
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            if np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2) < ri + rj:
                return True
    return False


def compute_fluid_dynamics(circles, F_last_time_step=None, iteration=0):

    # Get the absolute path to the current working directory
    cwd = os.getcwd()
    # Create the "img" folder if it doesn't exist
    img_folder = os.path.join(cwd, "img")
    os.makedirs(img_folder, exist_ok=True)

    """ Lattice Boltzmann Simulation """
    
    print("Re="+str(Re))
    # Lattice speeds / weights
    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
    cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
    weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1
    
    # Initial Conditions
    np.random.seed(42)
    X, Y = np.meshgrid(range(Nx), range(Ny))

    F = F_last_time_step
    if iteration==0:
        F = np.ones((Ny,Nx,NL)) #* rho0 / NL
        F += 0.02*np.random.randn(Ny,Nx,NL)
        F[:,:,3] += initial_ux # 2 * (1+0.2*np.cos(2*np.pi*X/Nx*4))
    
    rho = np.sum(F,2)
    for i in idxs:
        F[:,:,i] *= rho0 / rho

    
    
    
    # Prep figure
    #fig = plt.figure(figsize=(6,2), dpi=80)
    
    # Simulation Main Loop
    for it in range(Nt):
        # Cylinder boundary
        cylinders_list = create_cylinders(circles) # a list of cylinders, each has its own mask
        cylinders = create_cylinder_as_one(circles) # multiple cylinders in one mask
        boundary_masks = [get_boundary_values(cylinder) for cylinder in cylinders_list]
        
        # Set transmissive boundaries for walls
        F[:,-1,[6,7,8]] = F[:,-2,[6,7,8]]
        F[:,0,[2,3,4]] = F[:,1,[2,3,4]]


        # Drift
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
            F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)
        
        
        # Set reflective boundaries
        bndryF = F[cylinders,:]
        bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]
    
        
        # Calculate fluid variables
        rho = np.sum(F,2)
        ux  = np.sum(F*cxs,2) / rho
        uy  = np.sum(F*cys,2) / rho


        # Apply Collision
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:,:,i] = rho * w * ( 1 + 3*(cx*ux+cy*uy)  + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2 )
        
        F += -(1.0/tau) * (F - Feq)
        
        # Apply boundary 
        F[cylinders,:] = bndryF

        # Compute drag 
        drags, drag_mask_dict = compute_drag(ux, uy, boundary_masks, rho=rho0)

        
        # print(ux.max())
        # plot in real time - color 1/2 particles blue, other half red
        if (plotRealTime and (it % 10) == 0) or (it == Nt-1):
            plt.cla()
            ux[cylinders] = 0
            uy[cylinders] = 0
            vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
            vorticity[cylinders] = np.nan
            vorticity = np.ma.array(vorticity, mask=cylinders)
            plt.imshow(vorticity, cmap='bwr')
            plt.imshow(np.sqrt(ux**2+uy**2), cmap='bwr')
            plt.imshow(~cylinders, cmap='gray', alpha=0.3)
            plt.clim(-.1, .1)                                                                # this iteration is the number of agents*MAX_STEPS (in main.py), it is the number of rewards per game
            plt.title("drags="+str({key: round(value, 2) for key, value in drags.items()})+" Iter="+str(iteration) + " max u=({:.2f}".format(ux.max())+ ",  {:.2f})".format(uy.max())+ " Re={:.1f}".format(Re))
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)   
            ax.set_aspect('equal')  
            #plt.pause(0.001)

            # Save figure
            print("saving time instances in img_folder:"+str(img_folder))
            filename = os.path.join(img_folder, f"latticeboltzmann{iteration:03d}.png")
            plt.savefig(filename,dpi=240)
            #plt.show()
            
        
        if read_stopit_file()==1:
            break


        
    return drags, drag_mask_dict, F, ux, uy, cylinders


