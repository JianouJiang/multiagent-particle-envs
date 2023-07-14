import numpy as np
import matplotlib.pyplot as plt

def plot_airfoil(t, chord_length, leading_edge_radius_factor):
    x = np.linspace(0, 1, 100)
    y_t = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    
    # Modify the leading edge shape
    y = y_t * (1 - leading_edge_radius_factor * x**2)
    
    # Scale x for the chord length
    x *= chord_length
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='Upper Surface')
    plt.plot(x, -y, label='Lower Surface')
    plt.xlabel('Chord Position')
    plt.ylabel('Distance from Chord Line')
    plt.title(f'NACA 00{int(t*100)} Airfoil')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()

# Plot a NACA 0012 airfoil with a chord length of 2 and a leading edge radius factor of 0.1
plot_airfoil(0.22, 1, 1)
