import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import plotly.graph_objects as go


model_type = 'discrete'  # 'continuous' or 'discrete'
random_theta = False       # Only applies if model_type == 'discrete'

# Spatial Config
num_sites = 11          
num_time_steps = 100    # Plot resolution / iterations
center = num_sites // 2

# 1.CTQW
if model_type == 'continuous':
    max_time = 15.0     # Total time duration of the physical walk
    
    # Construct Hamiltonian Adjacency Matrix
    H = np.zeros((num_sites, num_sites))
    for i in range(num_sites):
        if i > 0:
            H[i, i-1] = 1.0
        if i < num_sites - 1:
            H[i, i+1] = 1.0
            
    # Initialize 1D Wavefunction State Vector
    psi_0 = np.zeros(num_sites, dtype=complex)
    psi_0[center] = 1.0
    
    # Sample Continuous Time Timeline
    time_entries = np.linspace(0, max_time, num_time_steps + 1)
    prob_grid = []
    
    for t in time_entries:
        # Time-evolution operator: U(t) = exp(-i * H * t)
        U = la.expm(-1j * H * t)
        psi_t = U @ psi_0
        prob_grid.append(np.abs(psi_t)**2)
        
    prob_grid = np.array(prob_grid)
    site_probs = prob_grid[-1]
    plot_y_max = max_time
    y_label = "Continuous Time (t)"
    title_text = "Continuous-Time Quantum Walk (CTQW) Spacetime Evolution"

# 2. DTQW
else:
    L_idx, R_idx = 0, 1
    psi = np.zeros((num_sites, 2), dtype=complex)

    psi[center, L_idx] = 1.0 / np.sqrt(2)
    psi[center, R_idx] = 1j / np.sqrt(2) 
    
    prob_grid = []
    prob_grid.append(np.sum(np.abs(psi)**2, axis=1))
    
    theta = np.pi / 4

    c_matrix = np.array([
        [np.cos(theta),  np.sin(theta)],
        [np.sin(theta), -np.cos(theta)]
    ], dtype=complex)
    
    # 2. DTQW Evolution
    for t in range(num_time_steps):
        new_psi_after_coin = np.zeros_like(psi)
        
        # 1. Site-dependent random coin
        for x in range(num_sites):
            # Generate a unique theta for this specific site at this time step
            local_theta = np.random.uniform(0, 2 * np.pi) if random_theta else theta
            
            c_matrix = np.array([
                [np.cos(local_theta),  np.sin(local_theta)],
                [np.sin(local_theta), -np.cos(local_theta)]
            ], dtype=complex)
            
            new_psi_after_coin[x] = c_matrix @ psi[x]
            
        # 2. Shift Operation
        new_psi_after_shift = np.zeros_like(psi)
        for x in range(num_sites):
            #Bounce logic
            if x > 0:
                new_psi_after_shift[x - 1, L_idx] += new_psi_after_coin[x, L_idx]
            else:
                new_psi_after_shift[0, R_idx] += new_psi_after_coin[x, L_idx]
                
            if x < num_sites - 1:
                new_psi_after_shift[x + 1, R_idx] += new_psi_after_coin[x, R_idx]
            else:
                new_psi_after_shift[num_sites - 1, L_idx] += new_psi_after_coin[x, R_idx]
        
        # Update and Normalize
        psi = new_psi_after_shift
        norm = np.linalg.norm(psi)
        if norm > 0:
            psi /= norm
            
        prob_grid.append(np.sum(np.abs(psi)**2, axis=1))
        
    prob_grid = np.array(prob_grid)
    site_probs = prob_grid[-1]
    plot_y_max = num_time_steps
    y_label = "Discrete Time Step (t)"
    title_text = "Discrete-Time Random Environment Spacetime Evolution"



# Plotting Configuration
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor='white', 
                               gridspec_kw={'width_ratios': [1, 1.2]})

# --- Subplot 1: 1D Chain Snapshot at Final Time (Left Panel) ---
ax1.plot([0, num_sites - 1], [0, 0], color='#dee2e6', linewidth=2, zorder=1)

for i in range(num_sites):
    prob = site_probs[i]
    if prob > 0.02:
        color = '#ff7f0e'  
        label_text = f"{prob:.2f}"
        text_color = "white"
    else:
        color = '#f8f9fa'
        label_text = "0"
        text_color = "#6c757d"
    ax1.scatter(i, 0, s=600, color=color, edgecolors='#6c757d', linewidth=1.5, zorder=2)
    ax1.text(i, 0, label_text, color=text_color, ha='center', va='center',
             fontsize=8, fontweight='bold', zorder=3)
    ax1.text(i, -0.4, f"x[{i+1}]", color='#495057', ha='center', va='center', fontsize=9)

ax1.set_aspect('equal')
ax1.set_xlim(-0.7, num_sites - 0.3)
ax1.set_ylim(-0.8, 0.8)
ax1.axis('off')
ax1.set_title("1D Chain Snapshot at Final Interval", fontsize=12, pad=10)


# --- Subplot 2: Spacetime Heatmap (Right Panel) ---
im = ax2.imshow(
    prob_grid,
    aspect='auto',
    origin='lower',
    interpolation='bilinear',   
    cmap='inferno',             
    extent=[0.5, num_sites + 0.5, 0, plot_y_max]
)

plt.colorbar(im, ax=ax2, label='Probability Density')
ax2.set_xlim(0.5, num_sites + 0.5)
ax2.set_ylim(0, plot_y_max)
ax2.set_xticks(range(2, num_sites + 1, 2)) 

ax2.set_xlabel("Site Index (x)", fontsize=11)
ax2.set_ylabel(y_label, fontsize=11)
ax2.set_title(title_text, fontsize=12, pad=12)

plt.tight_layout()
plt.show()

np.set_printoptions(linewidth=200, formatter={'float_kind': '{:6.3f}'.format})
print("\n" + "="*95)
print("FINAL CONSOLIDATED 2D DATA HISTORY GRID (prob_grid)")
print("="*95)
for idx, row in enumerate(prob_grid):
    if idx % 10 == 0 or idx == len(prob_grid) - 1:
        print(f"Row [{idx:3d}] {row}")

fig = go.Figure(data=[go.Surface(z=prob_grid.T)])

fig.update_layout(
    title='3D Quantum Walk Probability Landscape',
    scene=dict(
        xaxis_title='Time Step (t)',
        yaxis_title='Site Index (x)',
        zaxis_title='Probability Density'
    ),
    autosize=True
)

fig.show()
