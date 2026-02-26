import torch
import time
import math
import numpy as np
import matplotlib.pyplot as plt

print("Wedge Flow Problem - Non-Conservative 2D Euler Equations")
print("=" * 70)

torch.manual_seed(5)
np.random.seed(5)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Physical constants
gamma = 1.4
r_inf = 1.0
p_inf = 1.0
M_inf = 2.0
v_inf = 0.0
u_inf = math.sqrt(gamma*p_inf/r_inf)*M_inf

# Geometry
x_min = 0.0
x_max = 2
y_min = 0.0
y_max = 2.0
w_angle = math.radians(10)
w_start = 1

# Hyperparameters
num_ib = 750
num_r = 8000  # Reduced for faster training

print(f"Problem setup:")
print(f"   - Domain: x ∈ [{x_min}, {x_max}], y ∈ [{y_min}, {y_max}]")
print(f"   - Wedge angle: {math.degrees(w_angle):.1f}°")
print(f"   - Mach number: {M_inf}")
print(f"   - Freestream velocity: ({u_inf:.2f}, {v_inf:.2f})")

# 2D Non-conservative Euler PDE definitions
def pde_non_conservative(state, coords):
    """Non-conservative 2D Euler equations"""
    # Coordinates
    x = coords[0]
    y = coords[1]

    # State variables
    r = state[:, [0]]    # density
    p = state[:, [1]]    # pressure
    u = state[:, [2]]    # x-velocity
    v = state[:, [3]]    # y-velocity
    
    # Compute gradients
    r_x = torch.autograd.grad(r, x, grad_outputs=torch.ones_like(r), create_graph=True)[0]
    r_y = torch.autograd.grad(r, y, grad_outputs=torch.ones_like(r), create_graph=True)[0]
    
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    
    # Non-conservative form of 2D Euler equations
    # Continuity: ∂ρ/∂t + u∂ρ/∂x + v∂ρ/∂y + ρ(∂u/∂x + ∂v/∂y) = 0
    # For steady flow, ∂ρ/∂t = 0
    r1 = u*r_x + v*r_y + r*(u_x + v_y)
    # X-momentum: ρ(u∂u/∂x + v∂u/∂y) + ∂p/∂x = 0    
    r2 = r*(u*u_x + v*u_y) + p_x    
    # Y-momentum: ρ(u∂v/∂x + v∂v/∂y) + ∂p/∂y = 0    
    r3 = r*(u*v_x + v*v_y) + p_y    
    # Energy equation (simplified for steady flow)    
    # Assume isentropic flow: p/ρ^γ = constant    
    # This gives us: γp∇·v + v·∇p = 0    
    r4 = gamma*p*(u_x + v_y) + u*p_x + v*p_y
    
    # Add artificial viscosity for numerical stability
    d = 0.12*(torch.abs(u_x) - u_x) + 1 + 0.12*(torch.abs(v_y) - v_y) + 1e-8  # Small epsilon to prevent division by zero
    
    return r1/d, r2/d, r3/d, r4/d

# Generator Network
class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        t1 = 15
        
        torch.manual_seed(0)
        self.map = torch.nn.Sequential(
            torch.nn.Linear(2, t1), torch.nn.Tanh(),
            torch.nn.Linear(t1, t1), torch.nn.Tanh(),
            torch.nn.Linear(t1, t1), torch.nn.Tanh(),
            torch.nn.Linear(t1, t1), torch.nn.Tanh(),
            torch.nn.Linear(t1, 5)
        )
    
    def forward(self, a: tuple) -> torch.Tensor:
        # Force positive density and pressure
        res = self.map(torch.cat(a, 1))
        res[:,0] = torch.exp(res[:,0])  # positive density
        res[:,1] = torch.exp(res[:,1])  # positive pressure
        return res

# Loss Functions
def loss_pde(coords):
    """PDE loss function"""
    global mse_pde, res_r, res_p, res_u, res_v, mse_v
    
    for x in coords: 
        x.requires_grad = True
    
    state = generator(coords)
    r1, r2, r3, r4 = pde_non_conservative(state, coords)
    
    f = (r1**2).mean() + (r2**2).mean() + (r3**2).mean() + (r4**2).mean()
    
    mse_pde = f.item()
    res_r = (r1**2).mean().item()
    res_p = (r2**2).mean().item()
    res_u = (r3**2).mean().item()
    res_v = (r4**2).mean().item()
    
    f_v = (state[:,[4]]**2).mean()
    mse_v = f_v.item()
    
    return f, f_v

def loss_slip(coords, thetas):
    """Slip boundary condition loss"""
    for x in coords: 
        x.requires_grad = True
    
    state = generator(coords)
    x = coords[0]
    y = coords[1]
    
    r = state[:,[0]]
    p = state[:,[1]]
    u = state[:,[2]]
    v = state[:,[3]]
    
    cos = torch.cos(thetas)
    sin = torch.sin(thetas)
    
    p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]
    
    p_theta = p_x * cos + p_y * sin
    u_theta = u * cos + v * sin
    
    loss = (u_theta**2).mean()
    
    return loss

def loss_inlet(x_bc, u_bc):
    """Inlet boundary condition loss"""
    u_pred = generator(x_bc)
    
    loss = ((u_pred[:,0] - u_bc[0])**2).mean() + \
           ((u_pred[:,1] - u_bc[1])**2).mean() + \
           ((u_pred[:,2] - u_bc[2])**2).mean() + \
           ((u_pred[:,3] - u_bc[3])**2).mean()
    
    return loss

def loss_sym(coords, thetas):
    """Symmetry boundary condition loss"""
    for x in coords: 
        x.requires_grad = True
    
    state = generator(coords)
    
    v = state[:, [3]]
    
    fb = (v**2).mean()
    
    return fb

def evaluate():
    """Evaluation function for optimizer"""
    global mse_bc
    
    optimizer.zero_grad()
    
    loss_r, loss_v = loss_pde(x_r)
    loss_b = loss_sym(x_b, t_b)
    loss_l = loss_inlet(x_l, u_l)
    loss_c = loss_slip(x_w, t_w)
    
    loss_bc = loss_c + loss_l + loss_b
    
    mse_bc = loss_bc.item()
    
    loss = loss_r + 1 * loss_bc + 1e-2 * loss_v
    loss.backward()
    
    return loss

if __name__ == '__main__':
    # Set up hardware
    torch.set_default_dtype(torch.float32)
    
    # Boundary conditions setup
    # Inlet boundary (left)
    x_l = torch.full((num_ib, 1), x_min, device=device)
    y_l = torch.linspace(y_min, y_max, num_ib, device=device).reshape(-1,1)
    u_l = (r_inf, p_inf, u_inf, v_inf)
    
    # Top boundary
    x_t = torch.linspace(x_min, x_max, num_ib, device=device).reshape(-1,1)
    y_t = torch.full((num_ib, 1), y_max, device=device)
    t_t = torch.full((num_ib,1), -torch.pi/2, device=device)
    
    # Bottom boundary (symmetry)
    x_b = torch.linspace(x_min, w_start, num_ib+1, device=device)[:-1].reshape(-1,1)
    y_b = torch.full((num_ib, 1), y_min, device=device)
    t_b = torch.full((num_ib,1), torch.pi/2, device=device)
    
    # Wedge boundary
    x_w = torch.linspace(w_start, x_max, num_ib, device=device).reshape(-1,1)
    geom = lambda x: (x-w_start)*math.tan(w_angle)
    y_w = y_min + geom(x_w)
    t_w = torch.full((num_ib,1), torch.pi/2 + w_angle, device=device)
    
    # Convert to tuples
    x_l = (x_l, y_l)
    x_t = (x_t, y_t)
    x_b = (x_b, y_b)
    x_w = (x_w, y_w)
    
    # Domain points - try to read mesh file, fallback to grid if not available
    try:
        with open('fluent_mesh.dat', 'r') as file:
            lines = file.readlines()
        
        # Skip the first line and split each line into a list of elements
        data = [line.strip().split() for line in lines[1:]]
        data_in = np.array(data, dtype=float)
        
        x_in = data_in[:,1]
        y_in = data_in[:,2]
        
        print(f"Loaded mesh with {len(x_in)} points")
    except FileNotFoundError:
        print("Mesh file not found, using regular grid")
        # Generate regular grid
        n1 = 101
        xp, yp = np.meshgrid(np.linspace(x_min, x_max, n1), np.linspace(y_min, y_max, n1))
        xf = xp.flatten()
        yf = yp.flatten()
        xin = np.vstack([xf, yf]).T
        
        # Filter points above wedge
        t1 = y_min + geom(xin[:,0])
        t1[np.isnan(t1)] = 0
        mapping = xin[:,1] > t1
        mapping1 = xin[:,0] < w_start
        mapping = mapping | mapping1
        xin1 = xin[mapping,:]
        
        x_in = xin1[:,0]
        y_in = xin1[:,1]
    
    x_test1 = torch.tensor(x_in, device=device, dtype=torch.float32)
    y_test1 = torch.tensor(y_in, device=device, dtype=torch.float32)
    x_test1 = torch.reshape(x_test1, (-1,1))
    y_test1 = torch.reshape(y_test1, (-1,1))
    x_test = (x_test1, y_test1)
    x_r = x_test
    
    print(f"Using {len(x_test1)} domain points for training")
    
    # Set up models and optimizer
    generator = Generator().to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)  # Use Adam for better stability
    
    # Initialize tracking variables
    mse_pde = 0.0
    mse_bc = 0.0
    mse_v = 0.0
    res_r = 0.0
    res_p = 0.0
    res_u = 0.0
    res_v = 0.0
    
    # Training loop
    epoch = 0
    start_time = time.time()
    
    print("Starting training...")
    print("Epoch | Time  | MSE_r   | MSE_v   | MSE_bc  ")
    print("-" * 45)
    
    for i in range(1001):
        epoch_start_time = time.time()
        
        # Optimize loss function using Adam
        loss = evaluate()
        optimizer.step()
        
        # Print status
        if epoch % 50 == 0 or epoch < 10:
            print(f'{epoch:4d}  | {time.time() - epoch_start_time:.2f}  | {mse_pde:.2e} | {mse_v:.2e} | {mse_bc:.2e}')
        
        epoch += 1
        
        # Early stopping criterion
        if mse_pde + mse_bc < 1e-5:
            print(f"Converged at epoch {epoch}!")
            break
    
    # Final evaluation
    print(f"\n🎉 Training completed in {time.time() - start_time:.2f} seconds")
    print(f"Final losses: MSE_PDE={mse_pde:.2e}, MSE_BC={mse_bc:.2e}")
    
    # Generate predictions
    with torch.no_grad():
        u_pred = generator(x_test)
        
    print("\nFinal solution statistics:")
    print(f"   - Density range: [{u_pred[:,0].min():.3f}, {u_pred[:,0].max():.3f}]")
    print(f"   - Pressure range: [{u_pred[:,1].min():.3f}, {u_pred[:,1].max():.3f}]")
    print(f"   - X-velocity range: [{u_pred[:,2].min():.3f}, {u_pred[:,2].max():.3f}]")
    print(f"   - Y-velocity range: [{u_pred[:,3].min():.3f}, {u_pred[:,3].max():.3f}]")
    
    # Generate prediction grid for visualization
    print("\nGenerating predictions for visualization...")
    nx_viz, ny_viz = 50, 50
    x_viz = torch.linspace(x_min, x_max, nx_viz)
    y_viz = torch.linspace(y_min, y_max, ny_viz)
    X_viz, Y_viz = torch.meshgrid(x_viz, y_viz, indexing='ij')
    
    # Flatten and remove points inside wedge
    x_flat = X_viz.flatten().unsqueeze(1)
    y_flat = Y_viz.flatten().unsqueeze(1)
    
    # Create mask for points outside wedge
    mask_viz = ~((x_flat >= w_start) & (y_flat <= (x_flat - w_start) * math.tan(w_angle)))
    x_pred_viz = x_flat[mask_viz.squeeze()].to(device)
    y_pred_viz = y_flat[mask_viz.squeeze()].to(device)
    
    # Get predictions
    with torch.no_grad():
        state_pred = generator((x_pred_viz, y_pred_viz))
        rho_pred = state_pred[:, 0].cpu()
        p_pred = state_pred[:, 1].cpu()
        u_pred_viz = state_pred[:, 2].cpu()
        v_pred_viz = state_pred[:, 3].cpu()
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Create scatter plots since we have irregular grids
    x_np = x_pred_viz.cpu().numpy().flatten()
    y_np = y_pred_viz.cpu().numpy().flatten()
    
    # Density
    sc1 = axes[0,0].scatter(x_np, y_np, c=rho_pred.numpy(), cmap='viridis', s=1)
    axes[0,0].set_title('Density (ρ)')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    plt.colorbar(sc1, ax=axes[0,0])
    
    # Pressure
    sc2 = axes[0,1].scatter(x_np, y_np, c=p_pred.numpy(), cmap='plasma', s=1)
    axes[0,1].set_title('Pressure (p)')
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    plt.colorbar(sc2, ax=axes[0,1])
    
    # X-velocity
    sc3 = axes[1,0].scatter(x_np, y_np, c=u_pred_viz.numpy(), cmap='coolwarm', s=1)
    axes[1,0].set_title('X-Velocity (u)')
    axes[1,0].set_xlabel('x')  
    axes[1,0].set_ylabel('y')
    plt.colorbar(sc3, ax=axes[1,0])
    
    # Y-velocity
    sc4 = axes[1,1].scatter(x_np, y_np, c=v_pred_viz.numpy(), cmap='coolwarm', s=1)
    axes[1,1].set_title('Y-Velocity (v)')
    axes[1,1].set_xlabel('x')
    axes[1,1].set_ylabel('y')
    plt.colorbar(sc4, ax=axes[1,1])
    
    # Add wedge outline to all plots
    for ax in axes.flat:
        # Wedge surface
        x_wedge_line = np.linspace(w_start, x_max, 100)
        y_wedge_line = (x_wedge_line - w_start) * math.tan(w_angle)
        ax.plot(x_wedge_line, y_wedge_line, 'k-', linewidth=2, label='Wedge')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
    
    plt.suptitle(f'Wedge Flow - Non-Conservative 2D Euler (PINN Solution)\\nMach {M_inf}, Wedge Angle {math.degrees(w_angle):.1f}°', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("Wedge Non-Conservative problem completed!")