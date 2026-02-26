import torch
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

print("Wedge Flow Problem - Conservative 2D Euler Equations")
print("=" * 70)

torch.manual_seed(5)
np.random.seed(5)

# Device setup
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
num_r = 10000

print(f"Problem setup:")
print(f"   - Domain: x ∈ [{x_min}, {x_max}], y ∈ [{y_min}, {y_max}]")
print(f"   - Wedge angle: {math.degrees(w_angle):.1f}°")
print(f"   - Mach number: {M_inf}")
print(f"   - Freestream velocity: ({u_inf:.2f}, {v_inf:.2f})")

# 2D Euler PDE definitions
def pde(state, coords):
    """Conservative 2D Euler equations with artificial viscosity"""
    # Coordinates
    x = coords[0]
    y = coords[1]

    # State variables
    r = state[:, [0]]    # density
    p = state[:, [1]]    # pressure
    u = state[:, [2]]    # x-velocity
    v = state[:, [3]]    # y-velocity
    viscosity = state[:, [4]]**2  # artificial viscosity
    
    # Total energy
    rE = p/(gamma - 1) + 0.5*r*(u**2 + v**2)

    # Conservative variables
    u1 = r
    u2 = r*u
    u3 = r*v
    u4 = rE

    # Flux functions in x-direction
    f1 = r*u
    f2 = r*u*u + p
    f3 = r*u*v
    f4 = (rE + p)*u
    
    # Flux functions in y-direction
    g1 = r*v
    g2 = r*v*u
    g3 = r*v*v + p
    g4 = (rE + p)*v
    
    # Compute gradients
    f1_x = torch.autograd.grad(f1, x, grad_outputs=torch.ones_like(f1), create_graph=True)[0]
    f2_x = torch.autograd.grad(f2, x, grad_outputs=torch.ones_like(f2), create_graph=True)[0]
    f3_x = torch.autograd.grad(f3, x, grad_outputs=torch.ones_like(f3), create_graph=True)[0]
    f4_x = torch.autograd.grad(f4, x, grad_outputs=torch.ones_like(f4), create_graph=True)[0]

    g1_y = torch.autograd.grad(g1, y, grad_outputs=torch.ones_like(g1), create_graph=True)[0]
    g2_y = torch.autograd.grad(g2, y, grad_outputs=torch.ones_like(g2), create_graph=True)[0]
    g3_y = torch.autograd.grad(g3, y, grad_outputs=torch.ones_like(g3), create_graph=True)[0]
    g4_y = torch.autograd.grad(g4, y, grad_outputs=torch.ones_like(g4), create_graph=True)[0]

    # Second derivatives for artificial viscosity
    u1_x = torch.autograd.grad(u1, x, grad_outputs=torch.ones_like(u1), create_graph=True)[0]
    u2_x = torch.autograd.grad(u2, x, grad_outputs=torch.ones_like(u2), create_graph=True)[0]
    u3_x = torch.autograd.grad(u3, x, grad_outputs=torch.ones_like(u3), create_graph=True)[0]
    u4_x = torch.autograd.grad(u4, x, grad_outputs=torch.ones_like(u4), create_graph=True)[0]

    u1_xx = torch.autograd.grad(u1_x, x, grad_outputs=torch.ones_like(u1_x), create_graph=True)[0]
    u2_xx = torch.autograd.grad(u2_x, x, grad_outputs=torch.ones_like(u2_x), create_graph=True)[0]
    u3_xx = torch.autograd.grad(u3_x, x, grad_outputs=torch.ones_like(u3_x), create_graph=True)[0]
    u4_xx = torch.autograd.grad(u4_x, x, grad_outputs=torch.ones_like(u4_x), create_graph=True)[0]

    u1_y = torch.autograd.grad(u1, y, grad_outputs=torch.ones_like(u1), create_graph=True)[0]
    u2_y = torch.autograd.grad(u2, y, grad_outputs=torch.ones_like(u2), create_graph=True)[0]
    u3_y = torch.autograd.grad(u3, y, grad_outputs=torch.ones_like(u3), create_graph=True)[0]
    u4_y = torch.autograd.grad(u4, y, grad_outputs=torch.ones_like(u4), create_graph=True)[0]

    u1_yy = torch.autograd.grad(u1_y, y, grad_outputs=torch.ones_like(u1_y), create_graph=True)[0]
    u2_yy = torch.autograd.grad(u2_y, y, grad_outputs=torch.ones_like(u2_y), create_graph=True)[0]
    u3_yy = torch.autograd.grad(u3_y, y, grad_outputs=torch.ones_like(u3_y), create_graph=True)[0]
    u4_yy = torch.autograd.grad(u4_y, y, grad_outputs=torch.ones_like(u4_y), create_graph=True)[0]

    # PDE residuals with artificial viscosity
    r1 = f1_x + g1_y - viscosity * (u1_xx + u1_yy)
    r2 = f2_x + g2_y - viscosity * (u2_xx + u2_yy)
    r3 = f3_x + g3_y - viscosity * (u3_xx + u3_yy)
    r4 = f4_x + g4_y - viscosity * (u4_xx + u4_yy)

    return torch.cat([r1, r2, r3, r4], dim=1)

# Neural Network Definition
class WedgePN(torch.nn.Module):
    def __init__(self, layers):
        super(WedgePN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(torch.nn.Linear(layers[i], layers[i + 1]))
        self.activation = torch.tanh

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

# Generate collocation points
def generate_collocation_points():
    # Interior points
    x_int = torch.rand(num_r, 1) * (x_max - x_min) + x_min
    y_int = torch.rand(num_r, 1) * (y_max - y_min) + y_min
    
    # Remove points inside the wedge (simple triangular mask)
    mask = ~((x_int >= w_start) & (y_int <= (x_int - w_start) * math.tan(w_angle)))
    x_int = x_int[mask.squeeze()]
    y_int = y_int[mask.squeeze()]
    
    x_int.requires_grad_(True)
    y_int.requires_grad_(True)
    
    return x_int, y_int

# Generate boundary points
def generate_boundary_points():
    # Inlet boundary
    x_inlet = torch.zeros(num_ib // 4, 1)
    y_inlet = torch.rand(num_ib // 4, 1) * y_max
    
    # Outlet boundary
    x_outlet = torch.full((num_ib // 4, 1), x_max)
    y_outlet = torch.rand(num_ib // 4, 1) * y_max
    
    # Upper boundary
    x_upper = torch.rand(num_ib // 4, 1) * x_max
    y_upper = torch.full((num_ib // 4, 1), y_max)
    
    # Wedge surface
    x_wedge = torch.rand(num_ib // 4, 1) * (x_max - w_start) + w_start
    y_wedge = (x_wedge - w_start) * math.tan(w_angle)
    
    x_bc = torch.cat([x_inlet, x_outlet, x_upper, x_wedge])
    y_bc = torch.cat([y_inlet, y_outlet, y_upper, y_wedge])
    
    return x_bc, y_bc

# Initialize model
print("Building neural network model...")
model = WedgePN([2, 50, 50, 50, 50, 5]).to(device)  # 5 outputs: rho, p, u, v, nu

# Generate training points
print("Generating collocation and boundary points...")
x_int, y_int = generate_collocation_points()
x_bc, y_bc = generate_boundary_points()

print(f"   - Interior points: {len(x_int)}")
print(f"   - Boundary points: {len(x_bc)}")

# Move to device
x_int, y_int = x_int.to(device), y_int.to(device)
x_bc, y_bc = x_bc.to(device), y_bc.to(device)

# Training function
def train_wedge_pinn():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Starting training...")
    
    for epoch in range(5000):  # Reduced epochs for demonstration
        optimizer.zero_grad()
        
        # Interior loss (PDE)
        coords_int = [x_int, y_int]
        state_int = model(torch.cat([x_int, y_int], dim=1))
        pde_residual = pde(state_int, coords_int)
        loss_pde = torch.mean(pde_residual**2)
        
        # Boundary conditions (simplified)
        state_bc = model(torch.cat([x_bc, y_bc], dim=1))
        
        # Inlet: freestream conditions
        inlet_mask = (x_bc == 0).squeeze()
        if inlet_mask.any():
            rho_inlet = state_bc[inlet_mask, 0]
            u_inlet = state_bc[inlet_mask, 2]
            v_inlet = state_bc[inlet_mask, 3]
            p_inlet = state_bc[inlet_mask, 1]
            
            loss_inlet = torch.mean((rho_inlet - r_inf)**2 + (u_inlet - u_inf)**2 + 
                                   (v_inlet - v_inf)**2 + (p_inlet - p_inf)**2)
        else:
            loss_inlet = torch.tensor(0.0, device=device)
        
        # Wedge surface: no-slip condition (u = v = 0)
        wedge_mask = (y_bc <= (x_bc - w_start) * math.tan(w_angle) + 1e-3).squeeze()
        if wedge_mask.any():
            u_wedge = state_bc[wedge_mask, 2]
            v_wedge = state_bc[wedge_mask, 3]
            loss_wedge = torch.mean(u_wedge**2 + v_wedge**2)
        else:
            loss_wedge = torch.tensor(0.0, device=device)
        
        # Total loss
        loss = loss_pde + 10*loss_inlet + 10*loss_wedge
        
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Total={loss.item():.2e}, PDE={loss_pde.item():.2e}, "
                  f"Inlet={loss_inlet.item():.2e}, Wedge={loss_wedge.item():.2e}")

# Train the model
train_wedge_pinn()

# Generate prediction grid for visualization
print("Generating predictions for visualization...")
nx_viz, ny_viz = 50, 50
x_viz = torch.linspace(x_min, x_max, nx_viz)
y_viz = torch.linspace(y_min, y_max, ny_viz)
X_viz, Y_viz = torch.meshgrid(x_viz, y_viz, indexing='ij')

# Flatten and remove points inside wedge
x_flat = X_viz.flatten().unsqueeze(1)
y_flat = Y_viz.flatten().unsqueeze(1)

# Create mask for points outside wedge
mask_viz = ~((x_flat >= w_start) & (y_flat <= (x_flat - w_start) * math.tan(w_angle)))
x_pred = x_flat[mask_viz.squeeze()].to(device)
y_pred = y_flat[mask_viz.squeeze()].to(device)

# Get predictions
with torch.no_grad():
    state_pred = model(torch.cat([x_pred, y_pred], dim=1))
    rho_pred = state_pred[:, 0].cpu()
    p_pred = state_pred[:, 1].cpu()
    u_pred = state_pred[:, 2].cpu()
    v_pred = state_pred[:, 3].cpu()

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Create scatter plots since we have irregular grids
x_np = x_pred.cpu().numpy().flatten()
y_np = y_pred.cpu().numpy().flatten()

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
sc3 = axes[1,0].scatter(x_np, y_np, c=u_pred.numpy(), cmap='coolwarm', s=1)
axes[1,0].set_title('X-Velocity (u)')
axes[1,0].set_xlabel('x')  
axes[1,0].set_ylabel('y')
plt.colorbar(sc3, ax=axes[1,0])

# Y-velocity
sc4 = axes[1,1].scatter(x_np, y_np, c=v_pred.numpy(), cmap='coolwarm', s=1)
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

plt.suptitle(f'Wedge Flow - Conservative 2D Euler (PINN Solution)\\nMach {M_inf}, Wedge Angle {math.degrees(w_angle):.1f}°', fontsize=14)
plt.tight_layout()
plt.show()

print("Wedge Conservative problem completed!")