import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt

print("Sod Shock Tube Problem - Conservative Euler Equations")
print("=" * 70)

# Seeds
torch.manual_seed(12)
np.random.seed(12)

# Domain parameters
x_min = 0
x_max = 1
t_max = 0.10
nx = 101
nt = 101
ct = 1
er_c1 = 1e-3

# Device setup
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Calculate gradients using torch.autograd.grad
def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

# Convert torch tensor into np.array
def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or ' +
                        'np.ndarray, but got {}'.format(type(input)))

# Initial conditions for Sod shock tube
def IC(x):
    N = len(x)
    rho_init = np.zeros((x.shape[0]))
    u_init = np.zeros((x.shape[0]))
    p_init = np.zeros((x.shape[0]))

    # rho, p - initial condition (classic Sod shock tube)
    for i in range(N):
        if (x[i] <= 0.5):
            rho_init[i] = 1.0
            p_init[i] = 1.0
        else:
            rho_init[i] = 0.125
            p_init[i] = 0.1

    return rho_init, u_init, p_init

# Generate Neural Network adaptive tanh(ax) activation function
class ParamTanh(nn.Module):
    def __init__(self, alpha):
        super(ParamTanh, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=True))

    def forward(self, x):
        return torch.tanh(self.alpha * x)

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential()
        tn = 15
        self.net.add_module('Linear_layer_1', nn.Linear(2, tn))
        self.net.add_module('Tanh_layer_1', nn.Tanh())

        for num in range(2, 5):
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(tn, tn))
            self.net.add_module('Tanh_layer_%d' % (num), ParamTanh(alpha=0.9))
        self.net.add_module('Linear_layer_final', nn.Linear(tn, 4))

    # Forward Feed
    def forward(self, x):
        return self.net(x)

    # Loss function for PDE (Conservative Euler equations)
    def loss_pde(self, x):
        try:
            y = self.net(x)
            rho, p, u, nui = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
            
            # Add small epsilon to prevent division by zero
            rho = torch.clamp(rho, min=1e-6)
            p = torch.clamp(p, min=1e-6)
            
            U2 = rho*u
            U3 = 0.5*rho*u**2 + p/0.4
            
            # Flux terms
            F2 = rho*u**2 + p
            F3 = u*(U3 + p)
            
            gamma = 1.4

            # Gradients and partial derivatives
            drho_g = gradients(rho, x)[0]
            rho_t, rho_x = drho_g[:, :1], drho_g[:, 1:]

            drho_gg = gradients(rho_x, x)[0]
            rho_tx, rho_xx = drho_gg[:, 0:1], drho_gg[:, 1:2]  # Keep dimensions consistent

            du_g = gradients(u, x)[0]
            u_t, u_x = du_g[:, :1], du_g[:, 1:]

            dU2_g = gradients(U2, x)[0]
            U2_t, U2_x = dU2_g[:, :1], dU2_g[:, 1:]
            d2U2_g = gradients(U2_x, x)[0]
            U2_tx, U2_xx = d2U2_g[:, :1], d2U2_g[:, 1:]
            
            dU3_g = gradients(U3, x)[0]
            U3_t, U3_x = dU3_g[:, :1], dU3_g[:, 1:]
            d2U3_g = gradients(U3_x, x)[0]
            U3_tx, U3_xx = d2U3_g[:, :1], d2U3_g[:, 1:]
            
            dF2_g = gradients(F2, x)[0]
            F2_t, F2_x = dF2_g[:, :1], dF2_g[:, 1:]
            
            dF3_g = gradients(F3, x)[0]
            F3_t, F3_x = dF3_g[:, :1], dF3_g[:, 1:]

            # Artificial viscosity (use torch.abs instead of abs, add small epsilon for stability)
            d = 0.12*(torch.abs(u_x)-u_x) + 1 + 1e-8
            nu = nui**2

            # Conservative form of Euler equations with artificial viscosity
            # Add torch.clamp to prevent extreme values
            f = (torch.clamp(((rho_t + U2_x - nu*rho_xx)/d)**2, max=1e6)).mean() + \
                (torch.clamp(((U2_t + F2_x - nu*U2_xx)/d)**2, max=1e6)).mean() + \
                (torch.clamp(((U3_t + F3_x - nu*U3_xx)/d)**2, max=1e6)).mean() + ((nu)**2).mean()

            return f
        
        except Exception as e:
            print(f"Error in loss_pde: {e}")
            return torch.tensor(1e6, device=x.device, requires_grad=True)

    # Loss function for initial condition
    def loss_ic(self, x_ic, rho_ic, u_ic, p_ic):
        y_ic = self.net(x_ic)
        rho_ic_nn, p_ic_nn, u_ic_nn = y_ic[:, 0], y_ic[:, 1], y_ic[:, 2]

        # Loss function for the initial condition
        loss_ics = ((u_ic_nn - u_ic) ** 2).mean() + \
                   ((rho_ic_nn - rho_ic) ** 2).mean() + \
                   ((p_ic_nn - p_ic) ** 2).mean()

        return loss_ics

# Generate training data
print("Generating training data...")
x = np.linspace(x_min, x_max, nx)
t = np.linspace(0, t_max, nt)
t_grid, x_grid = np.meshgrid(t, x)
T = t_grid.flatten()[:, None]
X = x_grid.flatten()[:, None]

# Interior points
x_int = X[:, 0][:, None]
t_int = T[:, 0][:, None]
x_int_train = np.hstack((t_int, x_int))

# Initial condition points
x_ic = x_grid[:, 0][:, None]
t_ic = t_grid[:, 0][:, None]
x_ic_train = np.hstack((t_ic, x_ic))

# Generate initial conditions
rho_ic_train, u_ic_train, p_ic_train = IC(x_ic)

# Convert to tensors
x_ic_train = torch.tensor(x_ic_train, dtype=torch.float32).to(device)
x_int_train = torch.tensor(x_int_train, requires_grad=True, dtype=torch.float32).to(device)
rho_ic_train = torch.tensor(rho_ic_train, dtype=torch.float32).to(device)
u_ic_train = torch.tensor(u_ic_train, dtype=torch.float32).to(device)
p_ic_train = torch.tensor(p_ic_train, dtype=torch.float32).to(device)

# Initialize model
print("Building PINN model...")
model = DNN().to(device)

# Training with Adam optimizer first
print("Starting training with Adam optimizer...")
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 5000
tic = time.time()

for epoch in range(epochs):
    optimizer.zero_grad()
    loss_pde = model.loss_pde(x_int_train)
    loss_ic = model.loss_ic(x_ic_train, rho_ic_train, u_ic_train, p_ic_train)
    loss = loss_pde + 1*loss_ic
    loss.backward()
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Clear cache periodically to prevent memory buildup
    if epoch % 100 == 0:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    loss_value = loss.item()
    if epoch % 500 == 0:
        print(f'Epoch {epoch}: loss {loss_value:.6f}')
    
    # Check for NaN or extremely large losses
    if torch.isnan(loss) or loss_value > 1e10:
        print(f"Training unstable at epoch {epoch}, stopping...")
        break
    
    if loss_value < er_c1:
        print(f'Converged at epoch {epoch}!')
        break

# Define closure for LBFGS
def closure():
    try:
        optimizer.zero_grad()
        loss_pde = model.loss_pde(x_int_train)
        loss_ic = model.loss_ic(x_ic_train, rho_ic_train, u_ic_train, p_ic_train)
        loss = loss_pde + 1*loss_ic
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        return loss
    except Exception as e:
        print(f"Error in closure: {e}")
        return torch.tensor(1e6, requires_grad=True)

epochs = 200
toc = time.time()

toc = time.time()
print(f'Training completed in {toc - tic:.2f} seconds (Adam only)')

# Generate predictions
print("Generating predictions and plots...")
x_test = np.linspace(x_min, x_max, ct*nx)
t_test = np.linspace(t_max, t_max, 1)
t_grid_test, x_grid_test = np.meshgrid(t_test, x_test)
T_test = t_grid_test.flatten()[:, None]
X_test = x_grid_test.flatten()[:, None]
x_test_tensor = torch.tensor(np.hstack((T_test, X_test)), dtype=torch.float32).to(device)
u_pred = to_numpy(model(x_test_tensor))

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Density
axes[0, 0].plot(x_test, u_pred[:, 0], 'b-', linewidth=2, label='PINN')
axes[0, 0].set_title('Density ρ(x,t)')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('ρ')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Pressure
axes[0, 1].plot(x_test, u_pred[:, 1], 'r-', linewidth=2, label='PINN')
axes[0, 1].set_title('Pressure p(x,t)')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('p')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Velocity
axes[1, 0].plot(x_test, u_pred[:, 2], 'g-', linewidth=2, label='PINN')
axes[1, 0].set_title('Velocity u(x,t)')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('u')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Viscosity
axes[1, 1].plot(x_test, u_pred[:, 3]**2, 'm-', linewidth=2, label='PINN')
axes[1, 1].set_title('Artificial Viscosity ν²(x,t)')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('ν²')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.suptitle('Sod Shock Tube - Conservative Euler Equations (PINN Solution)', fontsize=14)
plt.tight_layout()
plt.show()

print("Sod Conservative problem completed!")