import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt

print("Sod Shock Tube Problem - Non-Conservative Euler Equations")
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
er_c1 = 1e-3

# Device setup
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Calculate gradients
def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or np.ndarray, but got {}'.format(type(input)))

# Initial conditions for Sod shock tube
def IC(x):
    N = len(x)
    rho_init = np.zeros((x.shape[0]))
    u_init = np.zeros((x.shape[0]))
    p_init = np.zeros((x.shape[0]))
    
    for i in range(N):
        if (x[i] <= 0.5):
            rho_init[i] = 1.0
            p_init[i] = 1.0
        else:
            rho_init[i] = 0.125
            p_init[i] = 0.1
    
    return rho_init, u_init, p_init

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
        self.net.add_module('Linear_layer_final', nn.Linear(tn, 3))  # Only 3 outputs for non-conservative
    
    def forward(self, x):
        return self.net(x)
    
    # Non-conservative form of Euler equations
    def loss_pde(self, x):
        y = self.net(x)
        rho, p, u = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        
        gamma = 1.4
        
        # Gradients
        drho_g = gradients(rho, x)[0]
        rho_t, rho_x = drho_g[:, :1], drho_g[:, 1:]
        
        du_g = gradients(u, x)[0]
        u_t, u_x = du_g[:, :1], du_g[:, 1:]
        
        dp_g = gradients(p, x)[0]
        p_t, p_x = dp_g[:, :1], dp_g[:, 1:]
        
        # Non-conservative Euler equations
        # Continuity: ∂ρ/∂t + ∂(ρu)/∂x = 0
        eq1 = rho_t + rho*u_x + u*rho_x
        
        # Momentum: ∂(ρu)/∂t + ∂(ρu² + p)/∂x = 0  
        # Expanded: ρ∂u/∂t + u∂ρ/∂t + ∂p/∂x + 2ρu∂u/∂x + u²∂ρ/∂x = 0
        eq2 = rho*u_t + u*rho_t + p_x + 2*rho*u*u_x + u**2*rho_x
        
        # Energy (simplified): ∂p/∂t + u∂p/∂x + γp∂u/∂x = 0
        eq3 = p_t + u*p_x + gamma*p*u_x
        
        # Simple artificial viscosity
        nu = 0.01
        
        # Add simple diffusion terms
        rho_xx = gradients(rho_x, x)[0][:, 1:]
        u_xx = gradients(u_x, x)[0][:, 1:]
        p_xx = gradients(p_x, x)[0][:, 1:]
        
        f = (eq1 - nu*rho_xx).pow(2).mean() + (eq2 - nu*u_xx).pow(2).mean() + (eq3 - nu*p_xx).pow(2).mean()
        
        return f
    
    def loss_ic(self, x_ic, rho_ic, u_ic, p_ic):
        y_ic = self.net(x_ic)
        rho_ic_nn, p_ic_nn, u_ic_nn = y_ic[:, 0], y_ic[:, 1], y_ic[:, 2]
        
        loss_ics = ((u_ic_nn - u_ic) ** 2).mean() + \
                   ((rho_ic_nn - rho_ic) ** 2).mean() + \
                   ((p_ic_nn - p_ic) ** 2).mean()
        
        return loss_ics

print("Generating training data...")
x = np.linspace(x_min, x_max, nx)
t = np.linspace(0, t_max, nt)
t_grid, x_grid = np.meshgrid(t, x)
T = t_grid.flatten()[:, None]
X = x_grid.flatten()[:, None]

x_int = X[:, 0][:, None]
t_int = T[:, 0][:, None]  
x_int_train = np.hstack((t_int, x_int))

x_ic = x_grid[:, 0][:, None]
t_ic = t_grid[:, 0][:, None]
x_ic_train = np.hstack((t_ic, x_ic))

rho_ic_train, u_ic_train, p_ic_train = IC(x_ic)

# Convert to tensors
x_ic_train = torch.tensor(x_ic_train, dtype=torch.float32).to(device)
x_int_train = torch.tensor(x_int_train, requires_grad=True, dtype=torch.float32).to(device)
rho_ic_train = torch.tensor(rho_ic_train, dtype=torch.float32).to(device)
u_ic_train = torch.tensor(u_ic_train, dtype=torch.float32).to(device)
p_ic_train = torch.tensor(p_ic_train, dtype=torch.float32).to(device)

print("Building PINN model...")
model = DNN().to(device)

def closure():
    optimizer.zero_grad()
    loss_pde = model.loss_pde(x_int_train)
    loss_ic = model.loss_ic(x_ic_train, rho_ic_train, u_ic_train, p_ic_train)
    loss = loss_pde + loss_ic
    loss.backward()
    return loss

print("Starting training...")
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(1, 1001):
    loss = optimizer.step(closure)
    loss_value = loss.item() if not isinstance(loss, float) else loss
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: loss {loss_value:.6f}')

print("Generating predictions and plots...")
x_test = np.linspace(x_min, x_max, nx)
t_test = np.linspace(t_max, t_max, 1)
t_grid_test, x_grid_test = np.meshgrid(t_test, x_test)
T_test = t_grid_test.flatten()[:, None]
X_test = x_grid_test.flatten()[:, None]
x_test_tensor = torch.tensor(np.hstack((T_test, X_test)), dtype=torch.float32).to(device)
u_pred = to_numpy(model(x_test_tensor))

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Density
axes[0].plot(x_test, u_pred[:, 0], 'b-', linewidth=2, label='PINN')
axes[0].set_title('Density ρ(x,t)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('ρ') 
axes[0].legend()
axes[0].grid(True)

# Pressure
axes[1].plot(x_test, u_pred[:, 1], 'r-', linewidth=2, label='PINN')
axes[1].set_title('Pressure p(x,t)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('p')
axes[1].legend()
axes[1].grid(True)

# Velocity  
axes[2].plot(x_test, u_pred[:, 2], 'g-', linewidth=2, label='PINN')
axes[2].set_title('Velocity u(x,t)')
axes[2].set_xlabel('x')
axes[2].set_ylabel('u')
axes[2].legend()
axes[2].grid(True)

plt.suptitle('Sod Shock Tube - Non-Conservative Euler Equations (PINN Solution)', fontsize=14) 
plt.tight_layout()
plt.show()

print("Sod Non-Conservative problem completed!")