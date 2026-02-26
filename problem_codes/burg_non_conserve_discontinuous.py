import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

print("Burgers Equation - Non-Conservative with Discontinuous Initial Conditions")
print("=" * 75)

# Set seeds and device
torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Domain and data
Tf = 0.2
N_u, N_f = 100, 10000
lb, ub = np.array([-1.0, 0.0]), np.array([1.0, 0.2])
nu = 0.01  # Fixed viscosity for non-conservative form

def gen_data():
    x = np.linspace(-1, 1, 256)[:, None]
    t = np.linspace(0, Tf, 100)[:, None]
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_ic = np.where(X_star[:, 1] == 0.0, np.where(X_star[:, 0] < 0, 1.0, 0.0), np.nan)
    u_star = np.copy(u_ic)
    u_star[np.isnan(u_star)] = 0.0
    return X_star, u_star[:, None], x, t

def lhs_sample(N, lb, ub):
    return lb + (ub - lb) * lhs(2, N)

# Standard PINN model for non-conservative Burgers
class BurgersPINN(nn.Module):
    def __init__(self, layers, lb, ub):
        super().__init__()
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
    
    def forward(self, x):
        x_norm = (x - self.lb) / (self.ub - self.lb)
        for i, layer in enumerate(self.layers[:-1]):
            x_norm = torch.tanh(layer(x_norm))
        return self.layers[-1](x_norm)
    
    def loss(self, xu, u, xf):
        # Data loss
        u_pred = self.forward(xu)
        loss_u = nn.MSELoss()(u_pred, u)
        
        # Physics loss (non-conservative Burgers: u_t + u*u_x = nu*u_xx)
        xf.requires_grad_(True)
        u_f = self.forward(xf)
        grads = autograd.grad(u_f, xf, torch.ones_like(u_f), create_graph=True)[0]
        u_x, u_t = grads[:, 0:1], grads[:, 1:2]
        u_xx = autograd.grad(u_x, xf, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        
        # Non-conservative Burgers equation
        f_pred = u_t + u_f * u_x - nu * u_xx
        loss_f = nn.MSELoss()(f_pred, torch.zeros_like(f_pred))
        
        return loss_u + loss_f

print("Generating data...")
X_star, u_star, x, t = gen_data()
t0_indices = np.where(X_star[:, 1] == 0.0)[0]
idx_u = np.random.choice(t0_indices, N_u, replace=False)
X_u = torch.tensor(X_star[idx_u], dtype=torch.float32).to(device)
U_u = torch.tensor(u_star[idx_u], dtype=torch.float32).to(device)
X_f = torch.tensor(lhs_sample(N_f, lb, ub), dtype=torch.float32).to(device)

print("Building PINN model...")
model = BurgersPINN([2, 20, 20, 20, 20, 1], lb, ub).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Starting training...")
for i in range(10000):
    optimizer.zero_grad()
    loss = model.loss(X_u, U_u, X_f)
    loss.backward()
    optimizer.step()
    if i % 1000 == 0:
        print(f"Iter {i}, Loss: {loss.item():.4e}")

print("Generating predictions and plots...")
X_test = torch.tensor(X_star, dtype=torch.float32).to(device)
u_pred = model.forward(X_test).cpu().detach().numpy().reshape(len(t), len(x))

X, T = np.meshgrid(x, t)
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
cp = ax1.contourf(T, X, u_pred, 100, cmap="rainbow")
fig.colorbar(cp, ax=ax1)
ax1.set_title('Burgers Non-Conservative Discontinuous IC')
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('Space (x)')
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(T, X, u_pred, cmap="rainbow")
ax2.set_title('3D surface')
ax2.set_xlabel('Time (t)')
ax2.set_ylabel('Space (x)')
ax2.set_zlabel('u(x,t)')
plt.tight_layout()
plt.show()

print("Burgers Non-Conservative Discontinuous IC problem completed!")