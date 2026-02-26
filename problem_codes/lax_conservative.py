import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt

print("Lax Problem - Conservative Euler Equations")
print("=" * 70)

# Seeds
torch.manual_seed(12)
np.random.seed(12)

# Domain parameters
x_min = 0
x_max = 1
t_max = 0.15
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

# Initial conditions for Lax problem
def IC(x):
    N = len(x)
    rho_init = np.zeros((x.shape[0]))
    u_init = np.zeros((x.shape[0]))
    p_init = np.zeros((x.shape[0]))

    # Lax problem - different from Sod shock tube
    for i in range(N):
        if (x[i] <= 0.5):
            rho_init[i] = 0.445
            p_init[i] = 3.528
            u_init[i] = 0.698
        else:
            rho_init[i] = 0.5
            p_init[i] = 0.571
            u_init[i] = 0

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
        
        # Output layer with 4 outputs (rho, p, u, nu)
        self.net.add_module('Linear_layer_final', nn.Linear(tn, 4))

    # Forward Feed
    def forward(self, x):
        return self.net(x)

    # Loss function for PDE
    def loss_pde(self, x):
        y = self.net(x)
        # Extract rho, p, u, and nu from the network output
        rho, p, u, nu_raw = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]

        # Apply softplus to nu_raw to ensure it's positive (diffusion coefficient)
        nu = F.softplus(nu_raw)

        U2 = rho*u
        U3 = 0.5*rho*u**2 + p/0.4  # Assuming gamma = 1.4

        F2 = rho*u**2 + p
        F3 = u*(U3 + p)

        # Gradients and partial derivatives
        drho_g = gradients(rho, x)[0]
        rho_t, rho_x = drho_g[:, :1], drho_g[:, 1:]

        du_g = gradients(u, x)[0]
        u_t, u_x = du_g[:, :1], du_g[:, 1:]

        # Compute second derivatives
        drho_x = gradients(rho, x)[0][:, 1:]
        rho_xx = gradients(drho_x, x)[0][:, 1:]

        dU2_x = gradients(U2, x)[0][:, 1:]
        U2_xx = gradients(dU2_x, x)[0][:, 1:]

        dU3_x = gradients(U3, x)[0][:, 1:]
        U3_xx = gradients(dU3_x, x)[0][:, 1:]
        
        # Time derivatives of U2 and U3
        dU2_g = gradients(U2, x)[0]
        U2_t, U2_x = dU2_g[:, :1], dU2_g[:, 1:]
        
        dU3_g = gradients(U3, x)[0]
        U3_t, U3_x = dU3_g[:, :1], dU3_g[:, 1:]
        
        # Time derivatives of F2 and F3
        dF2_g = gradients(F2, x)[0]
        F2_t, F2_x = dF2_g[:, :1], dF2_g[:, 1:]
        
        dF3_g = gradients(F3, x)[0]
        F3_t, F3_x = dF3_g[:, :1], dF3_g[:, 1:]

        # d factor as in original code
        d = 0.12*(abs(rho_x)-rho_x) + 1

        # Main PDE residuals using the predicted nu for diffusion terms
        loss_pde_terms = (((rho_t + U2_x - nu*rho_xx)/d)**2).mean() + \
                        (((U2_t + F2_x - nu*U2_xx)/d)**2).mean() + \
                        (((U3_t + F3_x - nu*U3_xx)/d)**2).mean()
        
        # Regularization term for the predicted nu
        nu_regularization_term = 1e-0 * (nu**2).mean()

        f = loss_pde_terms + nu_regularization_term

        return f

    # Loss function for initial condition
    def loss_ic(self, x_ic, rho_ic, u_ic, p_ic):
        y_ic = self.net(x_ic)
        # Extract rho_ic_nn, p_ic_nn, u_ic_nn from the network output (ignore nu for IC)
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


# Training Loop (Adam only)
epochs_to_run = 300
print_every_n_epochs = 50
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Starting training for {epochs_to_run} epochs (Adam only)...")
start_overall_time = time.time()

for epoch in range(1, epochs_to_run + 1):
    optimizer.zero_grad()
    loss_pde = model.loss_pde(x_int_train)
    loss_ic = model.loss_ic(x_ic_train, rho_ic_train, u_ic_train, p_ic_train)
    total_loss = loss_pde + 3 * loss_ic
    total_loss.backward()
    optimizer.step()

    if epoch % print_every_n_epochs == 0:
        elapsed_time = time.time() - start_overall_time
        # Get the mean predicted nu over the training domain for printing
        with torch.no_grad():
            predictions_for_print = model(x_int_train)
            nu_pred_raw_for_print = predictions_for_print[:, 3:4]
            nu_predicted_mean = F.softplus(nu_pred_raw_for_print).mean().item()
        print(f"Epoch {epoch}/{epochs_to_run} | Total Loss: {total_loss.item():.6e} | "
              f"PDE Loss: {loss_pde.item():.6e} | IC Loss: {loss_ic.item():.6e} | "
              f"Mean Predicted nu: {nu_predicted_mean:.6e} | "
              f"Time Elapsed: {elapsed_time:.2f}s")

end_overall_time = time.time()
print(f"Training finished after {epochs_to_run} epochs (Adam only).")
print(f"Total training time: {end_overall_time - start_overall_time:.2f} seconds")

# Generate predictions and plot results
print("Generating predictions and plots...")

# Create test data for plotting at t=t_max across the x domain
t_test = torch.full((nx, 1), t_max, dtype=torch.float32).to(device)
x_test = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device)
test_data = torch.hstack((t_test, x_test))

model.eval()
with torch.no_grad():
    # Get all predictions from the model
    predictions = model(test_data)
    # Separate the outputs: rho, p, u, and the raw nu prediction
    rho_pred, p_pred, u_pred, nu_pred_raw = predictions[:,0], predictions[:,1], predictions[:,2], predictions[:,3]
    # Apply softplus to nu to ensure it's positive for plotting
    nu_pred = F.softplus(nu_pred_raw)

# Plotting results
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(to_numpy(x_test), to_numpy(rho_pred), 'b-', linewidth=2, label='PINN Density')
plt.title('Density at t=0.15')
plt.xlabel('x')
plt.ylabel('ρ')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(to_numpy(x_test), to_numpy(u_pred), 'r-', linewidth=2, label='PINN Velocity')
plt.title('Velocity at t=0.15')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(to_numpy(x_test), to_numpy(p_pred), 'g-', linewidth=2, label='PINN Pressure')
plt.title('Pressure at t=0.15')
plt.xlabel('x')
plt.ylabel('p')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(to_numpy(x_test), to_numpy(nu_pred), 'm-', linewidth=2, label='Predicted nu')
plt.title('Predicted Diffusion Coefficient (ν) at t=0.15')
plt.xlabel('x')
plt.ylabel('ν')
plt.legend()
plt.grid(True)

plt.suptitle('Lax Problem - Conservative Euler Equations (PINN Solution)', fontsize=14)
plt.tight_layout()
plt.show()

print("Lax Conservative problem completed!")