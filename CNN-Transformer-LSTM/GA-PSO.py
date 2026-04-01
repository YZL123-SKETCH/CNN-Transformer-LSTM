import torch
import torch.nn as nn
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import random
import os

from dataset import DataLoader
from Module.CNN_Transformer_LSTM import CNN_Transformer_LSTM
from config import get_args

# =============================================================================
# PSO-GA Hybrid Hyperparameter Optimization
# Optimized parameters:
# hidden_size, num_layers, dropout, learning_rate, batch_size, multi-head attention
# Multi-heads strictly satisfy: embed_dim % mha_heads == 0
# =============================================================================

# Load configuration & device
args = get_args()
device = args.device

# Hyperparameter search ranges
param_bounds = {
    'hidden_size':    (16, 128),
    'num_layers':     (1, 4),
    'dropout':        (0.0, 0.5),
    'learning_rate':  (1e-4, 1e-3),
    'batch_size':     (8, 64),
    'mha_heads':      [1, 2, 4, 8, 16, 32, 64]   # Valid heads for multi-head attention
}

# Load dataset
loader = DataLoader(
    filename=args.filename,
    split=0.77,
    cols=[]
)

# Record optimization history
history = {
    'iterations': [], 'loss': [],
    'hidden_size': [], 'num_layers': [], 'dropout': [],
    'learning_rate': [], 'batch_size': [], 'mha_heads': []
}

# Fix random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# =============================================================================
# Model evaluation function
# =============================================================================
def evaluate_model(params):
    hidden_size   = int(params['hidden_size'])
    num_layers    = int(params['num_layers'])
    dropout       = float(params['dropout'])
    learning_rate = float(params['learning_rate'])
    batch_size    = int(params['batch_size'])
    mha_heads     = int(params['mha_heads'])

    # Build CNN-Transformer-LSTM model
    model = CNN_Transformer_LSTM(
        input_size=args.input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=args.output_size,
        dropout=dropout,
        mha_heads=mha_heads
    ).to(device)

    criterion = nn.MSEloss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    val_losses = []
    for epoch in range(1000):
        model.train()
        for x_r, x_t, x_l, y in loader.generate_train_batch(args.sequence_length, batch_size):
            x_r = torch.tensor(x_r, dtype=torch.float32).unsqueeze(-1).to(device)
            x_t = torch.tensor(x_t, dtype=torch.float32).unsqueeze(-1).to(device)
            x_l = torch.tensor(x_l, dtype=torch.float32).unsqueeze(-1).to(device)
            y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

            optimizer.zero_grad()
            pred = model(x_r, x_t, x_l)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_batch_count = 0
        with torch.no_grad():
            for x_r, x_t, x_l, y in loader.generate_test_batch(args.sequence_length, batch_size):
                x_r = torch.tensor(x_r, dtype=torch.float32).unsqueeze(-1).to(device)
                x_t = torch.tensor(x_t, dtype=torch.float32).unsqueeze(-1).to(device)
                x_l = torch.tensor(x_l, dtype=torch.float32).unsqueeze(-1).to(device)
                y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

                pred = model(x_r, x_t, x_l)
                total_val_loss += criterion(pred, y).item()
                val_batch_count += 1

        if val_batch_count > 0:
            val_losses.append(total_val_loss / val_batch_count)

    # Final loss: average of last 10 validation losses
    final_loss = np.mean(val_losses[-10:])

    # Save history
    history['iterations'].append(len(history['iterations']) + 1)
    history['loss'].append(final_loss)
    history['hidden_size'].append(hidden_size)
    history['num_layers'].append(num_layers)
    history['dropout'].append(dropout)
    history['learning_rate'].append(learning_rate)
    history['batch_size'].append(batch_size)
    history['mha_heads'].append(mha_heads)

    print(f"[{datetime.datetime.now()}] Val Loss: {final_loss:.6f} | Params: {params}")
    return final_loss

# =============================================================================
# Initialize population for genetic algorithm
# =============================================================================
def generate_initial_population(pop_size=10):
    population = []
    for _ in range(pop_size):
        individual = {
            'hidden_size': random.randint(*param_bounds['hidden_size']),
            'num_layers': random.randint(*param_bounds['num_layers']),
            'dropout': round(random.uniform(*param_bounds['dropout']), 3),
            'learning_rate': round(random.uniform(*param_bounds['learning_rate']), 5),
            'batch_size': random.randint(*param_bounds['batch_size']),
            'mha_heads': random.choice(param_bounds['mha_heads'])
        }
        population.append(individual)
    return population

# =============================================================================
# GA crossover & mutation
# =============================================================================
def crossover_and_mutate(parent1, parent2):
    child = {}
    for key in parent1:
        child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
        if random.random() < 0.2:
            if key == 'mha_heads':
                child[key] = random.choice(param_bounds['mha_heads'])
            elif key in ['hidden_size', 'num_layers', 'batch_size']:
                child[key] = random.randint(*param_bounds[key])
            else:
                child[key] = round(random.uniform(*param_bounds[key]), 3 if key == 'dropout' else 5)
    return child

# =============================================================================
# PSO-GA Hybrid Optimization Main Function
# =============================================================================
def pso_ga_optimization(max_iters=150, population_size=10, no_improve_limit=5):
    population = generate_initial_population(population_size)
    velocity = [np.zeros(len(param_bounds)) for _ in range(population_size)]
    pbest = population.copy()
    pbest_scores = [evaluate_model(ind) for ind in pbest]

    gbest_idx = np.argmin(pbest_scores)
    gbest = pbest[gbest_idx]
    gbest_score = pbest_scores[gbest_idx]

    no_improve_count = 0
    dim_keys = list(param_bounds.keys())

    for it in range(max_iters):
        print(f"\nIteration {it+1}/{max_iters} | Best Val Loss: {gbest_score:.6f}")

        for i, individual in enumerate(population):
            new_params = {}
            for d, key in enumerate(dim_keys):
                # Keep multi-head attention valid (discrete selection only)
                if key == "mha_heads":
                    new_params[key] = random.choice(param_bounds['mha_heads'])
                    continue

                # PSO velocity & position update
                phi1, phi2 = random.random(), random.random()
                w, c1, c2 = 0.5, 1.5, 1.5
                v_old = velocity[i][d]
                x = individual[key]
                lb, ub = param_bounds[key]

                norm_x = (x - lb) / (ub - lb)
                norm_pbest = (pbest[i][key] - lb) / (ub - lb)
                norm_gbest = (gbest[key] - lb) / (ub - lb)

                v_new = w * v_old + c1 * phi1 * (norm_pbest - norm_x) + c2 * phi2 * (norm_gbest - norm_x)
                norm_x_new = np.clip(norm_x + v_new, 0, 1)
                new_val = lb + norm_x_new * (ub - lb)

                if isinstance(lb, int):
                    new_val = int(np.clip(round(new_val), lb, ub))
                else:
                    new_val = round(np.clip(new_val, lb, ub), 3 if key == 'dropout' else 5)

                new_params[key] = new_val
                velocity[i][d] = v_new

            population[i] = new_params

        # Evaluate all particles
        scores = [evaluate_model(ind) for ind in population]

        # Update personal best
        for i in range(population_size):
            if scores[i] < pbest_scores[i]:
                pbest[i] = population[i]
                pbest_scores[i] = scores[i]

        # Update global best
        current_min_score = min(pbest_scores)
        if current_min_score < gbest_score:
            gbest_score = current_min_score
            gbest = pbest[np.argmin(pbest_scores)]
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Trigger GA if no improvement
        if no_improve_count >= no_improve_limit:
            print("No obvious improvement, apply GA crossover and mutation...")
            new_population = [crossover_and_mutate(*random.sample(pbest, 2)) for _ in range(population_size)]
            population = new_population
            velocity = [np.zeros(len(dim_keys)) for _ in range(population_size)]
            no_improve_count = 0

    print("\n================ Optimization Finished ================")
    print("Best Hyperparameters:", gbest)
    print("Best Validation Loss:", gbest_score)
    return gbest

# =============================================================================
# Run optimization & save results
# =============================================================================
if __name__ == "__main__":
    best_hyperparams = pso_ga_optimization()

    # Save optimization record to Excel
    result_df = pd.DataFrame(history)
    result_df.to_excel("PSO_GA_Optimization_Results.xlsx", index=False)

    # Plot convergence curve
    plt.figure(figsize=(10, 6))
    plt.plot(history['iterations'], history['loss'], marker='o', color='blue')
    plt.title("PSO-GA Convergence Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Validation Loss (MSE)")
    plt.grid(True)
    plt.savefig("PSO_GA_Convergence_Curve.png", dpi=300)
    plt.show()

