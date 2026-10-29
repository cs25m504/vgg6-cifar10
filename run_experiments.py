import subprocess
import sys
import time

# List of hyperparameter variations required
activations = ["sigmoid", "relu", "silu", "gelu", "tanh"]
optimizers = ["sgd", "nesterov", "adam", "adagrad", "rmsprop", "nadam"]

epochs = 50
batch_size = 64
learning_rate = 0.001
seed = 42

total_runs = len(activations) * len(optimizers)
current_run = 0

print(f"Starting Hyperparameter Sweep: {total_runs} total combinations.")
start_time = time.time()

for act in activations:
    for opt in optimizers:
        current_run += 1
        print(f"\n==================================================")
        print(f"Run [{current_run}/{total_runs}]: Activation={act.upper()} | Optimizer={opt.upper()}")
        print(f"==================================================")
        
        # Build command to execute train.py
        cmd = [
            sys.executable, "train.py",
            "--activation", act,
            "--optimizer", opt,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", str(learning_rate),
            "--seed", str(seed),
            "--use_wandb", "True"  # Automatically logs each run to W&B
        ]
        
        try:
            # Run the training process
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error occurred during run: Activation={act}, Optimizer={opt}")
            print(f"Error details: {e}")
            continue

elapsed_time = (time.time() - start_time) / 3600
print(f"\n🎉 All {total_runs} experiment combinations completed in {elapsed_time:.2f} hours!")
