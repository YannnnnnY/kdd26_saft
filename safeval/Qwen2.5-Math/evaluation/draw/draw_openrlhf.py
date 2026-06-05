import matplotlib.pyplot as plt
import os
from datetime import datetime

# --- 1. Data Preparation ---
# The data is manually parsed from your provided output logs, including all learning rates.
data = {
    'train_sft': {
        '2e-5': {
            16: {'gsm8k': 69.9, 'math': 32.4, 'svamp': 79.2, 'asdiv': 79.2, 'mawps': 86.7, 'carp_en': 36.6, 'tabmwp': 42.2, 'minerva_math': 11.0, 'gaokao2023en': 27.5, 'olympiadbench': 13.5, 'college_math': 11.9, 'avg': 44.6},
            32: {'gsm8k': 75.0, 'math': 39.0, 'svamp': 83.4, 'asdiv': 81.2, 'mawps': 89.1, 'carp_en': 41.6, 'tabmwp': 43.9, 'minerva_math': 17.6, 'gaokao2023en': 36.1, 'olympiadbench': 14.4, 'college_math': 13.9, 'avg': 48.7},
            128: {'gsm8k': 74.8, 'math': 43.3, 'svamp': 86.2, 'asdiv': 82.0, 'mawps': 88.6, 'carp_en': 42.9, 'tabmwp': 48.0, 'minerva_math': 19.5, 'gaokao2023en': 34.8, 'olympiadbench': 19.1, 'college_math': 14.2, 'avg': 50.3}
        },
        '5e-6': {
            16: {'gsm8k': 77.6, 'math': 46.0, 'svamp': 87.5, 'asdiv': 82.7, 'mawps': 89.2, 'carp_en': 45.6, 'tabmwp': 43.5, 'minerva_math': 21.3, 'gaokao2023en': 40.5, 'olympiadbench': 20.1, 'college_math': 15.5, 'avg': 51.8},
            32: {'gsm8k': 78.3, 'math': 45.7, 'svamp': 87.6, 'asdiv': 80.2, 'mawps': 88.8, 'carp_en': 45.1, 'tabmwp': 42.6, 'minerva_math': 20.2, 'gaokao2023en': 41.3, 'olympiadbench': 20.7, 'college_math': 16.0, 'avg': 51.5},
            128: {'gsm8k': 77.2, 'math': 46.7, 'svamp': 87.8, 'asdiv': 81.3, 'mawps': 89.2, 'carp_en': 45.9, 'tabmwp': 46.8, 'minerva_math': 21.7, 'gaokao2023en': 40.3, 'olympiadbench': 21.3, 'college_math': 15.1, 'avg': 52.1}
        },
        '2e-6': {
            16: {'gsm8k': 75.9, 'math': 43.6, 'svamp': 85.9, 'asdiv': 80.0, 'mawps': 87.9, 'carp_en': 42.3, 'tabmwp': 40.6, 'minerva_math': 20.2, 'gaokao2023en': 38.2, 'olympiadbench': 18.8, 'college_math': 13.9, 'avg': 49.8},
            32: {'gsm8k': 76.7, 'math': 43.6, 'svamp': 86.6, 'asdiv': 80.9, 'mawps': 88.7, 'carp_en': 42.9, 'tabmwp': 42.6, 'minerva_math': 19.5, 'gaokao2023en': 39.7, 'olympiadbench': 20.1, 'college_math': 13.8, 'avg': 50.5},
            128: {'gsm8k': 77.0, 'math': 42.6, 'svamp': 86.5, 'asdiv': 80.7, 'mawps': 88.1, 'carp_en': 43.5, 'tabmwp': 42.6, 'minerva_math': 18.8, 'gaokao2023en': 37.7, 'olympiadbench': 19.6, 'college_math': 13.1, 'avg': 50.0}
        },
        '5e-7': {
            16: {'gsm8k': 70.7, 'math': 39.8, 'svamp': 81.7, 'asdiv': 76.4, 'mawps': 84.3, 'carp_en': 41.4, 'tabmwp': 40.8, 'minerva_math': 14.3, 'gaokao2023en': 37.7, 'olympiadbench': 15.7, 'college_math': 12.6, 'avg': 46.9},
            32: {'gsm8k': 73.0, 'math': 40.1, 'svamp': 83.1, 'asdiv': 77.4, 'mawps': 85.4, 'carp_en': 41.0, 'tabmwp': 41.8, 'minerva_math': 16.9, 'gaokao2023en': 35.8, 'olympiadbench': 15.7, 'college_math': 12.6, 'avg': 47.5},
            128: {'gsm8k': 76.3, 'math': 42.0, 'svamp': 86.0, 'asdiv': 80.0, 'mawps': 87.7, 'carp_en': 43.4, 'tabmwp': 43.0, 'minerva_math': 20.2, 'gaokao2023en': 38.7, 'olympiadbench': 15.7, 'college_math': 13.0, 'avg': 49.6}
        }
    },
    'train_critical': {
        '2e-5': {
            16: {'gsm8k': 57.1, 'math': 17.1, 'svamp': 67.6, 'asdiv': 68.9, 'mawps': 82.2, 'carp_en': 28.6, 'tabmwp': 36.3, 'minerva_math': 5.9, 'gaokao2023en': 17.7, 'olympiadbench': 4.1, 'college_math': 9.5, 'avg': 35.9},
            32: {'gsm8k': 64.8, 'math': 32.4, 'svamp': 72.9, 'asdiv': 75.8, 'mawps': 86.1, 'carp_en': 41.6, 'tabmwp': 38.2, 'minerva_math': 12.1, 'gaokao2023en': 26.5, 'olympiadbench': 12.3, 'college_math': 18.0, 'avg': 43.7},
            128: {'gsm8k': 74.1, 'math': 41.0, 'svamp': 85.9, 'asdiv': 81.9, 'mawps': 90.3, 'carp_en': 42.4, 'tabmwp': 46.4, 'minerva_math': 19.9, 'gaokao2023en': 33.8, 'olympiadbench': 18.7, 'college_math': 17.9, 'avg': 50.2}
        },
        '5e-6': {
            16: {'gsm8k': 80.5, 'math': 49.4, 'svamp': 87.0, 'asdiv': 83.9, 'mawps': 92.1, 'carp_en': 48.7, 'tabmwp': 49.6, 'minerva_math': 23.9, 'gaokao2023en': 39.0, 'olympiadbench': 23.6, 'college_math': 19.9, 'avg': 54.3},
            32: {'gsm8k': 80.3, 'math': 52.3, 'svamp': 88.9, 'asdiv': 84.3, 'mawps': 91.9, 'carp_en': 48.8, 'tabmwp': 47.2, 'minerva_math': 23.2, 'gaokao2023en': 44.2, 'olympiadbench': 25.0, 'college_math': 24.9, 'avg': 55.5},
            128: {'gsm8k': 81.2, 'math': 49.5, 'svamp': 89.2, 'asdiv': 84.0, 'mawps': 91.4, 'carp_en': 47.6, 'tabmwp': 49.5, 'minerva_math': 20.6, 'gaokao2023en': 42.1, 'olympiadbench': 26.2, 'college_math': 18.9, 'avg': 54.6}
        },
        '2e-6': {
            16: {'gsm8k': 82.3, 'math': 50.1, 'svamp': 89.6, 'asdiv': 84.2, 'mawps': 92.4, 'carp_en': 46.8, 'tabmwp': 46.3, 'minerva_math': 22.8, 'gaokao2023en': 40.3, 'olympiadbench': 25.5, 'college_math': 19.1, 'avg': 54.5},
            32: {'gsm8k': 81.4, 'math': 49.8, 'svamp': 89.5, 'asdiv': 84.2, 'mawps': 92.5, 'carp_en': 46.5, 'tabmwp': 45.1, 'minerva_math': 20.2, 'gaokao2023en': 42.6, 'olympiadbench': 25.3, 'college_math': 17.9, 'avg': 54.1},
            128: {'gsm8k': 81.0, 'math': 47.3, 'svamp': 89.6, 'asdiv': 83.6, 'mawps': 91.1, 'carp_en': 45.3, 'tabmwp': 43.7, 'minerva_math': 21.3, 'gaokao2023en': 38.2, 'olympiadbench': 23.6, 'college_math': 15.8, 'avg': 52.8}
        },
        '5e-7': {
            16: {'gsm8k': 78.8, 'math': 44.3, 'svamp': 87.3, 'asdiv': 81.3, 'mawps': 88.6, 'carp_en': 44.6, 'tabmwp': 43.8, 'minerva_math': 15.4, 'gaokao2023en': 39.2, 'olympiadbench': 18.1, 'college_math': 14.3, 'avg': 50.5},
            32: {'gsm8k': 80.3, 'math': 46.3, 'svamp': 88.7, 'asdiv': 82.8, 'mawps': 90.2, 'carp_en': 46.3, 'tabmwp': 43.3, 'minerva_math': 14.3, 'gaokao2023en': 41.3, 'olympiadbench': 20.0, 'college_math': 15.3, 'avg': 51.7},
            128: {'gsm8k': 82.9, 'math': 55.9, 'svamp': 89.9, 'asdiv': 83.9, 'mawps': 91.3, 'carp_en': 51.8, 'tabmwp': 43.9, 'minerva_math': 19.5, 'gaokao2023en': 45.2, 'olympiadbench': 28.9, 'college_math': 21.8, 'avg': 55.9}
        }
    }
}


# --- 2. Plotting Setup ---
# Create a directory to save the output plots
save_dir = "evaluation_plots"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Get current timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define plotting parameters based on the data structure
batch_sizes = [16, 32, 128]
# Order learning rates from highest to lowest for legend clarity
learning_rates = ['2e-5', '5e-6', '2e-6', '5e-7']
methods = ['train_sft', 'train_critical']
metrics = [
    'gsm8k', 'math', 'svamp', 'asdiv', 'mawps', 'carp_en',
    'tabmwp', 'minerva_math', 'gaokao2023en', 'olympiadbench', 'college_math', 'avg'
]

# Define colors, markers, and display names for each training method
colors = {
    'train_sft': '#1f77b4',      # Blue
    'train_critical': '#ff7f0e'  # Orange
}
markers = {
    'train_sft': 'o',           # Circle
    'train_critical': 's'       # Square
}
method_names = {
    'train_sft': 'SFT',
    'train_critical': 'Critical'
}
# Define linestyles for different learning rates
linestyles = {
    '2e-5': '-',   # Solid
    '5e-6': '--',  # Dashed
    '2e-6': ':',   # Dotted
    '5e-7': '-.'   # Dash-dot
}


# Set global matplotlib parameters for a consistent and bold look
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14

# --- 3. Generate Plots ---
# Create a figure and a 3x4 grid of subplots
fig, axes = plt.subplots(3, 4, figsize=(24, 18))
fig.suptitle('Comparison of SFT vs. Critical Training Across Metrics', fontsize=26, fontweight='bold')

# Flatten the 2D axes array for easy iteration
ax_flat = axes.flatten()

for i, metric in enumerate(metrics):
    ax = ax_flat[i]
    
    # Plot data for each method and learning rate combination
    for method in methods:
        for lr in learning_rates:
            # Get the linestyle for the current learning rate
            linestyle = linestyles[lr]
            
            # Extract scores for the current metric across all batch sizes
            scores = [data[method][lr][bs][metric] for bs in batch_sizes]
            
            # Create a descriptive label for the legend
            label = f'{method_names[method]} (LR={lr})'
            
            # Plot the data
            ax.plot(batch_sizes, scores,
                      color=colors[method],
                      marker=markers[method],
                      linestyle=linestyle,
                      linewidth=2.5,
                      markersize=8,
                      label=label)
    
    # Configure each subplot
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{metric.upper()}', fontsize=16)
    ax.set_xticks(batch_sizes) # Ensure ticks are at the exact batch sizes
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc='best', fontsize=9) # Adjusted fontsize for better fit

# Adjust layout to prevent titles and labels from overlapping
plt.tight_layout(rect=[0, 0, 1, 0.96])

# --- 4. Save the Figure ---
# Define filenames with timestamp
png_filename = f"sft_vs_critical_comparison_{timestamp}.png"
pdf_filename = f"sft_vs_critical_comparison_{timestamp}.pdf"
png_path = os.path.join(save_dir, png_filename)
pdf_path = os.path.join(save_dir, pdf_filename)

# Save the figure in both PNG and PDF formats
fig.savefig(png_path)
fig.savefig(pdf_path)

plt.close(fig)

print(f"✅ Plot saved successfully!")
print(f"PNG image: {png_path}")
print(f"PDF document: {pdf_path}")