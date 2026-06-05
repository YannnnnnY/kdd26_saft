import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# --- 1. 数据准备 ---

# 数据源一: 您最初的 "Original" 训练结果 (来自日志文件)
data = {
    'train_sft': {
        '2e-5': {
            16: {'gsm8k': 70.5, 'math': 30.9, 'svamp': 81.0, 'asdiv': 78.4, 'mawps': 87.1, 'carp_en': 35.8, 'tabmwp': 45.0, 'minerva_math': 11.0, 'gaokao2023en': 28.6, 'olympiadbench': 11.3, 'college_math': 12.2, 'avg': 44.7},
            32: {'gsm8k': 73.1, 'math': 39.0, 'svamp': 85.0, 'asdiv': 80.1, 'mawps': 88.9, 'carp_en': 39.7, 'tabmwp': 49.1, 'minerva_math': 16.5, 'gaokao2023en': 32.5, 'olympiadbench': 15.7, 'college_math': 13.1, 'avg': 48.4},
            128: {'gsm8k': 77.2, 'math': 45.1, 'svamp': 86.9, 'asdiv': 82.0, 'mawps': 88.4, 'carp_en': 43.8, 'tabmwp': 49.4, 'minerva_math': 18.8, 'gaokao2023en': 40.3, 'olympiadbench': 19.7, 'college_math': 15.2, 'avg': 51.5}
        },
        '2e-6': {
            16: {'gsm8k': 78.7, 'math': 44.6, 'svamp': 86.6, 'asdiv': 80.8, 'mawps': 89.0, 'carp_en': 44.3, 'tabmwp': 43.5, 'minerva_math': 17.6, 'gaokao2023en': 39.2, 'olympiadbench': 19.9, 'college_math': 13.7, 'avg': 50.7},
            32: {'gsm8k': 77.6, 'math': 44.6, 'svamp': 87.9, 'asdiv': 81.4, 'mawps': 89.4, 'carp_en': 45.5, 'tabmwp': 46.6, 'minerva_math': 19.1, 'gaokao2023en': 37.1, 'olympiadbench': 19.1, 'college_math': 13.4, 'avg': 51.1},
            128: {'gsm8k': 77.9, 'math': 45.0, 'svamp': 87.8, 'asdiv': 82.1, 'mawps': 89.2, 'carp_en': 45.7, 'tabmwp': 49.3, 'minerva_math': 18.4, 'gaokao2023en': 40.0, 'olympiadbench': 19.9, 'college_math': 14.5, 'avg': 51.8}
        }
    },
    'train_critical': {
        '2e-5': {
            16: {'gsm8k': 56.4, 'math': 18.5, 'svamp': 63.1, 'asdiv': 70.2, 'mawps': 80.0, 'carp_en': 28.3, 'tabmwp': 33.2, 'minerva_math': 9.6, 'gaokao2023en': 20.3, 'olympiadbench': 4.4, 'college_math': 8.4, 'avg': 35.7},
            32: {'gsm8k': 64.3, 'math': 29.3, 'svamp': 75.6, 'asdiv': 74.8, 'mawps': 88.0, 'carp_en': 38.0, 'tabmwp': 42.2, 'minerva_math': 10.3, 'gaokao2023en': 27.5, 'olympiadbench': 10.7, 'college_math': 15.4, 'avg': 43.3},
            128: {'gsm8k': 73.5, 'math': 44.9, 'svamp': 86.3, 'asdiv': 83.5, 'mawps': 91.9, 'carp_en': 45.6, 'tabmwp': 47.4, 'minerva_math': 17.6, 'gaokao2023en': 37.1, 'olympiadbench': 20.1, 'college_math': 23.2, 'avg': 51.9}
        },
        '2e-6': {
            16: {'gsm8k': 82.1, 'math': 52.5, 'svamp': 90.5, 'asdiv': 85.1, 'mawps': 92.0, 'carp_en': 47.6, 'tabmwp': 48.6, 'minerva_math': 24.6, 'gaokao2023en': 45.2, 'olympiadbench': 25.8, 'college_math': 21.9, 'avg': 56.0},
            32: {'gsm8k': 81.5, 'math': 50.7, 'svamp': 89.7, 'asdiv': 84.7, 'mawps': 92.2, 'carp_en': 46.2, 'tabmwp': 48.8, 'minerva_math': 25.0, 'gaokao2023en': 40.3, 'olympiadbench': 27.1, 'college_math': 18.8, 'avg': 55.0},
            128: {'gsm8k': 80.2, 'math': 50.0, 'svamp': 88.2, 'asdiv': 84.0, 'mawps': 92.1, 'carp_en': 49.6, 'tabmwp': 50.5, 'minerva_math': 21.0, 'gaokao2023en': 43.9, 'olympiadbench': 25.5, 'college_math': 19.0, 'avg': 54.9}
        }
    }
}

# 数据源二: Llama Factory (LF) 的运行结果
lf_data = {
    'train_sft': {
        '2e-5': {
            16: {'gsm8k': 69.4, 'math': 31.2, 'svamp': 80.3, 'asdiv': 78.4, 'mawps': 86.7, 'carp_en': 35.2, 'tabmwp': 43.2, 'minerva_math': 10.7, 'gaokao2023en': 25.2, 'olympiadbench': 10.2, 'college_math': 11.8, 'avg': 43.8},
            32: {'gsm8k': 74.8, 'math': 37.7, 'svamp': 84.8, 'asdiv': 80.9, 'mawps': 87.8, 'carp_en': 39.8, 'tabmwp': 45.5, 'minerva_math': 18.4, 'gaokao2023en': 35.3, 'olympiadbench': 17.2, 'college_math': 14.2, 'avg': 48.8},
            128: {'gsm8k': 77.6, 'math': 47.5, 'svamp': 85.9, 'asdiv': 83.1, 'mawps': 89.0, 'carp_en': 43.5, 'tabmwp': 55.1, 'minerva_math': 20.2, 'gaokao2023en': 40.5, 'olympiadbench': 20.9, 'college_math': 18.6, 'avg': 52.9}
        },
        '2e-6': {
            16: {'gsm8k': 77.8, 'math': 43.9, 'svamp': 87.9, 'asdiv': 81.8, 'mawps': 89.1, 'carp_en': 44.0, 'tabmwp': 45.2, 'minerva_math': 17.6, 'gaokao2023en': 39.2, 'olympiadbench': 20.0, 'college_math': 13.5, 'avg': 50.9},
            32: {'gsm8k': 76.6, 'math': 42.8, 'svamp': 87.7, 'asdiv': 82.2, 'mawps': 89.5, 'carp_en': 43.5, 'tabmwp': 49.1, 'minerva_math': 18.0, 'gaokao2023en': 38.7, 'olympiadbench': 18.1, 'college_math': 13.8, 'avg': 50.9},
            128: {'gsm8k': 74.3, 'math': 43.8, 'svamp': 76.5, 'asdiv': 74.5, 'mawps': 79.5, 'carp_en': 45.1, 'tabmwp': 38.9, 'minerva_math': 14.0, 'gaokao2023en': 38.4, 'olympiadbench': 19.0, 'college_math': 14.5, 'avg': 47.1}
        }
    },
    'train_critical': {
        '2e-5': {
            16: {'gsm8k': 52.0, 'math': 17.0, 'svamp': 66.4, 'asdiv': 67.9, 'mawps': 82.6, 'carp_en': 29.7, 'tabmwp': 35.3, 'minerva_math': 6.2, 'gaokao2023en': 19.2, 'olympiadbench': 4.6, 'college_math': 9.6, 'avg': 35.5},
            32: {'gsm8k': 62.9, 'math': 27.6, 'svamp': 71.7, 'asdiv': 74.3, 'mawps': 84.6, 'carp_en': 36.0, 'tabmwp': 37.9, 'minerva_math': 12.1, 'gaokao2023en': 26.0, 'olympiadbench': 9.6, 'college_math': 13.4, 'avg': 41.5},
            128: {'gsm8k': 74.2, 'math': 44.3, 'svamp': 86.2, 'asdiv': 82.3, 'mawps': 90.1, 'carp_en': 44.3, 'tabmwp': 43.2, 'minerva_math': 20.6, 'gaokao2023en': 35.1, 'olympiadbench': 21.5, 'college_math': 22.0, 'avg': 51.3}
        },
        '2e-6': {
            16: {'gsm8k': 81.1, 'math': 48.0, 'svamp': 90.1, 'asdiv': 84.8, 'mawps': 92.8, 'carp_en': 46.1, 'tabmwp': 48.1, 'minerva_math': 22.1, 'gaokao2023en': 37.9, 'olympiadbench': 27.1, 'college_math': 18.2, 'avg': 54.2},
            32: {'gsm8k': 81.2, 'math': 48.6, 'svamp': 90.2, 'asdiv': 85.6, 'mawps': 92.4, 'carp_en': 44.8, 'tabmwp': 47.6, 'minerva_math': 21.3, 'gaokao2023en': 40.0, 'olympiadbench': 23.3, 'college_math': 17.2, 'avg': 53.8},
            128: {'gsm8k': 77.9, 'math': 39.6, 'svamp': 88.8, 'asdiv': 83.8, 'mawps': 90.0, 'carp_en': 40.4, 'tabmwp': 49.0, 'minerva_math': 17.3, 'gaokao2023en': 34.5, 'olympiadbench': 15.3, 'college_math': 12.8, 'avg': 49.9}
        }
    }
}

# 将 LF 数据整合到主数据结构中，并使用新键名
data['lf_train_sft'] = lf_data['train_sft']
data['lf_train_critical'] = lf_data['train_critical']


# --- 2. 绘图设置 ---
# 创建保存目录
save_dir = "training_results_plots_final"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 获取当前时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 定义全部四种方法的绘图参数
batch_sizes = [16, 32, 128]
learning_rates = ['2e-5', '2e-6']
methods = ['train_sft', 'train_critical', 'lf_train_sft', 'lf_train_critical']
metrics = [
    'gsm8k', 'math', 'svamp', 'asdiv', 'mawps', 'carp_en', 
    'tabmwp', 'minerva_math', 'gaokao2023en', 'olympiadbench', 'college_math', 'avg'
]

# 为四种方法设置清晰的颜色、标记和名称
colors = {
    'train_sft': '#1f77b4',         # Original SFT: 蓝色
    'train_critical': '#ff7f0e',    # Original Critical: 橙色
    'lf_train_sft': '#2ca02c',      # LF SFT: 绿色
    'lf_train_critical': '#d62728'  # LF Critical: 红色
}
markers = {
    'train_sft': 'o',               # SFT类型用圆形
    'train_critical': 's',          # Critical类型用方形
    'lf_train_sft': 'o',            # SFT类型用圆形
    'lf_train_critical': 's'        # Critical类型用方形
}
method_names = {
    'train_sft': 'SFT (Original)',
    'train_critical': 'Critical (Original)',
    'lf_train_sft': 'LF SFT',
    'lf_train_critical': 'LF Critical'
}

# 设置matplotlib参数
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14

# --- 3. 开始绘图 ---
# 增大画布尺寸以容纳更多图例
fig, axes = plt.subplots(3, 4, figsize=(26, 20))
fig.suptitle('Comprehensive Comparison of All Training Runs', fontsize=26, fontweight='bold')

ax_flat = axes.flatten()

for i, metric in enumerate(metrics):
    ax = ax_flat[i]
    
    # 循环绘制所有组合的曲线
    for method in methods:
        for lr in learning_rates:
            linestyle = '-' if lr == '2e-5' else '--'
            scores = [data[method][lr][bs][metric] for bs in batch_sizes]
            label = f'{method_names[method]} (LR={lr})'
            
            ax.plot(batch_sizes, scores, 
                   color=colors[method],
                   marker=markers[method],
                   linestyle=linestyle,
                   linewidth=2.5, 
                   markersize=8,
                   label=label)
    
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{metric.upper()}', fontsize=16)
    ax.set_xticks(batch_sizes)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # 调整图例字体大小以适应8个条目
    ax.legend(loc='best', fontsize=8) 

# 自动调整子图布局
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存最终的组合图像
png_filename = f"full_comparison_{timestamp}.png"
pdf_filename = f"full_comparison_{timestamp}.pdf"
png_path = os.path.join(save_dir, png_filename)
pdf_path = os.path.join(save_dir, pdf_filename)

fig.savefig(png_path)
fig.savefig(pdf_path)

print(f"最终完整对比图像 (PNG) 已保存至: {png_path}")
print(f"最终完整对比图像 (PDF) 已保存至: {pdf_path}")

plt.close(fig)