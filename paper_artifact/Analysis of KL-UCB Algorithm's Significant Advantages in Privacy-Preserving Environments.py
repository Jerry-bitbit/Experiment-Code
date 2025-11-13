import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os


class RealDataEnvironment:
    def __init__(self, dataset_name="adult", n_arms=5, seed=42):
        self.n_arms = n_arms
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Load real dataset
        self.data = self.load_real_dataset(dataset_name)
        if self.data is None:
            raise ValueError(f"Failed to load dataset: {dataset_name}")

        self.margins = self.data["margins"]
        self.labels = self.data["labels"]
        self.n_samples = len(self.margins)
        self.current_idx = 0

        # Initialize arm parameters based on real data characteristics
        self.arm_offsets = np.linspace(-2.0, 2.0, n_arms)

        print(f"Real data environment initialized with {self.n_samples} samples")

    def load_real_dataset(self, dataset_name="adult", test_size=0.3):
        """Load and prepare real dataset"""
        try:
            if dataset_name.lower() == "adult":
                data = fetch_openml("adult", version=2, as_frame=True)
            else:  # german credit
                data = fetch_openml("credit-g", version=1, as_frame=True)

            df = data.frame.copy()
            df.replace("?", np.nan, inplace=True)
            df.dropna(inplace=True)

            # Prepare target variable
            if dataset_name == "adult":
                y = (df["class"] == ">50K").astype(int).values
            else:
                y = (df["class"] == "bad").astype(int).values

            X = df.drop(columns=["class"])

            # Identify numeric and categorical columns
            numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
            categorical_cols = [c for c in X.columns if c not in numeric_cols]

            # Create preprocessing pipeline
            preprocessor = ColumnTransformer([
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
            ])

            # Split data
            X_train, X_stream, y_train, y_stream = train_test_split(
                X, y, test_size=test_size, random_state=self.seed, stratify=y
            )

            # Reset indices
            X_train = X_train.reset_index(drop=True)
            X_stream = X_stream.reset_index(drop=True)

            # Fit classifier
            X_train_processed = preprocessor.fit_transform(X_train)
            classifier = LogisticRegression(max_iter=500, solver="saga", n_jobs=1, random_state=self.seed)
            classifier.fit(X_train_processed, y_train)

            # Get decision scores for stream data
            X_stream_processed = preprocessor.transform(X_stream)
            margins = classifier.decision_function(X_stream_processed)

            print(f"{dataset_name} dataset loaded: {len(margins)} samples")

            return {
                "margins": margins.astype(float),
                "labels": y_stream.astype(int),
                "dataset_name": dataset_name
            }

        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            return None

    def reset(self):
        self.current_idx = 0

    def get_sample(self):
        """Get next sample from the dataset"""
        idx = self.current_idx
        self.current_idx = (self.current_idx + 1) % self.n_samples
        return self.margins[idx], self.labels[idx]

    def get_reward(self, margin, label, arm):
        """Calculate reward based on arm selection and real data"""
        # Apply arm-specific offset to margin
        adjusted_margin = margin + self.arm_offsets[arm]
        # Predict using adjusted margin
        prediction = 1 if adjusted_margin >= 0 else 0
        # Reward is 1 if prediction matches true label
        return 1.0 if prediction == label else 0.0

    def get_best_arm_reward(self, margin, label):
        """Get reward of best arm for regret calculation"""
        rewards = [self.get_reward(margin, label, arm) for arm in range(self.n_arms)]
        return max(rewards)


class UCB:
    def __init__(self, n_arms, alpha=1.0):
        self.n_arms = n_arms
        self.alpha = alpha
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        total_counts = np.sum(self.counts)
        if total_counts == 0:
            return np.random.randint(self.n_arms)

        ucb_values = self.values + self.alpha * np.sqrt(2 * np.log(total_counts + 1) / (self.counts + 1e-8))
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward


class KL_UCB:
    def __init__(self, n_arms, c=0.0, eps=1e-15):
        self.n_arms = n_arms
        self.c = c
        self.eps = eps
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def kl_bernoulli(self, p, q):
        p = np.clip(p, self.eps, 1 - self.eps)
        q = np.clip(q, self.eps, 1 - self.eps)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def kl_ucb_value(self, p, t, arm):
        if self.counts[arm] == 0:
            return float('inf')

        log_term = (np.log(t + 1) + self.c * np.log(np.log(t + 1 + self.eps))) / self.counts[arm]
        right = p
        left = p

        for _ in range(10):
            q = (left + right) / 2
            kl_val = self.kl_bernoulli(p, q)
            if kl_val < log_term:
                left = q
            else:
                right = q

        return (left + right) / 2

    def select_arm(self, t):
        if t < self.n_arms:
            return t

        ucb_values = []
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                ucb_values.append(float('inf'))
            else:
                ucb_val = self.kl_ucb_value(self.values[arm], t, arm)
                ucb_values.append(ucb_val)

        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward


def run_real_data_experiment(algorithm_class, params, ldp_levels, bucket_sizes, n_arms=5, T=500, n_runs=3):
    """运行参数扫描实验并返回热力图数据 - 使用真实数据"""
    results_matrix = np.zeros((len(ldp_levels), len(bucket_sizes)))

    for i, ldp in enumerate(ldp_levels):
        for j, buckets in enumerate(bucket_sizes):
            total_regrets = []

            for run in range(n_runs):
                # 使用真实数据环境
                env = RealDataEnvironment(dataset_name="adult", n_arms=n_arms, seed=run)

                if algorithm_class == UCB:
                    agent = algorithm_class(n_arms, alpha=params.get('alpha', 1.0))
                else:
                    agent = algorithm_class(n_arms, c=params.get('c', 0.0))

                cumulative_regret = 0

                for t in range(T):
                    # 获取真实数据样本
                    margin, label = env.get_sample()

                    # 选择臂
                    if algorithm_class == UCB:
                        arm = agent.select_arm()
                    else:
                        arm = agent.select_arm(t)

                    # 计算真实奖励
                    true_reward = env.get_reward(margin, label, arm)
                    best_reward = env.get_best_arm_reward(margin, label)

                    # 添加LDP噪声
                    observed_reward = true_reward
                    if ldp > 0:
                        if np.random.random() < ldp:
                            observed_reward = 1 - observed_reward  # 翻转比特

                    # 分桶量化
                    if buckets > 1:
                        observed_reward = np.floor(observed_reward * buckets) / buckets

                    # 更新代理
                    agent.update(arm, observed_reward)

                    # 计算遗憾
                    instant_regret = best_reward - true_reward
                    cumulative_regret += instant_regret

                total_regrets.append(cumulative_regret)

            # 计算平均最终遗憾
            avg_final_regret = np.mean(total_regrets)
            results_matrix[i, j] = avg_final_regret

            print(f"LDP: {ldp}, Buckets: {buckets}, Regret: {avg_final_regret:.1f}")

    return results_matrix


def plot_heatmaps():
    """生成热力图 - 使用真实数据"""
    # 参数范围
    ldp_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    bucket_sizes = [1, 2, 4, 8, 16, 32]

    # 设置样式
    plt.style.use('default')
    sns.set_palette("viridis")

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # 算法配置
    algorithms = [
        (UCB, {'alpha': 1.0}, "UCB (α=1.0)"),
        (UCB, {'alpha': 2.0}, "UCB (α=2.0)"),
        (KL_UCB, {'c': 0.0}, "KL-UCB (c=0.0)"),
        (KL_UCB, {'c': 1.0}, "KL-UCB (c=1.0)")
    ]

    all_results = []
    min_regret = float('inf')
    max_regret = 0

    # 运行所有实验
    for idx, (algo_class, params, title) in enumerate(algorithms):
        print(f"Running {title} with real data...")
        results = run_real_data_experiment(algo_class, params, ldp_levels, bucket_sizes, n_arms=5, T=500, n_runs=3)
        all_results.append(results)
        min_regret = min(min_regret, np.min(results))
        max_regret = max(max_regret, np.max(results))

    # 绘制热力图
    for idx, (results, (algo_class, params, title)) in enumerate(zip(all_results, algorithms)):
        ax = axes[idx]

        # 创建热力图
        im = ax.imshow(results, cmap='viridis', aspect='auto',
                       vmin=min_regret, vmax=max_regret)

        # 设置坐标轴
        ax.set_xticks(np.arange(len(bucket_sizes)))
        ax.set_yticks(np.arange(len(ldp_levels)))
        ax.set_xticklabels([f'{b}' for b in bucket_sizes])
        ax.set_yticklabels([f'{ldp:.1f}' for ldp in ldp_levels])

        # 设置标签
        ax.set_xlabel('Number of Buckets')
        ax.set_ylabel('LDP Noise Level')
        ax.set_title(f'{title}\nFinal Regret Heatmap (Real Data)')

        # 在热力图上显示数值
        for i in range(len(ldp_levels)):
            for j in range(len(bucket_sizes)):
                text = ax.text(j, i, f'{results[i, j]:.0f}',
                               ha="center", va="center", color="w", fontsize=9)

    # 添加颜色条
    fig.colorbar(im, ax=axes, shrink=0.6, label='Cumulative Regret')

    plt.tight_layout()
    plt.savefig('ucb_klucb_heatmaps_real_data.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 生成对比热力图
    plot_comparison_heatmap(all_results, algorithms, ldp_levels, bucket_sizes)


def plot_comparison_heatmap(all_results, algorithms, ldp_levels, bucket_sizes):
    """生成算法对比热力图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    klucb_best = np.minimum(all_results[2], all_results[3])  # KL-UCB最佳
    ucb_best = np.minimum(all_results[0], all_results[1])  # UCB最佳
    advantage = ucb_best - klucb_best  # 正值表示KL-UCB更好

    im1 = axes[0].imshow(advantage, cmap='RdYlBu', aspect='auto')
    axes[0].set_xticks(np.arange(len(bucket_sizes)))
    axes[0].set_yticks(np.arange(len(ldp_levels)))
    axes[0].set_xticklabels([f'{b}' for b in bucket_sizes])
    axes[0].set_yticklabels([f'{ldp:.1f}' for ldp in ldp_levels])
    axes[0].set_xlabel('Number of Buckets')
    axes[0].set_ylabel('LDP Noise Level')
    axes[0].set_title('KL-UCB Advantage over UCB\n(Positive = KL-UCB Better)')

    for i in range(len(ldp_levels)):
        for j in range(len(bucket_sizes)):
            text = axes[0].text(j, i, f'{advantage[i, j]:.0f}',
                                ha="center", va="center", color="black", fontsize=9)

    best_algo = np.zeros_like(advantage, dtype=int)
    for i in range(len(ldp_levels)):
        for j in range(len(bucket_sizes)):
            best_algo[i, j] = 0 if ucb_best[i, j] < klucb_best[i, j] else 1

    im2 = axes[1].imshow(best_algo, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
    axes[1].set_xticks(np.arange(len(bucket_sizes)))
    axes[1].set_yticks(np.arange(len(ldp_levels)))
    axes[1].set_xticklabels([f'{b}' for b in bucket_sizes])
    axes[1].set_yticklabels([f'{ldp:.1f}' for ldp in ldp_levels])
    axes[1].set_xlabel('Number of Buckets')
    axes[1].set_ylabel('LDP Noise Level')
    axes[1].set_title('Best Algorithm\n(0=UCB, 1=KL-UCB)')

    fig.colorbar(im1, ax=axes[0], shrink=0.8, label='Regret Advantage')
    fig.colorbar(im2, ax=axes[1], shrink=0.8, ticks=[0, 1], label='Algorithm')

    plt.tight_layout()
    plt.savefig('algorithm_comparison_heatmap_real_data.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print(f"当前工作目录: {os.getcwd()}")
    print("Starting UCB vs KL-UCB parameter sweep with REAL DATA...")
    print("Dataset: Adult (OpenML)")
    print("=" * 60)

    # 设置随机种子以确保可重复性
    np.random.seed(42)

    plot_heatmaps()
    print("Heatmaps generated successfully with real data!")
    print("Generated files:")
    print("  - ucb_klucb_heatmaps_real_data.png")
    print("  - algorithm_comparison_heatmap_real_data.png")