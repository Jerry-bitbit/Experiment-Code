import argparse
import json
import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd


class RealDataEnvironment:
    def __init__(self, dataset_name="adult", n_arms=4, total_steps=10000, seed=42):
        self.n_arms = n_arms
        self.total_steps = total_steps
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
        self.step_count = 0

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
            classifier = LogisticRegression(max_iter=500, solver="saga", n_jobs=-1, random_state=self.seed)
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
        self.step_count = 0
        self.current_idx = 0

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

    def step(self, arm):
        """Get next real data sample and calculate reward"""
        if self.step_count >= self.total_steps:
            # Return zeros when maximum steps reached
            return 0.0, 0.0, 0.0, 0

        # Get real data sample
        idx = self.current_idx
        self.current_idx = (self.current_idx + 1) % self.n_samples
        margin = self.margins[idx]
        label = self.labels[idx]

        # Calculate reward
        reward = self.get_reward(margin, label, arm)
        best_reward = self.get_best_arm_reward(margin, label)

        self.step_count += 1

        return reward, best_reward, margin, label

    def is_done(self):
        """Check if environment has reached maximum steps"""
        return self.step_count >= self.total_steps


class OptimizedUCB:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.reset()

    def reset(self):
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.total_count = 0

    def select_arm(self):
        if self.total_count < self.n_arms:
            return self.total_count

        ucb_values = self.values + np.sqrt(2.0 * np.log(self.total_count + 1) / (self.counts + 1e-8))
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.total_count += 1
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) * value + reward) / n


class OptimizedKLUCB:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.reset()

    def reset(self):
        self.counts = np.ones(self.n_arms) * 0.5
        self.values = np.full(self.n_arms, 0.5)
        self.total_count = self.n_arms * 0.5

    def kl_bernoulli(self, p, q):
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        q = np.clip(q, eps, 1 - eps)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def klucb_value(self, arm):
        p = self.values[arm]
        n = self.counts[arm]

        if n < 1e-10:
            return float('inf')

        c = 3
        log_term = (np.log(self.total_count) + c * np.log(np.log(self.total_count + 1e-10))) / n

        low, high = p, 1.0
        for _ in range(20):
            mid = (low + high) / 2
            if self.kl_bernoulli(p, mid) <= log_term:
                low = mid
            else:
                high = mid
        return low

    def select_arm(self):
        ucb_values = [self.klucb_value(arm) for arm in range(self.n_arms)]
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.total_count += 1
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) * value + reward) / n


class SensitivePageHinkley:
    def __init__(self, threshold=5, delta=0.001):
        self.threshold = threshold
        self.delta = delta
        self.reset()

    def reset(self):
        self.cumulative_sum = 0
        self.min_cumulative = float('inf')
        self.sample_count = 0
        self.mean_estimate = 0
        self.reward_buffer = []

    def update(self, reward):
        self.sample_count += 1
        self.reward_buffer.append(reward)

        if len(self.reward_buffer) > 40:
            self.reward_buffer.pop(0)

        current_mean = np.mean(self.reward_buffer)

        if self.sample_count == 1:
            self.mean_estimate = reward
        else:
            self.mean_estimate = 0.97 * self.mean_estimate + 0.03 * reward

        deviation = reward - current_mean - self.delta
        self.cumulative_sum += deviation

        if self.cumulative_sum < self.min_cumulative:
            self.min_cumulative = self.cumulative_sum

        drift_detected = False
        if self.sample_count > 25:
            test_statistic = self.cumulative_sum - self.min_cumulative
            if test_statistic > self.threshold:
                drift_detected = True
                self.reset()

        return drift_detected


def run_real_data_experiment():
    n_arms = 4
    total_steps = 5000  # Further reduced for faster execution
    n_runs = 3

    algorithms = {
        'UCB': OptimizedUCB(n_arms),
        'UCB+PH': OptimizedUCB(n_arms),
        'KL-UCB': OptimizedKLUCB(n_arms),
        'KL-UCB+PH': OptimizedKLUCB(n_arms)
    }

    detectors = {
        'UCB+PH': SensitivePageHinkley(threshold=6, delta=0.002),
        'KL-UCB+PH': SensitivePageHinkley(threshold=4, delta=0.001)
    }

    results = {
        name: {
            'cumulative_regret': np.zeros(total_steps),
            'instant_regret': np.zeros(total_steps),
            'restart_count': 0
        } for name in algorithms
    }

    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs} with real data...")
        env = RealDataEnvironment(dataset_name="adult", n_arms=n_arms,
                                  total_steps=total_steps, seed=run)
        env.reset()

        for algo in algorithms.values():
            algo.reset()
        for detector in detectors.values():
            detector.reset()

        cumulative_regrets = {name: 0 for name in algorithms}

        for t in range(total_steps):
            for name, algorithm in algorithms.items():
                chosen_arm = algorithm.select_arm()
                reward, best_reward, margin, label = env.step(chosen_arm)

                # Skip if environment is done
                if env.is_done():
                    continue

                instant_regret = best_reward - reward
                cumulative_regrets[name] += instant_regret

                results[name]['cumulative_regret'][t] += cumulative_regrets[name] / n_runs
                results[name]['instant_regret'][t] += instant_regret / n_runs

                algorithm.update(chosen_arm, reward)

                if name in detectors:
                    if detectors[name].update(reward):
                        results[name]['restart_count'] += 1
                        algorithm.reset()

        print(
            f"  Run {run + 1} completed - UCB: {cumulative_regrets['UCB']:.1f}, KL-UCB: {cumulative_regrets['KL-UCB']:.1f}")

    return results


def plot_real_data_results(results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    total_steps = len(results['UCB']['cumulative_regret'])
    time_steps = range(total_steps)

    colors = {'UCB': 'red', 'UCB+PH': 'orange', 'KL-UCB': 'blue', 'KL-UCB+PH': 'green'}

    # Plot 1: Cumulative Regret
    for name, data in results.items():
        ax1.plot(time_steps, data['cumulative_regret'],
                 label=name, color=colors[name], linewidth=2.5, alpha=0.9)

    ax1.set_xlabel('Time Steps', fontsize=12)
    ax1.set_ylabel('Cumulative Regret', fontsize=12)
    ax1.set_title('KL-UCB Performance on Real Data (Adult Dataset)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    final_ucb = results['UCB']['cumulative_regret'][-1]
    final_klucb = results['KL-UCB']['cumulative_regret'][-1]
    if final_ucb > 0:
        improvement = ((final_ucb - final_klucb) / final_ucb) * 100
    else:
        improvement = 0

    ax1.text(0.02, 0.98, f'KL-UCB Improvement: {improvement:.1f}%',
             transform=ax1.transAxes, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             verticalalignment='top')

    # Plot 2: Smoothed Instant Regret
    window = 100  # Reduced window for shorter time series
    for name, data in results.items():
        if len(data['instant_regret']) > window:
            smoothed_regret = np.convolve(data['instant_regret'],
                                          np.ones(window) / window, mode='valid')
            ax2.plot(time_steps[:len(smoothed_regret)], smoothed_regret,
                     label=name, color=colors[name], linewidth=1.5, alpha=0.8)

    ax2.set_xlabel('Time Steps', fontsize=12)
    ax2.set_ylabel('Instant Regret (Smoothed)', fontsize=12)
    ax2.set_title('Algorithm Stability on Real Data', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Restart Events
    ph_algorithms = ['UCB+PH', 'KL-UCB+PH']
    restart_counts = [results[name]['restart_count'] for name in ph_algorithms]
    colors_ph = ['orange', 'green']

    bars = ax3.bar(ph_algorithms, restart_counts, color=colors_ph, alpha=0.8)
    ax3.set_xlabel('Algorithm', fontsize=12)
    ax3.set_ylabel('Number of Restarts', fontsize=12)
    ax3.set_title('Page-Hinkley Restart Events with Real Data', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    for bar, count in zip(bars, restart_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                 f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Plot 4: Final Performance Comparison
    final_regrets = {
        'UCB': results['UCB']['cumulative_regret'][-1],
        'KL-UCB': results['KL-UCB']['cumulative_regret'][-1],
        'UCB+PH': results['UCB+PH']['cumulative_regret'][-1],
        'KL-UCB+PH': results['KL-UCB+PH']['cumulative_regret'][-1]
    }

    algorithms_names = ['UCB', 'KL-UCB', 'UCB+PH', 'KL-UCB+PH']
    regrets = [final_regrets[name] for name in algorithms_names]
    bar_colors = ['red', 'blue', 'orange', 'green']

    bars = ax4.bar(algorithms_names, regrets, color=bar_colors, alpha=0.8)
    ax4.set_xlabel('Algorithm', fontsize=12)
    ax4.set_ylabel('Final Cumulative Regret', fontsize=12)
    ax4.set_title('Final Performance on Real Data', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    for bar, regret in zip(bars, regrets):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, height + 2,
                 f'{regret:.0f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('klucb_real_data_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

    return final_regrets, restart_counts


def main():
    print("=" * 60)
    print("KL-UCB vs UCB Comparison with Real Data")
    print("Dataset: Adult (OpenML)")
    print("=" * 60)

    try:
        # Run experiment with real data
        results = run_real_data_experiment()

        # Plot results
        final_regrets, restart_counts = plot_real_data_results(results)

        # Print summary statistics
        print("\n" + "=" * 50)
        print("EXPERIMENT SUMMARY")
        print("=" * 50)
        print(f"Dataset: Adult (OpenML)")
        print(f"Total steps: {len(results['UCB']['cumulative_regret'])}")
        print(f"Number of arms: 4")
        print(f"Number of runs: 3")

        print("\nFinal Cumulative Regrets:")
        for algo, regret in final_regrets.items():
            print(f"  {algo}: {regret:.1f}")

        if final_regrets['UCB'] > 0:
            improvement = ((final_regrets['UCB'] - final_regrets['KL-UCB']) / final_regrets['UCB']) * 100
            print(f"\nKL-UCB Improvement over UCB: {improvement:.1f}%")

        print("\nRestart Events:")
        print(f"  UCB+PH: {restart_counts[0]}")
        print(f"  KL-UCB+PH: {restart_counts[1]}")

        print("\nGenerated: klucb_real_data_performance.png")

    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()