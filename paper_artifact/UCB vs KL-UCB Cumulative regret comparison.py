import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
from scipy import stats
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class AdultBanditEnvironment:
    """
    Adult dataset bandit environment for UCB and KL-UCB algorithms
    """

    def __init__(self, n_arms=10, random_state=42):
        self.n_arms = n_arms
        self.random_state = random_state
        np.random.seed(random_state)

        # Load and preprocess Adult dataset
        self.X, self.y = self._load_adult_data()
        self.n_samples = len(self.X)

        # Create synthetic rewards for each arm
        self.arm_rewards = self._create_arm_rewards()

        # Reset environment
        self.reset()

    def _load_adult_data(self):
        """Load and preprocess Adult dataset from OpenML"""
        print("Loading Adult dataset from OpenML...")
        try:
            # Fetch Adult dataset from OpenML
            adult = fetch_openml(name='adult', version=2, as_frame=True)
            X = adult.data
            y = adult.target

            # Preprocessing
            # Handle categorical variables
            categorical_columns = X.select_dtypes(include=['category', 'object']).columns
            numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

            # Encode categorical variables
            label_encoders = {}
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le

            # Encode target variable
            y = LabelEncoder().fit_transform(y)

            # Standardize numerical features
            scaler = StandardScaler()
            X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

            # Remove any remaining NaN values
            X = X.fillna(0)

            print(f"Adult dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
            return X.values, y

        except Exception as e:
            print(f"Error loading Adult dataset: {e}")
            print("Creating synthetic dataset as fallback...")
            return self._create_synthetic_data()

    def _create_synthetic_data(self):
        """Create synthetic dataset if OpenML fails"""
        n_samples = 2000
        n_features = 20

        # Create synthetic features
        X = np.random.normal(0, 1, (n_samples, n_features))

        # Create synthetic binary classification target
        # with some meaningful patterns
        weights = np.random.normal(0, 1, n_features)
        logits = X @ weights + np.random.normal(0, 0.5, n_samples)
        y = (logits > 0).astype(int)

        return X, y

    def _create_arm_rewards(self):
        """Create reward distributions for each arm based on dataset characteristics"""
        # Use dataset features to create meaningful reward distributions
        n_clusters = self.n_arms

        # Simple clustering based on first two principal components
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)

        # Create arms based on data density
        arm_centers = []
        for i in range(n_clusters):
            # Sample points from dense regions
            center_idx = np.random.choice(len(X_pca), size=100, replace=False)
            center = np.mean(X_pca[center_idx], axis=0)
            arm_centers.append(center)

        arm_rewards = []
        for i in range(n_clusters):
            # Assign base success probability based on arm quality
            base_prob = 0.3 + 0.6 * (i / n_clusters)  # Arms have different qualities

            # Add some randomness
            noise = np.random.normal(0, 0.1)
            success_prob = np.clip(base_prob + noise, 0.1, 0.9)

            arm_rewards.append({
                'success_prob': success_prob,
                'center': arm_centers[i],
                'variance': 0.1 + 0.1 * np.random.random()
            })

        # Sort arms by quality for meaningful comparison
        arm_rewards.sort(key=lambda x: x['success_prob'])
        return arm_rewards

    def reset(self):
        """Reset environment state"""
        self.current_step = 0
        return self._get_context()

    def _get_context(self):
        """Get current context (feature vector)"""
        if self.current_step < self.n_samples:
            return self.X[self.current_step]
        else:
            # Wrap around if we exceed dataset size
            idx = self.current_step % self.n_samples
            return self.X[idx]

    def step(self, arm):
        """Execute one step: choose arm and get reward"""
        if self.current_step >= 10000:  # Limit steps for practical reasons
            return None, 0, True

        context = self._get_context()

        # Get arm reward probability
        arm_info = self.arm_rewards[arm]
        base_prob = arm_info['success_prob']

        # Add context-dependent component
        context_effect = 0.1 * np.mean(context[:5])  # Use first 5 features
        success_prob = np.clip(base_prob + context_effect, 0.1, 0.9)

        # Sample reward
        reward = np.random.binomial(1, success_prob)

        self.current_step += 1
        done = self.current_step >= 10000

        return context, reward, done


class UCB:
    """Upper Confidence Bound algorithm"""

    def __init__(self, n_arms, alpha=2.0):
        self.n_arms = n_arms
        self.alpha = alpha
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_steps = 0

    def select_arm(self):
        """Select arm using UCB strategy"""
        if self.total_steps < self.n_arms:
            # Initial exploration: try each arm once
            return self.total_steps

        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            if self.counts[arm] > 0:
                bonus = np.sqrt(self.alpha * np.log(self.total_steps) / self.counts[arm])
                ucb_values[arm] = self.values[arm] + bonus
            else:
                ucb_values[arm] = float('inf')

        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        """Update algorithm with observed reward"""
        self.counts[chosen_arm] += 1
        self.total_steps += 1

        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]

        # Update running average
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward


class KLUCB:
    """KL-UCB algorithm for Bernoulli rewards"""

    def __init__(self, n_arms, c=0, tolerance=1e-4, max_iterations=50):
        self.n_arms = n_arms
        self.c = c
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_steps = 0

    def kl_bernoulli(self, p, q):
        """KL divergence between two Bernoulli distributions"""
        eps = 1e-15  # Avoid log(0)
        p = np.clip(p, eps, 1 - eps)
        q = np.clip(q, eps, 1 - eps)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def kl_ucb_value(self, arm):
        """Compute KL-UCB value for an arm"""
        if self.counts[arm] == 0:
            return float('inf')

        p = self.values[arm]
        log_term = np.log(self.total_steps + 1e-10) / self.counts[arm]
        log_term += self.c * np.log(np.log(self.total_steps + 1e-10))

        # Binary search for q
        low = p
        high = 1.0 - 1e-10

        for _ in range(self.max_iterations):
            q = (low + high) / 2
            kl = self.kl_bernoulli(p, q)

            if kl < log_term:
                low = q
            else:
                high = q

            if high - low < self.tolerance:
                break

        return (low + high) / 2

    def select_arm(self):
        """Select arm using KL-UCB strategy"""
        if self.total_steps < self.n_arms:
            return self.total_steps

        klucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            klucb_values[arm] = self.kl_ucb_value(arm)

        return np.argmax(klucb_values)

    def update(self, chosen_arm, reward):
        """Update algorithm with observed reward"""
        self.counts[chosen_arm] += 1
        self.total_steps += 1

        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]

        # Update running average
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward


def run_experiment(algorithm_class, n_arms=10, n_steps=5000, random_state=42):
    """Run bandit experiment with given algorithm"""
    print(f"Running {algorithm_class.__name__} experiment...")

    # Create environment and algorithm
    env = AdultBanditEnvironment(n_arms=n_arms, random_state=random_state)
    algorithm = algorithm_class(n_arms=n_arms)

    # Storage for results
    cumulative_regret = []
    instant_regrets = []
    cumulative_reward = 0

    # Find best arm for regret calculation
    best_arm_probs = [arm['success_prob'] for arm in env.arm_rewards]
    best_arm_prob = max(best_arm_probs)

    for step in range(n_steps):
        # Select arm
        arm = algorithm.select_arm()

        # Get reward
        _, reward, done = env.step(arm)

        if done:
            break

        # Update algorithm
        algorithm.update(arm, reward)

        # Calculate regret
        instant_regret = best_arm_prob - env.arm_rewards[arm]['success_prob']
        instant_regrets.append(instant_regret)
        cumulative_reward += reward

        # Calculate cumulative regret
        current_regret = (step + 1) * best_arm_prob - cumulative_reward
        cumulative_regret.append(current_regret)

    return {
        'cumulative_regret': np.array(cumulative_regret),
        'instant_regret': np.array(instant_regrets),
        'final_regret': cumulative_regret[-1] if cumulative_regret else 0,
        'total_reward': cumulative_reward,
        'arm_counts': algorithm.counts
    }


def run_comparison_experiment():
    """Run comparison experiment between UCB and KL-UCB"""
    print("Starting UCB vs KL-UCB comparison experiment...")
    print("=" * 60)

    # Experiment parameters
    n_arms = 10
    n_steps = 3000
    n_seeds = 5

    # Storage for results
    ucb_results = []
    klucb_results = []

    for seed in range(n_seeds):
        print(f"\n--- Running seed {seed + 1}/{n_seeds} ---")

        # Run UCB
        ucb_result = run_experiment(UCB, n_arms=n_arms, n_steps=n_steps, random_state=seed)
        ucb_results.append(ucb_result)

        # Run KL-UCB
        klucb_result = run_experiment(KLUCB, n_arms=n_arms, n_steps=n_steps, random_state=seed + 1000)
        klucb_results.append(klucb_result)

        print(f"UCB final regret: {ucb_result['final_regret']:.2f}")
        print(f"KL-UCB final regret: {klucb_result['final_regret']:.2f}")

    return ucb_results, klucb_results


def create_comparison_plot(ucb_results, klucb_results):
    """Create comparison plot"""
    print("Generating comparison plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Extract cumulative regrets
    ucb_regrets = [result['cumulative_regret'] for result in ucb_results]
    klucb_regrets = [result['cumulative_regret'] for result in klucb_results]

    # Ensure consistent length
    min_length = min(min(len(r) for r in ucb_regrets), min(len(r) for r in klucb_regrets))
    time_steps = np.arange(min_length)

    # Calculate statistics
    ucb_data = np.array([r[:min_length] for r in ucb_regrets])
    klucb_data = np.array([r[:min_length] for r in klucb_regrets])

    ucb_mean = np.mean(ucb_data, axis=0)
    ucb_std = np.std(ucb_data, axis=0)
    klucb_mean = np.mean(klucb_data, axis=0)
    klucb_std = np.std(klucb_data, axis=0)

    # Plot cumulative regret comparison
    ax1.plot(time_steps, ucb_mean, 'b-', linewidth=2, label='UCB', alpha=0.8)
    ax1.plot(time_steps, klucb_mean, 'r-', linewidth=2, label='KL-UCB', alpha=0.8)

    ax1.fill_between(time_steps, ucb_mean - ucb_std, ucb_mean + ucb_std, color='blue', alpha=0.2)
    ax1.fill_between(time_steps, klucb_mean - klucb_std, klucb_mean + klucb_std, color='red', alpha=0.2)

    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Cumulative Regret', fontsize=12)
    ax1.set_title('UCB vs KL-UCB: Cumulative Regret', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot final regret distribution
    ucb_final = [result['final_regret'] for result in ucb_results]
    klucb_final = [result['final_regret'] for result in klucb_results]

    ax2.boxplot([ucb_final, klucb_final], labels=['UCB', 'KL-UCB'])
    ax2.set_ylabel('Final Regret', fontsize=12)
    ax2.set_title('Final Regret Distribution', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Add statistics
    final_ucb = np.mean(ucb_final)
    final_klucb = np.mean(klucb_final)

    if final_ucb > 0:
        improvement = (final_ucb - final_klucb) / final_ucb * 100
    else:
        improvement = 0

    stats_text = f'Final Regret:\nUCB: {final_ucb:.1f} ± {np.std(ucb_final):.1f}\nKL-UCB: {final_klucb:.1f} ± {np.std(klucb_final):.1f}\nImprovement: {improvement:.1f}%'

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


def save_results(ucb_results, klucb_results):
    """Save experiment results"""
    output_dir = "adult_bandit_results"
    os.makedirs(output_dir, exist_ok=True)

    # Save summary statistics
    summary = {
        'ucb_final_regrets': [r['final_regret'] for r in ucb_results],
        'klucb_final_regrets': [r['final_regret'] for r in klucb_results],
        'ucb_mean_regret': np.mean([r['final_regret'] for r in ucb_results]),
        'klucb_mean_regret': np.mean([r['final_regret'] for r in klucb_results]),
        'improvement_percentage': ((np.mean([r['final_regret'] for r in ucb_results]) -
                                    np.mean([r['final_regret'] for r in klucb_results])) /
                                   np.mean([r['final_regret'] for r in ucb_results]) * 100)
    }

    with open(os.path.join(output_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save time series data for each seed
    for i, (ucb_result, klucb_result) in enumerate(zip(ucb_results, klucb_results)):
        seed_data = pd.DataFrame({
            'step': np.arange(len(ucb_result['cumulative_regret'])),
            'ucb_cumulative_regret': ucb_result['cumulative_regret'],
            'ucb_instant_regret': ucb_result['instant_regret'],
            'klucb_cumulative_regret': klucb_result['cumulative_regret'],
            'klucb_instant_regret': klucb_result['instant_regret']
        })
        seed_data.to_csv(os.path.join(output_dir, f'timeseries_seed{i + 1}.csv'), index=False)

    print(f"Results saved to {output_dir}/")


def main():
    """Main function"""
    print("UCB vs KL-UCB Comparison on Adult Dataset")
    print("=" * 50)

    # Run experiments
    ucb_results, klucb_results = run_comparison_experiment()

    # Create and save plot
    fig = create_comparison_plot(ucb_results, klucb_results)

    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ucb_vs_klucb_adult_comparison.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")

    # Save results
    save_results(ucb_results, klucb_results)

    # Print final statistics
    print("\n=== Final Statistics ===")
    ucb_final = [r['final_regret'] for r in ucb_results]
    klucb_final = [r['final_regret'] for r in klucb_results]

    print(f"UCB final regret: {np.mean(ucb_final):.1f} ± {np.std(ucb_final):.1f}")
    print(f"KL-UCB final regret: {np.mean(klucb_final):.1f} ± {np.std(klucb_final):.1f}")

    improvement = (np.mean(ucb_final) - np.mean(klucb_final)) / np.mean(ucb_final) * 100
    print(f"KL-UCB improvement: {improvement:.1f}%")

    # Statistical test
    t_stat, p_value = stats.ttest_rel(ucb_final, klucb_final)
    print(f"Paired t-test: t = {t_stat:.3f}, p = {p_value:.4f}")

    if p_value < 0.05:
        print("Difference is statistically significant (p < 0.05)")
    else:
        print("Difference is not statistically significant")

    print("\nExperiment completed!")


if __name__ == "__main__":
    main()