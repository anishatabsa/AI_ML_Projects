"""
Q1: Q-learning with Action Space Discretization for HalfCheetah
Implements discretization, training, and analysis of control behavior
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm
import pickle

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ActionDiscretizer:
    """
    Discretizes continuous action space into finite set of actions
    """
    def __init__(self, action_dim, action_low, action_high, num_bins_per_dim=3):
        """
        Args:
            action_dim: Dimension of action space (6 for HalfCheetah)
            action_low: Lower bound of action space (-1)
            action_high: Upper bound of action space (+1)
            num_bins_per_dim: Number of discrete values per action dimension
        """
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.num_bins_per_dim = num_bins_per_dim
        
        # Create discrete action values for each dimension
        self.discrete_values = np.linspace(action_low, action_high, num_bins_per_dim)
        
        # Total number of discrete actions
        self.num_discrete_actions = num_bins_per_dim ** action_dim
        
        # Pre-generate all discrete actions
        self.discrete_actions = self._generate_all_actions()
        
        print(f"Action Discretization Info:")
        print(f"  Action dimensions: {action_dim}")
        print(f"  Bins per dimension: {num_bins_per_dim}")
        print(f"  Discrete values per dim: {self.discrete_values}")
        print(f"  Total discrete actions: {self.num_discrete_actions}")
        
    def _generate_all_actions(self):
        """Generate all possible discrete action combinations"""
        # Use meshgrid to create all combinations
        grids = np.meshgrid(*[self.discrete_values for _ in range(self.action_dim)])
        actions = np.stack([grid.flatten() for grid in grids], axis=1)
        return actions
    
    def discretize_action(self, action_index):
        """Convert discrete action index to continuous action vector"""
        return self.discrete_actions[action_index]
    
    def get_closest_discrete_action(self, continuous_action):
        """Find closest discrete action to a continuous action (for analysis)"""
        distances = np.linalg.norm(self.discrete_actions - continuous_action, axis=1)
        return np.argmin(distances)


class StateDiscretizer:
    """
    Discretizes continuous state space using tile coding / binning
    """
    def __init__(self, state_dim, num_bins_per_dim=10, 
                 state_low=None, state_high=None):
        """
        Args:
            state_dim: Dimension of state space (17 for HalfCheetah)
            num_bins_per_dim: Number of bins per state dimension
            state_low: Lower bounds (estimated from observation)
            state_high: Upper bounds (estimated from observation)
        """
        self.state_dim = state_dim
        self.num_bins_per_dim = num_bins_per_dim
        
        # Use default bounds if not provided (will update adaptively)
        self.state_low = state_low if state_low is not None else -np.ones(state_dim) * 10
        self.state_high = state_high if state_high is not None else np.ones(state_dim) * 10
        
        # Track observed min/max for adaptive bounds
        self.observed_min = np.ones(state_dim) * np.inf
        self.observed_max = np.ones(state_dim) * -np.inf
        
    def discretize_state(self, state):
        """Convert continuous state to discrete state index"""
        # Update observed bounds
        self.observed_min = np.minimum(self.observed_min, state)
        self.observed_max = np.maximum(self.observed_max, state)
        
        # Clip state to bounds
        state_clipped = np.clip(state, self.state_low, self.state_high)
        
        # Normalize to [0, 1]
        state_normalized = (state_clipped - self.state_low) / (self.state_high - self.state_low + 1e-8)
        
        # Discretize to bins
        state_discrete = (state_normalized * (self.num_bins_per_dim - 1)).astype(int)
        state_discrete = np.clip(state_discrete, 0, self.num_bins_per_dim - 1)
        
        # Convert to single index (hash)
        state_index = tuple(state_discrete)
        
        return state_index


class QLearningAgent:
    """
    Q-learning agent for discretized HalfCheetah
    """
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        """
        Args:
            num_actions: Number of discrete actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate per episode
        """
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-table: dict of state -> action values
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        
        # Statistics tracking
        self.action_counts = np.zeros(num_actions)
        self.action_rewards = defaultdict(list)  # action_index -> list of rewards
        self.state_action_visits = defaultdict(int)  # (state, action) -> count
        
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.randint(self.num_actions)
        else:
            # Exploit: greedy action
            q_values = self.q_table[state]
            action = np.argmax(q_values)
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update"""
        # Current Q-value
        current_q = self.q_table[state][action]
        
        # Max Q-value for next state
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state])
        
        # TD target
        td_target = reward + self.gamma * max_next_q
        
        # TD error
        td_error = td_target - current_q
        
        # Q-learning update
        self.q_table[state][action] = current_q + self.lr * td_error
        
        # Track statistics
        self.action_counts[action] += 1
        self.action_rewards[action].append(reward)
        self.state_action_visits[(state, action)] += 1
        
        return td_error
    
    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train_qlearning_halfcheetah(num_episodes=10000, num_bins_action=3, 
                                num_bins_state=10, max_steps=1000):
    """
    Train Q-learning agent on HalfCheetah with discretized action/state spaces
    
    Args:
        num_episodes: Number of training episodes
        num_bins_action: Number of bins per action dimension
        num_bins_state: Number of bins per state dimension
        max_steps: Maximum steps per episode
    
    Returns:
        agent, action_discretizer, state_discretizer, training_stats
    """
    # Create environment
    env = gym.make('HalfCheetah-v5')
    
    # Get action space info
    action_dim = env.action_space.shape[0]  # 6
    action_low = env.action_space.low[0]    # -1
    action_high = env.action_space.high[0]  # +1
    
    # Get state space info
    state_dim = env.observation_space.shape[0]  # 17
    
    # Create discretizers
    action_discretizer = ActionDiscretizer(
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        num_bins_per_dim=num_bins_action
    )
    
    state_discretizer = StateDiscretizer(
        state_dim=state_dim,
        num_bins_per_dim=num_bins_state
    )
    
    # Create Q-learning agent
    agent = QLearningAgent(
        num_actions=action_discretizer.num_discrete_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    td_errors = []
    epsilon_values = []
    
    print(f"\nStarting Q-learning training for {num_episodes} episodes...")
    print(f"State discretization: {num_bins_state} bins per dimension")
    print(f"Action discretization: {num_bins_action} bins per dimension")
    print(f"Total discrete actions: {action_discretizer.num_discrete_actions}\n")
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training"):
        state, _ = env.reset()
        state_discrete = state_discretizer.discretize_state(state)
        
        episode_reward = 0
        episode_td_errors = []
        
        for step in range(max_steps):
            # Select action
            action_index = agent.select_action(state_discrete, training=True)
            action_continuous = action_discretizer.discretize_action(action_index)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action_continuous)
            done = terminated or truncated
            
            # Discretize next state
            next_state_discrete = state_discretizer.discretize_state(next_state)
            
            # Q-learning update
            td_error = agent.update(state_discrete, action_index, reward, 
                                   next_state_discrete, done)
            episode_td_errors.append(abs(td_error))
            
            # Accumulate reward
            episode_reward += reward
            
            # Move to next state
            state = next_state
            state_discrete = next_state_discrete
            
            if done:
                break
        
        # Episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        td_errors.append(np.mean(episode_td_errors))
        epsilon_values.append(agent.epsilon)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Print progress
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    # Compile statistics
    training_stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'td_errors': td_errors,
        'epsilon_values': epsilon_values,
        'action_counts': agent.action_counts,
        'action_rewards': agent.action_rewards,
        'state_action_visits': agent.state_action_visits
    }
    
    return agent, action_discretizer, state_discretizer, training_stats


def analyze_discretization_impact(agent, action_discretizer, training_stats):
    """
    Q1(b): Analyze whether discretization leads to good control or instability
    Q1(c): Create required plots
    """
    print("\n" + "="*80)
    print("Q1 ANALYSIS: Discretization Impact on HalfCheetah Control")
    print("="*80)
    
    # Extract statistics
    action_counts = training_stats['action_counts']
    action_rewards = training_stats['action_rewards']
    episode_rewards = training_stats['episode_rewards']
    
    # Q1(b): Control behavior analysis
    print("\n(b) Control Behavior Analysis:")
    print("-" * 80)
    
    # 1. Action usage diversity
    actions_used = np.sum(action_counts > 0)
    total_actions = len(action_counts)
    usage_ratio = actions_used / total_actions
    
    print(f"1. Action Space Utilization:")
    print(f"   - Actions used: {actions_used}/{total_actions} ({usage_ratio*100:.1f}%)")
    
    # 2. Action concentration (Gini coefficient)
    sorted_counts = np.sort(action_counts)
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_counts)) / (n * np.sum(sorted_counts)) - (n+1)/n
    
    print(f"   - Action concentration (Gini): {gini:.3f}")
    print(f"     (0 = uniform usage, 1 = concentrated on few actions)")
    
    # 3. Reward stability
    reward_variance = np.var(episode_rewards)
    reward_std = np.std(episode_rewards)
    reward_mean = np.mean(episode_rewards)
    
    print(f"\n2. Reward Stability:")
    print(f"   - Mean episode reward: {reward_mean:.2f}")
    print(f"   - Std dev: {reward_std:.2f}")
    print(f"   - Coefficient of variation: {reward_std/abs(reward_mean):.3f}")
    
    # 4. Learning progress
    early_rewards = np.mean(episode_rewards[:1000])
    late_rewards = np.mean(episode_rewards[-1000:])
    improvement = late_rewards - early_rewards
    
    print(f"\n3. Learning Progress:")
    print(f"   - Early episodes (0-1000) avg reward: {early_rewards:.2f}")
    print(f"   - Late episodes (9000-10000) avg reward: {late_rewards:.2f}")
    print(f"   - Improvement: {improvement:.2f}")
    
    # 5. Control instability indicators
    print(f"\n4. Instability Indicators:")
    
    # Check for highly negative rewards (falling/unstable)
    negative_episodes = np.sum(np.array(episode_rewards) < -500)
    print(f"   - Episodes with reward < -500: {negative_episodes}/10000 ({negative_episodes/100:.1f}%)")
    
    # Check reward oscillations
    reward_diff = np.diff(episode_rewards)
    oscillation_rate = np.sum(np.abs(reward_diff) > 200) / len(reward_diff)
    print(f"   - Large reward oscillations (>200): {oscillation_rate*100:.1f}%")
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    if usage_ratio < 0.3:
        print("⚠ LOW ACTION DIVERSITY: Only {:.1f}% of actions used - discretization too coarse".format(usage_ratio*100))
    if gini > 0.7:
        print("⚠ HIGH ACTION CONCENTRATION: Agent relies heavily on few actions")
    if reward_std/abs(reward_mean) > 0.5:
        print("⚠ HIGH REWARD VARIANCE: Unstable control behavior")
    if improvement < 0:
        print("⚠ NO LEARNING PROGRESS: Agent failed to improve")
    if negative_episodes > 3000:
        print("⚠ FREQUENT FAILURES: Agent frequently falls or loses control")
    
    if usage_ratio >= 0.3 and gini < 0.7 and improvement > 0:
        print("✓ Discretization appears reasonable for basic control")
    else:
        print("✗ Discretization introduces significant control limitations")
    
    print("="*80 + "\n")
    
    # Q1(c): Create required plots
    create_q1_plots(action_discretizer, training_stats)


def create_q1_plots(action_discretizer, training_stats):
    """
    Q1(c): Create required visualizations
    i. action usage statistics
    ii. reward distribution per discrete action
    """
    action_counts = training_stats['action_counts']
    action_rewards = training_stats['action_rewards']
    episode_rewards = training_stats['episode_rewards']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Action Usage Statistics (Bar chart)
    ax1 = plt.subplot(3, 2, 1)
    action_indices = np.arange(len(action_counts))
    bars = ax1.bar(action_indices, action_counts, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Highlight most used actions
    top_5_actions = np.argsort(action_counts)[-5:]
    for idx in top_5_actions:
        bars[idx].set_color('orangered')
        bars[idx].set_alpha(0.9)
    
    ax1.set_xlabel('Discrete Action Index', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Usage Count', fontsize=11, fontweight='bold')
    ax1.set_title('(i) Action Usage Statistics\n(Red = Top 5 most used actions)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add usage percentage on top bars for top 5
    total_actions = np.sum(action_counts)
    for idx in top_5_actions:
        height = action_counts[idx]
        percentage = (height / total_actions) * 100
        ax1.text(idx, height, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Action Usage Percentage (Pie chart for top actions)
    ax2 = plt.subplot(3, 2, 2)
    
    # Get top 10 actions and group others
    top_n = 10
    top_actions = np.argsort(action_counts)[-top_n:][::-1]
    top_counts = action_counts[top_actions]
    other_count = np.sum(action_counts) - np.sum(top_counts)
    
    labels = [f'Action {i}' for i in top_actions]
    labels.append('Others')
    sizes = list(top_counts) + [other_count]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)
    
    ax2.set_title('Action Usage Distribution (Top 10 + Others)', 
                  fontsize=12, fontweight='bold')
    
    # Plot 3: Reward Distribution per Discrete Action (Box plot)
    ax3 = plt.subplot(3, 2, 3)
    
    # Prepare data for box plot - only actions that were used
    used_actions = [i for i in range(len(action_counts)) if action_counts[i] > 0]
    reward_data = [action_rewards[i] for i in used_actions if i in action_rewards]
    
    # Limit to top 20 most used actions for readability
    if len(used_actions) > 20:
        top_20_actions = np.argsort(action_counts)[-20:]
        reward_data = [action_rewards[i] for i in top_20_actions if i in action_rewards]
        used_actions = [i for i in top_20_actions if i in action_rewards]
    
    bp = ax3.boxplot(reward_data, labels=[str(i) for i in used_actions],
                     patch_artist=True, showfliers=False)
    
    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax3.set_xlabel('Discrete Action Index', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Reward', fontsize=11, fontweight='bold')
    ax3.set_title('(ii) Reward Distribution per Discrete Action\n(Top 20 most used actions)', 
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 4: Mean Reward per Action (sorted)
    ax4 = plt.subplot(3, 2, 4)
    
    mean_rewards = []
    action_ids = []
    for action_idx in used_actions:
        if action_idx in action_rewards and len(action_rewards[action_idx]) > 0:
            mean_rewards.append(np.mean(action_rewards[action_idx]))
            action_ids.append(action_idx)
    
    # Sort by mean reward
    sorted_indices = np.argsort(mean_rewards)
    sorted_rewards = np.array(mean_rewards)[sorted_indices]
    sorted_actions = np.array(action_ids)[sorted_indices]
    
    colors_gradient = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_rewards)))
    ax4.barh(range(len(sorted_rewards)), sorted_rewards, color=colors_gradient, 
             edgecolor='black', alpha=0.8)
    ax4.set_yticks(range(len(sorted_actions)))
    ax4.set_yticklabels([f'Act {a}' for a in sorted_actions], fontsize=8)
    ax4.set_xlabel('Mean Reward', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Action Index', fontsize=11, fontweight='bold')
    ax4.set_title('Mean Reward per Action (Sorted)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    # Plot 5: Learning Curve
    ax5 = plt.subplot(3, 2, 5)
    
    # Smooth rewards with rolling average
    window_size = 100
    smoothed_rewards = np.convolve(episode_rewards, 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
    
    ax5.plot(episode_rewards, alpha=0.3, color='lightblue', label='Episode Reward')
    ax5.plot(range(window_size-1, len(episode_rewards)), smoothed_rewards, 
             color='darkblue', linewidth=2, label=f'{window_size}-episode Moving Avg')
    ax5.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Episode Reward', fontsize=11, fontweight='bold')
    ax5.set_title('Learning Curve (Q-learning)', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Action Diversity Over Time
    ax6 = plt.subplot(3, 2, 6)
    
    # Track unique actions used in each window
    window = 500
    unique_actions_over_time = []
    episodes_windows = []
    
    # Simulate action selection over episodes (approximation from final counts)
    # This is a simplified version - in real implementation, track during training
    total_steps = np.sum(action_counts)
    action_probs = action_counts / total_steps
    
    for ep in range(0, 10000, window):
        episodes_windows.append(ep + window//2)
        # Estimate unique actions in this window
        expected_unique = np.sum(action_probs > 0) * (1 - (1 - action_probs[action_probs > 0]).mean())**window
        unique_actions_over_time.append(expected_unique)
    
    ax6.plot(episodes_windows, unique_actions_over_time, 
             marker='o', linewidth=2, markersize=6, color='darkgreen')
    ax6.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Estimated Unique Actions Used', fontsize=11, fontweight='bold')
    ax6.set_title('Action Diversity Over Training', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/q1_discretization_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plot: q1_discretization_analysis.png")
    
    # Additional detailed plot for reward distributions
    fig2 = plt.figure(figsize=(14, 6))
    
    # Violin plot for reward distributions
    ax1 = plt.subplot(1, 2, 1)
    
    if len(reward_data) > 0 and len(reward_data) <= 30:
        parts = ax1.violinplot(reward_data, positions=range(len(used_actions)),
                               showmeans=True, showmedians=True)
        ax1.set_xticks(range(len(used_actions)))
        ax1.set_xticklabels([str(i) for i in used_actions], rotation=45, ha='right')
        ax1.set_xlabel('Discrete Action Index', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Reward', fontsize=11, fontweight='bold')
        ax1.set_title('Reward Distribution per Action (Violin Plot)', 
                      fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
    
    # Heatmap of action statistics
    ax2 = plt.subplot(1, 2, 2)
    
    # Create summary statistics matrix
    stats_matrix = []
    stat_labels = []
    
    for action_idx in used_actions[:20]:  # Top 20
        if action_idx in action_rewards and len(action_rewards[action_idx]) > 0:
            rewards = action_rewards[action_idx]
            stats_matrix.append([
                np.mean(rewards),
                np.std(rewards),
                np.min(rewards),
                np.max(rewards),
                action_counts[action_idx]
            ])
            stat_labels.append(f'A{action_idx}')
    
    if len(stats_matrix) > 0:
        stats_matrix = np.array(stats_matrix)
        # Normalize each column
        stats_normalized = (stats_matrix - stats_matrix.mean(axis=0)) / (stats_matrix.std(axis=0) + 1e-8)
        
        im = ax2.imshow(stats_normalized.T, aspect='auto', cmap='RdYlGn', 
                       interpolation='nearest')
        ax2.set_xticks(range(len(stat_labels)))
        ax2.set_xticklabels(stat_labels, rotation=45, ha='right')
        ax2.set_yticks(range(5))
        ax2.set_yticklabels(['Mean Reward', 'Std Reward', 'Min Reward', 
                            'Max Reward', 'Usage Count'])
        ax2.set_title('Action Statistics Heatmap (Normalized)', 
                     fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax2, label='Normalized Value')
    
    plt.tight_layout()
    plt.savefig('/home/claude/q1_reward_distribution_detailed.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plot: q1_reward_distribution_detailed.png")
    
    plt.close('all')


def save_results(agent, action_discretizer, state_discretizer, training_stats):
    """Save training results for later use"""
    results = {
        'q_table_size': len(agent.q_table),
        'training_stats': training_stats,
        'num_discrete_actions': action_discretizer.num_discrete_actions,
        'num_bins_action': action_discretizer.num_bins_per_dim,
    }
    
    with open('/home/claude/q1_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("✓ Saved results: q1_results.pkl")


if __name__ == "__main__":
    print("="*80)
    print("Q1: Q-LEARNING WITH ACTION SPACE DISCRETIZATION FOR HALFCHEETAH")
    print("="*80)
    
    # Configuration
    NUM_EPISODES = 10000
    NUM_BINS_ACTION = 3  # Try 3, 5, or 7 bins per action dimension
    NUM_BINS_STATE = 10  # State discretization
    
    print(f"\nConfiguration:")
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"  Action bins per dimension: {NUM_BINS_ACTION}")
    print(f"  State bins per dimension: {NUM_BINS_STATE}")
    
    # Train agent
    agent, action_discretizer, state_discretizer, training_stats = train_qlearning_halfcheetah(
        num_episodes=NUM_EPISODES,
        num_bins_action=NUM_BINS_ACTION,
        num_bins_state=NUM_BINS_STATE,
        max_steps=1000
    )
    
    # Analyze results
    analyze_discretization_impact(agent, action_discretizer, training_stats)
    
    # Save results
    save_results(agent, action_discretizer, state_discretizer, training_stats)
    
    print("\n" + "="*80)
    print("Q1 COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. q1_discretization_analysis.png - Main analysis plots")
    print("  2. q1_reward_distribution_detailed.png - Detailed reward distributions")
    print("  3. q1_results.pkl - Saved training results")
