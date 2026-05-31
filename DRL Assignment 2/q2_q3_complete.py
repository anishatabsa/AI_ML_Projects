"""
Q2 & Q3: Complete Analysis and Algorithmic Modification
Based on Q1 results showing high action concentration and reward instability

Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import seaborn as sns
from tqdm import tqdm
import pickle

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EnhancedQLearningAgent:
    """Q-learning agent with detailed tracking for Q2 analysis"""
    
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 use_adaptive_lr=False, use_boltzmann=False):
        
        self.num_actions = num_actions
        self.initial_lr = learning_rate
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q3 modifications
        self.use_adaptive_lr = use_adaptive_lr
        self.use_boltzmann = use_boltzmann
        self.temperature = 10.0  # For Boltzmann exploration
        self.temp_decay = 0.995
        
        # Q-table
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        
        # Q2 Tracking: Component-specific metrics
        self.tracking = {
            # 1. Learning rate interaction
            'td_errors': [],
            'td_error_magnitude': [],
            'q_changes': [],
            'q_change_variance': [],
            'learning_rates': [],
            
            # 2. Max-operator
            'max_q_values': [],
            'max_actions': [],
            'q_value_spread': [],  # max - min over actions
            'second_best_gap': [],  # max - second_best
            'action_switch_rate': [],
            
            # 3. State-action visitation
            'state_action_counts': defaultdict(int),
            'unique_sa_per_episode': [],
            'coverage_ratio': [],
            'gini_coefficient': [],
            
            # 4. Reward propagation
            'immediate_rewards': [],
            'bootstrapped_values': [],
            'td_target_values': [],
            'value_backup_depth': []
        }
        
        self.step_count = 0
        self.episode_count = 0
        self.episode_actions = []
        self.previous_max_action = None
        
        # For adaptive learning rate (Q3)
        self.state_action_updates = defaultdict(int)
        
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy or Boltzmann"""
        q_values = self.q_table[state]
        
        if not training:
            return np.argmax(q_values)
        
        if self.use_boltzmann:
            # Boltzmann exploration (Q3 modification option)
            exp_q = np.exp((q_values - np.max(q_values)) / self.temperature)
            probs = exp_q / np.sum(exp_q)
            action = np.random.choice(self.num_actions, p=probs)
        else:
            # Epsilon-greedy
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.num_actions)
            else:
                action = np.argmax(q_values)
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update with comprehensive tracking"""
        
        # Current Q-value
        current_q = self.q_table[state][action]
        
        # Next state Q-values
        next_q_values = self.q_table[next_state]
        
        # Max operation
        if done:
            max_next_q = 0
            max_action = -1
        else:
            max_action = np.argmax(next_q_values)
            max_next_q = next_q_values[max_action]
        
        # TD target and error
        bootstrapped = self.gamma * max_next_q
        td_target = reward + bootstrapped
        td_error = td_target - current_q
        
        # Adaptive learning rate (Q3 option)
        if self.use_adaptive_lr:
            # Decay learning rate based on visit count
            visit_count = self.state_action_updates[(state, action)]
            self.lr = self.initial_lr / (1 + 0.01 * visit_count)
        
        # Q-learning update
        q_change = self.lr * td_error
        self.q_table[state][action] = current_q + q_change
        
        # === Q2 TRACKING ===
        
        # 1. Learning rate interaction
        self.tracking['td_errors'].append(td_error)
        self.tracking['td_error_magnitude'].append(abs(td_error))
        self.tracking['q_changes'].append(q_change)
        self.tracking['learning_rates'].append(self.lr)
        
        # Calculate Q-change variance over recent updates
        if len(self.tracking['q_changes']) >= 100:
            recent_variance = np.var(self.tracking['q_changes'][-100:])
            self.tracking['q_change_variance'].append(recent_variance)
        
        # 2. Max-operator tracking
        if not done:
            self.tracking['max_q_values'].append(max_next_q)
            self.tracking['max_actions'].append(max_action)
            
            # Q-value spread
            spread = np.max(next_q_values) - np.min(next_q_values)
            self.tracking['q_value_spread'].append(spread)
            
            # Second-best gap
            sorted_q = np.sort(next_q_values)
            if len(sorted_q) > 1:
                gap = sorted_q[-1] - sorted_q[-2]
                self.tracking['second_best_gap'].append(gap)
            
            # Action switching
            if self.previous_max_action is not None:
                switch = 1 if max_action != self.previous_max_action else 0
                self.tracking['action_switch_rate'].append(switch)
            self.previous_max_action = max_action
        
        # 3. State-action visitation
        self.tracking['state_action_counts'][(state, action)] += 1
        self.state_action_updates[(state, action)] += 1
        
        # 4. Reward propagation
        self.tracking['immediate_rewards'].append(reward)
        self.tracking['bootstrapped_values'].append(bootstrapped)
        self.tracking['td_target_values'].append(td_target)
        
        self.step_count += 1
        self.episode_actions.append(action)
        
        return td_error
    
    def end_episode(self):
        """Track episode-level statistics"""
        
        # Unique state-actions in episode
        unique_sa = len(set([(s, a) for s, a in self.tracking['state_action_counts'].keys()]))
        self.tracking['unique_sa_per_episode'].append(len(self.episode_actions))
        
        # Coverage ratio
        total_possible = self.num_actions  # Simplified
        coverage = unique_sa / (self.episode_count + 1)
        self.tracking['coverage_ratio'].append(coverage)
        
        # Gini coefficient for action distribution
        if len(self.episode_actions) > 0:
            action_counts = np.bincount(self.episode_actions, minlength=self.num_actions)
            if np.sum(action_counts) > 0:
                sorted_counts = np.sort(action_counts)
                n = len(sorted_counts)
                cumsum = np.cumsum(sorted_counts)
                gini = (2 * np.sum((np.arange(1, n+1)) * sorted_counts)) / (n * np.sum(sorted_counts)) - (n+1)/n
                self.tracking['gini_coefficient'].append(gini)
        
        self.episode_actions = []
        self.episode_count += 1
        
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if self.use_boltzmann:
            self.temperature = max(0.1, self.temperature * self.temp_decay)


class ActionDiscretizer:
    """Discretizes continuous action space"""
    def __init__(self, action_dim, action_low, action_high, num_bins_per_dim=3):
        self.action_dim = action_dim
        self.num_bins_per_dim = num_bins_per_dim
        self.discrete_values = np.linspace(action_low, action_high, num_bins_per_dim)
        self.num_discrete_actions = num_bins_per_dim ** action_dim
        
        # Generate all action combinations
        grids = np.meshgrid(*[self.discrete_values for _ in range(self.action_dim)])
        self.discrete_actions = np.stack([grid.flatten() for grid in grids], axis=1)
    
    def discretize_action(self, action_index):
        return self.discrete_actions[action_index]


class StateDiscretizer:
    """Discretizes continuous state space"""
    def __init__(self, state_dim, num_bins_per_dim=10):
        self.state_dim = state_dim
        self.num_bins_per_dim = num_bins_per_dim
        self.state_low = -np.ones(state_dim) * 10
        self.state_high = np.ones(state_dim) * 10
        
    def discretize_state(self, state):
        state_clipped = np.clip(state, self.state_low, self.state_high)
        state_normalized = (state_clipped - self.state_low) / (self.state_high - self.state_low + 1e-8)
        state_discrete = (state_normalized * (self.num_bins_per_dim - 1)).astype(int)
        state_discrete = np.clip(state_discrete, 0, self.num_bins_per_dim - 1)
        return tuple(state_discrete)


def train_qlearning(num_episodes=10000, num_bins_action=3, num_bins_state=10,
                    use_adaptive_lr=False, use_boltzmann=False, agent_name="Baseline"):
    """Train Q-learning agent with tracking"""
    
    env = gym.make('HalfCheetah-v5')
    
    action_discretizer = ActionDiscretizer(
        action_dim=env.action_space.shape[0],
        action_low=env.action_space.low[0],
        action_high=env.action_space.high[0],
        num_bins_per_dim=num_bins_action
    )
    
    state_discretizer = StateDiscretizer(
        state_dim=env.observation_space.shape[0],
        num_bins_per_dim=num_bins_state
    )
    
    agent = EnhancedQLearningAgent(
        num_actions=action_discretizer.num_discrete_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        use_adaptive_lr=use_adaptive_lr,
        use_boltzmann=use_boltzmann
    )
    
    episode_rewards = []
    episode_lengths = []
    
    print(f"\nTraining {agent_name}...")
    print(f"  Adaptive LR: {use_adaptive_lr}, Boltzmann: {use_boltzmann}")
    print(f"  Total discrete actions: {action_discretizer.num_discrete_actions}\n")
    
    for episode in tqdm(range(num_episodes), desc=f"Training {agent_name}"):
        state, _ = env.reset()
        state_discrete = state_discretizer.discretize_state(state)
        
        episode_reward = 0
        steps = 0
        
        for step in range(1000):
            action_index = agent.select_action(state_discrete, training=True)
            action_continuous = action_discretizer.discretize_action(action_index)
            
            next_state, reward, terminated, truncated, info = env.step(action_continuous)
            done = terminated or truncated
            
            next_state_discrete = state_discretizer.discretize_state(next_state)
            
            agent.update(state_discrete, action_index, reward, next_state_discrete, done)
            
            episode_reward += reward
            steps += 1
            state_discrete = next_state_discrete
            
            if done:
                break
        
        agent.end_episode()
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if (episode + 1) % 2000 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"  Episode {episode+1}: Avg Reward = {avg_reward:.2f}, ε = {agent.epsilon:.3f}")
    
    env.close()
    
    return agent, episode_rewards, episode_lengths


def analyze_q2(agent, episode_rewards, agent_name="Baseline"):
    """Q2: Comprehensive analysis of Q-learning update components"""
    
    print("\n" + "="*80)
    print(f"Q2 ANALYSIS: {agent_name}")
    print("="*80)
    
    fig = plt.figure(figsize=(20, 16))
    
    # ========================================================================
    # 1. LEARNING RATE INTERACTION
    # ========================================================================
    print("\n1. LEARNING RATE INTERACTION")
    print("-" * 80)
    
    # Plot 1a: TD Error Magnitude Over Time
    ax1 = plt.subplot(4, 4, 1)
    if len(agent.tracking['td_error_magnitude']) > 1000:
        # Smooth the data
        window = 500
        smoothed = np.convolve(agent.tracking['td_error_magnitude'], 
                              np.ones(window)/window, mode='valid')
        ax1.plot(smoothed, color='red', linewidth=2)
        ax1.set_xlabel('Step', fontsize=9, fontweight='bold')
        ax1.set_ylabel('|TD Error|', fontsize=9, fontweight='bold')
        ax1.set_title('1a. TD Error Magnitude (Smoothed)', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        early_td = np.mean(agent.tracking['td_error_magnitude'][:10000])
        late_td = np.mean(agent.tracking['td_error_magnitude'][-10000:])
        print(f"  Early |TD error|: {early_td:.2f}")
        print(f"  Late |TD error|: {late_td:.2f}")
        print(f"  Reduction: {(early_td - late_td)/early_td * 100:.1f}%")
    
    # Plot 1b: Q-Value Change Variance
    ax2 = plt.subplot(4, 4, 2)
    if len(agent.tracking['q_change_variance']) > 0:
        ax2.plot(agent.tracking['q_change_variance'], color='purple', linewidth=1.5)
        ax2.set_xlabel('Step', fontsize=9, fontweight='bold')
        ax2.set_ylabel('Var(ΔQ)', fontsize=9, fontweight='bold')
        ax2.set_title('1b. Q-Value Change Variance', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        print(f"  Q-change variance: {np.mean(agent.tracking['q_change_variance']):.2e}")
    
    # Plot 1c: Learning Rate Evolution (if adaptive)
    ax3 = plt.subplot(4, 4, 3)
    if agent.use_adaptive_lr and len(agent.tracking['learning_rates']) > 0:
        window = 500
        smoothed_lr = np.convolve(agent.tracking['learning_rates'], 
                                  np.ones(window)/window, mode='valid')
        ax3.plot(smoothed_lr, color='darkgreen', linewidth=2)
        ax3.set_xlabel('Step', fontsize=9, fontweight='bold')
        ax3.set_ylabel('Learning Rate α', fontsize=9, fontweight='bold')
        ax3.set_title('1c. Adaptive Learning Rate', fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        print(f"  Final avg LR: {np.mean(agent.tracking['learning_rates'][-1000:]):.4f}")
    else:
        ax3.axhline(y=agent.initial_lr, color='darkgreen', linewidth=2)
        ax3.set_xlabel('Step', fontsize=9, fontweight='bold')
        ax3.set_ylabel('Learning Rate α', fontsize=9, fontweight='bold')
        ax3.set_title('1c. Fixed Learning Rate', fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 0.15])
    
    # Plot 1d: TD Error Distribution
    ax4 = plt.subplot(4, 4, 4)
    td_errors = agent.tracking['td_errors']
    if len(td_errors) > 0:
        # Split into early and late
        split = len(td_errors) // 2
        early = td_errors[:split]
        late = td_errors[split:]
        
        ax4.hist(early, bins=50, alpha=0.5, color='red', label='Early', 
                density=True, range=(-200, 200))
        ax4.hist(late, bins=50, alpha=0.5, color='green', label='Late', 
                density=True, range=(-200, 200))
        ax4.set_xlabel('TD Error', fontsize=9, fontweight='bold')
        ax4.set_ylabel('Density', fontsize=9, fontweight='bold')
        ax4.set_title('1d. TD Error Distribution', fontsize=10, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        print(f"  Early TD std: {np.std(early):.2f}")
        print(f"  Late TD std: {np.std(late):.2f}")
    
    # ========================================================================
    # 2. MAX-OPERATOR OVER DISCRETIZED ACTIONS
    # ========================================================================
    print("\n2. MAX-OPERATOR ANALYSIS")
    print("-" * 80)
    
    # Plot 2a: Max Q-Value Evolution
    ax5 = plt.subplot(4, 4, 5)
    if len(agent.tracking['max_q_values']) > 1000:
        window = 500
        smoothed = np.convolve(agent.tracking['max_q_values'], 
                              np.ones(window)/window, mode='valid')
        ax5.plot(smoothed, color='darkblue', linewidth=2)
        ax5.set_xlabel('Step', fontsize=9, fontweight='bold')
        ax5.set_ylabel('max Q(s\', a\')', fontsize=9, fontweight='bold')
        ax5.set_title('2a. Max Q-Value Evolution', fontsize=10, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        final_max_q = np.mean(agent.tracking['max_q_values'][-1000:])
        print(f"  Final avg max Q-value: {final_max_q:.2f}")
    
    # Plot 2b: Q-Value Spread
    ax6 = plt.subplot(4, 4, 6)
    if len(agent.tracking['q_value_spread']) > 1000:
        window = 500
        smoothed = np.convolve(agent.tracking['q_value_spread'], 
                              np.ones(window)/window, mode='valid')
        ax6.plot(smoothed, color='darkred', linewidth=2)
        ax6.set_xlabel('Step', fontsize=9, fontweight='bold')
        ax6.set_ylabel('max Q - min Q', fontsize=9, fontweight='bold')
        ax6.set_title('2b. Q-Value Spread', fontsize=10, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        final_spread = np.mean(agent.tracking['q_value_spread'][-1000:])
        print(f"  Final Q-value spread: {final_spread:.2f}")
    
    # Plot 2c: Action Switching Rate
    ax7 = plt.subplot(4, 4, 7)
    if len(agent.tracking['action_switch_rate']) > 100:
        window = 500
        smoothed = np.convolve(agent.tracking['action_switch_rate'], 
                              np.ones(window)/window, mode='valid')
        ax7.plot(smoothed, color='orange', linewidth=2)
        ax7.set_xlabel('Step', fontsize=9, fontweight='bold')
        ax7.set_ylabel('Switch Rate', fontsize=9, fontweight='bold')
        ax7.set_title('2c. Max Action Switching Rate', fontsize=10, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        early_switch = np.mean(agent.tracking['action_switch_rate'][:10000])
        late_switch = np.mean(agent.tracking['action_switch_rate'][-10000:])
        print(f"  Early switch rate: {early_switch:.3f}")
        print(f"  Late switch rate: {late_switch:.3f}")
    
    # Plot 2d: Second-Best Gap
    ax8 = plt.subplot(4, 4, 8)
    if len(agent.tracking['second_best_gap']) > 1000:
        window = 500
        smoothed = np.convolve(agent.tracking['second_best_gap'], 
                              np.ones(window)/window, mode='valid')
        ax8.plot(smoothed, color='teal', linewidth=2)
        ax8.set_xlabel('Step', fontsize=9, fontweight='bold')
        ax8.set_ylabel('Q_best - Q_2nd', fontsize=9, fontweight='bold')
        ax8.set_title('2d. Gap to Second-Best Action', fontsize=10, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        final_gap = np.mean(agent.tracking['second_best_gap'][-1000:])
        print(f"  Final 1st-2nd gap: {final_gap:.2f}")
    
    # ========================================================================
    # 3. STATE-ACTION VISITATION IMBALANCE
    # ========================================================================
    print("\n3. STATE-ACTION VISITATION IMBALANCE")
    print("-" * 80)
    
    # Plot 3a: Visitation Distribution
    ax9 = plt.subplot(4, 4, 9)
    visit_counts = np.array(list(agent.tracking['state_action_counts'].values()))
    if len(visit_counts) > 0:
        ax9.hist(np.log10(visit_counts + 1), bins=50, color='purple', 
                alpha=0.7, edgecolor='black')
        ax9.set_xlabel('log10(Visits)', fontsize=9, fontweight='bold')
        ax9.set_ylabel('Count', fontsize=9, fontweight='bold')
        ax9.set_title('3a. State-Action Visit Distribution', fontsize=10, fontweight='bold')
        ax9.grid(True, alpha=0.3)
        
        total_sa = len(visit_counts)
        print(f"  Total (s,a) pairs visited: {total_sa}")
        print(f"  Visit range: [{np.min(visit_counts)}, {np.max(visit_counts)}]")
        print(f"  Mean visits: {np.mean(visit_counts):.2f}")
    
    # Plot 3b: Gini Coefficient
    ax10 = plt.subplot(4, 4, 10)
    if len(agent.tracking['gini_coefficient']) > 0:
        ax10.plot(agent.tracking['gini_coefficient'], color='darkred', linewidth=1.5)
        ax10.set_xlabel('Episode', fontsize=9, fontweight='bold')
        ax10.set_ylabel('Gini Coefficient', fontsize=9, fontweight='bold')
        ax10.set_title('3b. Action Concentration (Gini)', fontsize=10, fontweight='bold')
        ax10.grid(True, alpha=0.3)
        
        final_gini = np.mean(agent.tracking['gini_coefficient'][-100:])
        print(f"  Final Gini coefficient: {final_gini:.3f}")
    
    # Plot 3c: Coverage Over Time
    ax11 = plt.subplot(4, 4, 11)
    if len(agent.tracking['unique_sa_per_episode']) > 0:
        window = 100
        smoothed = np.convolve(agent.tracking['unique_sa_per_episode'], 
                              np.ones(window)/window, mode='valid')
        ax11.plot(smoothed, color='green', linewidth=2)
        ax11.set_xlabel('Episode', fontsize=9, fontweight='bold')
        ax11.set_ylabel('Unique (s,a) per Ep', fontsize=9, fontweight='bold')
        ax11.set_title('3c. State-Action Coverage', fontsize=10, fontweight='bold')
        ax11.grid(True, alpha=0.3)
    
    # Plot 3d: Power Law (Log-Log)
    ax12 = plt.subplot(4, 4, 12)
    if len(visit_counts) > 0:
        sorted_visits = np.sort(visit_counts)[::-1]
        ax12.loglog(range(1, len(sorted_visits)+1), sorted_visits, 
                   color='darkviolet', linewidth=2)
        ax12.set_xlabel('Rank (log)', fontsize=9, fontweight='bold')
        ax12.set_ylabel('Visits (log)', fontsize=9, fontweight='bold')
        ax12.set_title('3d. Visit Power Law', fontsize=10, fontweight='bold')
        ax12.grid(True, alpha=0.3, which='both')
    
    # ========================================================================
    # 4. DELAYED REWARD PROPAGATION
    # ========================================================================
    print("\n4. DELAYED REWARD PROPAGATION")
    print("-" * 80)
    
    # Plot 4a: Immediate vs Bootstrapped
    ax13 = plt.subplot(4, 4, 13)
    if len(agent.tracking['immediate_rewards']) > 1000:
        # Compute cumulative sums in windows
        window = 1000
        imm_windows = []
        boot_windows = []
        
        for i in range(0, len(agent.tracking['immediate_rewards']) - window, window):
            imm_windows.append(np.mean(agent.tracking['immediate_rewards'][i:i+window]))
            boot_windows.append(np.mean(agent.tracking['bootstrapped_values'][i:i+window]))
        
        ax13.plot(imm_windows, color='red', linewidth=2, label='Immediate (r)')
        ax13.plot(boot_windows, color='blue', linewidth=2, label='Bootstrapped (γ max Q)')
        ax13.set_xlabel('Window (×1000 steps)', fontsize=9, fontweight='bold')
        ax13.set_ylabel('Mean Value', fontsize=9, fontweight='bold')
        ax13.set_title('4a. Reward Components', fontsize=10, fontweight='bold')
        ax13.legend()
        ax13.grid(True, alpha=0.3)
        
        early_imm = np.mean(agent.tracking['immediate_rewards'][:10000])
        late_imm = np.mean(agent.tracking['immediate_rewards'][-10000:])
        early_boot = np.mean(agent.tracking['bootstrapped_values'][:10000])
        late_boot = np.mean(agent.tracking['bootstrapped_values'][-10000:])
        
        print(f"  Early: r={early_imm:.2f}, γQ={early_boot:.2f}")
        print(f"  Late: r={late_imm:.2f}, γQ={late_boot:.2f}")
        print(f"  Bootstrap growth: {late_boot - early_boot:.2f}")
    
    # Plot 4b: TD Target Evolution
    ax14 = plt.subplot(4, 4, 14)
    if len(agent.tracking['td_target_values']) > 1000:
        window = 500
        smoothed = np.convolve(agent.tracking['td_target_values'], 
                              np.ones(window)/window, mode='valid')
        ax14.plot(smoothed, color='darkgreen', linewidth=2)
        ax14.set_xlabel('Step', fontsize=9, fontweight='bold')
        ax14.set_ylabel('TD Target', fontsize=9, fontweight='bold')
        ax14.set_title('4b. TD Target Evolution', fontsize=10, fontweight='bold')
        ax14.grid(True, alpha=0.3)
    
    # Plot 4c: Reward/Bootstrap Ratio
    ax15 = plt.subplot(4, 4, 15)
    if len(agent.tracking['immediate_rewards']) > 1000:
        ratios = []
        window = 1000
        for i in range(0, len(agent.tracking['immediate_rewards']) - window, window):
            imm = np.mean(agent.tracking['immediate_rewards'][i:i+window])
            boot = np.mean(agent.tracking['bootstrapped_values'][i:i+window])
            if abs(boot) > 1e-6:
                ratios.append(imm / boot)
        
        ax15.plot(ratios, color='orange', linewidth=2)
        ax15.axhline(y=1.0, color='red', linestyle='--', label='r = γQ')
        ax15.set_xlabel('Window (×1000 steps)', fontsize=9, fontweight='bold')
        ax15.set_ylabel('r / (γ max Q)', fontsize=9, fontweight='bold')
        ax15.set_title('4c. Immediate/Bootstrap Ratio', fontsize=10, fontweight='bold')
        ax15.legend()
        ax15.grid(True, alpha=0.3)
        
        print(f"  Final ratio: {ratios[-1] if len(ratios) > 0 else 0:.3f}")
    
    # Learning Curve
    ax16 = plt.subplot(4, 4, 16)
    window = 100
    smoothed_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    ax16.plot(episode_rewards, alpha=0.3, color='lightblue', label='Raw')
    ax16.plot(range(window-1, len(episode_rewards)), smoothed_rewards, 
             color='darkblue', linewidth=2, label='100-ep avg')
    ax16.set_xlabel('Episode', fontsize=9, fontweight='bold')
    ax16.set_ylabel('Reward', fontsize=9, fontweight='bold')
    ax16.set_title('Learning Curve', fontsize=10, fontweight='bold')
    ax16.legend()
    ax16.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'/home/claude/q2_analysis_{agent_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {filename}")
    
    plt.close('all')
    
    return agent.tracking


def compare_agents_q3(baseline_rewards, modified_rewards, modification_name):
    """Q3: Compare before/after learning curves"""
    
    print("\n" + "="*80)
    print(f"Q3: COMPARISON - {modification_name}")
    print("="*80)
    
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: Learning Curves Comparison
    ax1 = plt.subplot(1, 3, 1)
    window = 100
    
    smoothed_baseline = np.convolve(baseline_rewards, np.ones(window)/window, mode='valid')
    smoothed_modified = np.convolve(modified_rewards, np.ones(window)/window, mode='valid')
    
    ax1.plot(range(window-1, len(baseline_rewards)), smoothed_baseline, 
            color='red', linewidth=2, label='Baseline', alpha=0.8)
    ax1.plot(range(window-1, len(modified_rewards)), smoothed_modified, 
            color='green', linewidth=2, label=modification_name, alpha=0.8)
    
    ax1.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Episode Reward (100-ep avg)', fontsize=11, fontweight='bold')
    ax1.set_title('Learning Curves Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance Improvement
    ax2 = plt.subplot(1, 3, 2)
    
    # Calculate improvement per episode
    improvement = np.array(modified_rewards) - np.array(baseline_rewards)
    smoothed_improvement = np.convolve(improvement, np.ones(window)/window, mode='valid')
    
    ax2.plot(range(window-1, len(improvement)), smoothed_improvement, 
            color='purple', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.fill_between(range(window-1, len(improvement)), 0, smoothed_improvement, 
                     where=smoothed_improvement>0, alpha=0.3, color='green', 
                     label='Improvement')
    ax2.fill_between(range(window-1, len(improvement)), 0, smoothed_improvement, 
                     where=smoothed_improvement<0, alpha=0.3, color='red', 
                     label='Degradation')
    
    ax2.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Reward Difference', fontsize=11, fontweight='bold')
    ax2.set_title('Performance Δ (Modified - Baseline)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Statistical Comparison
    ax3 = plt.subplot(1, 3, 3)
    
    # Compare distributions for early, mid, late phases
    phases = ['Early\n(0-3k)', 'Mid\n(3k-6k)', 'Late\n(6k-10k)']
    baseline_phases = [
        baseline_rewards[:3000],
        baseline_rewards[3000:6000],
        baseline_rewards[6000:]
    ]
    modified_phases = [
        modified_rewards[:3000],
        modified_rewards[3000:6000],
        modified_rewards[6000:]
    ]
    
    baseline_means = [np.mean(p) for p in baseline_phases]
    modified_means = [np.mean(p) for p in modified_phases]
    baseline_stds = [np.std(p) for p in baseline_phases]
    modified_stds = [np.std(p) for p in modified_phases]
    
    x = np.arange(len(phases))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, baseline_means, width, label='Baseline', 
                   yerr=baseline_stds, capsize=5, color='red', alpha=0.7)
    bars2 = ax3.bar(x + width/2, modified_means, width, label=modification_name, 
                   yerr=modified_stds, capsize=5, color='green', alpha=0.7)
    
    ax3.set_xlabel('Training Phase', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Mean Reward ± Std', fontsize=11, fontweight='bold')
    ax3.set_title('Phase-wise Performance', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(phases)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('/home/claude/q3_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: q3_comparison.png")
    
    # Print statistics
    print("\nSTATISTICAL SUMMARY:")
    print("-" * 80)
    print(f"{'Metric':<30} {'Baseline':>15} {'Modified':>15} {'Change':>15}")
    print("-" * 80)
    
    baseline_mean = np.mean(baseline_rewards)
    modified_mean = np.mean(modified_rewards)
    print(f"{'Mean Reward':<30} {baseline_mean:>15.2f} {modified_mean:>15.2f} "
          f"{modified_mean - baseline_mean:>15.2f}")
    
    baseline_std = np.std(baseline_rewards)
    modified_std = np.std(modified_rewards)
    print(f"{'Std Reward':<30} {baseline_std:>15.2f} {modified_std:>15.2f} "
          f"{modified_std - baseline_std:>15.2f}")
    
    baseline_final = np.mean(baseline_rewards[-1000:])
    modified_final = np.mean(modified_rewards[-1000:])
    print(f"{'Final 1000 Episodes':<30} {baseline_final:>15.2f} {modified_final:>15.2f} "
          f"{modified_final - baseline_final:>15.2f}")
    
    improvement_pct = ((modified_mean - baseline_mean) / abs(baseline_mean)) * 100
    print(f"\n{'Improvement':<30} {improvement_pct:>15.2f}%")
    
    print("-" * 80)
    
    plt.close('all')


if __name__ == "__main__":
    print("="*80)
    print("Q2 & Q3: COMPLETE ANALYSIS")
    print("="*80)
    
    # Q2: Train baseline and analyze all components
    print("\n" + "="*80)
    print("Q2: ANALYZING Q-LEARNING UPDATE COMPONENTS")
    print("="*80)
    
    baseline_agent, baseline_rewards, baseline_lengths = train_qlearning(
        num_episodes=10000,
        num_bins_action=3,
        num_bins_state=10,
        use_adaptive_lr=False,
        use_boltzmann=False,
        agent_name="Baseline Q-Learning"
    )
    
    baseline_tracking = analyze_q2(baseline_agent, baseline_rewards, "Baseline")
    
    # Q3: Test different modifications
    print("\n" + "="*80)
    print("Q3: TESTING ALGORITHMIC MODIFICATIONS")
    print("="*80)
    
    print("\nTesting Modification: Adaptive Learning Rate Schedule")
    print("-" * 80)
    
    modified_agent, modified_rewards, modified_lengths = train_qlearning(
        num_episodes=10000,
        num_bins_action=3,
        num_bins_state=10,
        use_adaptive_lr=True,
        use_boltzmann=False,
        agent_name="Adaptive LR"
    )
    
    modified_tracking = analyze_q2(modified_agent, modified_rewards, "Adaptive LR")
    
    # Compare
    compare_agents_q3(baseline_rewards, modified_rewards, "Adaptive Learning Rate")
    
    # Save all results
    results = {
        'baseline_rewards': baseline_rewards,
        'modified_rewards': modified_rewards,
        'baseline_tracking': baseline_tracking,
        'modified_tracking': modified_tracking
    }
    
    with open('/home/claude/q2_q3_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n✓ Saved: q2_q3_results.pkl")
    print("\n" + "="*80)
    print("Q2 & Q3 COMPLETE!")
    print("="*80)
