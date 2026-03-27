# rl_agent.py
# Q-Learning Agent for Adaptive TCP Variant Selection
# Trains agent to dynamically switch TCP based on network conditions

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import json
import os
import random
from rl_environment import TCPEnvironment, TCP_VARIANTS

random.seed(42)
np.random.seed(42)

# ── Q-Learning Agent ───────────────────────────────────────────────
class QLearningAgent:
    """
    Q-Learning agent for TCP variant selection
    
    Q-table: discretized states × actions
    Updates: Q(s,a) = Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
    """

    def __init__(self,
                 n_actions    = 6,
                 learning_rate= 0.1,
                 discount     = 0.95,
                 epsilon      = 1.0,
                 epsilon_min  = 0.01,
                 epsilon_decay= 0.995,
                 n_bins       = 10):

        self.n_actions     = n_actions
        self.lr            = learning_rate
        self.gamma         = discount
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_bins        = n_bins

        # State discretization bins
        self.bins = [
            np.linspace(0,   15,   n_bins),   # throughput
            np.linspace(0,   300,  n_bins),   # rtt
            np.linspace(0,   0.3,  n_bins),   # loss
            np.linspace(0,   1,    n_bins),   # network_load
            np.linspace(0,   1,    n_bins),   # app_type
        ]

        # Q-table: shape = (bins^5, n_actions)
        self.q_table = np.zeros([n_bins]*5 + [n_actions])

        # Training history
        self.rewards_history  = []
        self.epsilon_history  = []
        self.tcp_switches     = []
        self.episode_rewards  = []

    def discretize(self, state):
        """Convert continuous state to discrete indices"""
        indices = []
        for i, val in enumerate(state):
            idx = np.digitize(val, self.bins[i]) - 1
            idx = np.clip(idx, 0, self.n_bins-1)
            indices.append(idx)
        return tuple(indices)

    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions-1)
        state_idx = self.discretize(state)
        return int(np.argmax(self.q_table[state_idx]))

    def update(self, state, action, reward, next_state):
        """Q-table update"""
        s  = self.discretize(state)
        s_ = self.discretize(next_state)

        current_q = self.q_table[s][action]
        max_next_q = np.max(self.q_table[s_])
        new_q = current_q + self.lr * (
            reward + self.gamma * max_next_q - current_q
        )
        self.q_table[s][action] = new_q

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    def save(self, filepath="rl_model.pkl"):
        """Save trained agent"""
        with open(filepath, "wb") as f:
            pickle.dump({
                "q_table":     self.q_table,
                "epsilon":     self.epsilon,
                "n_actions":   self.n_actions,
                "n_bins":      self.n_bins,
                "bins":        self.bins,
                "lr":          self.lr,
                "gamma":       self.gamma,
            }, f)
        print(f"  ✅ RL model saved → {filepath}")

    def load(self, filepath="rl_model.pkl"):
        """Load trained agent"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table   = data["q_table"]
        self.epsilon   = data["epsilon"]
        self.n_actions = data["n_actions"]
        self.n_bins    = data["n_bins"]
        self.bins      = data["bins"]
        print(f"  ✅ RL model loaded ← {filepath}")

# ── Training ───────────────────────────────────────────────────────
def train(episodes=500, max_steps=100):
    print("="*55)
    print("  Q-LEARNING AGENT TRAINING")
    print("  TCP Variant: Adaptive Switching")
    print("="*55)

    env   = TCPEnvironment()
    agent = QLearningAgent(
        n_actions    = len(TCP_VARIANTS),
        learning_rate= 0.1,
        discount     = 0.95,
        epsilon      = 1.0,
        epsilon_min  = 0.01,
        epsilon_decay= 0.995,
        n_bins       = 10
    )

    episode_rewards = []
    tcp_usage       = {tcp: 0 for tcp in TCP_VARIANTS}
    switch_counts   = []
    best_reward     = -np.inf

    print(f"\n  Training for {episodes} episodes...")
    print(f"  Max steps per episode: {max_steps}")
    print(f"  TCP variants: {TCP_VARIANTS}")

    for ep in range(episodes):
        state        = env.reset()
        total_reward = 0
        switches     = 0
        prev_action  = -1

        for step in range(max_steps):
            action = agent.get_action(state)

            if prev_action != -1 and action != prev_action:
                switches += 1

            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)

            total_reward += reward
            tcp_usage[TCP_VARIANTS[action]] += 1
            prev_action = action
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        switch_counts.append(switches)

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save("rl_model.pkl")

        # Progress update every 50 episodes
        if (ep+1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_switch = np.mean(switch_counts[-50:])
            print(f"  Episode {ep+1:4d}/{episodes} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Avg Switches: {avg_switch:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    print(f"\n  ✅ Training complete!")
    print(f"  Best reward    : {best_reward:.3f}")
    print(f"  Final epsilon  : {agent.epsilon:.3f}")
    print(f"\n  TCP Usage distribution:")
    total_steps = sum(tcp_usage.values())
    for tcp, count in tcp_usage.items():
        pct = count/total_steps*100
        bar = "█" * int(pct/3)
        print(f"    {tcp:8s} {bar} {pct:.1f}%")

    return agent, episode_rewards, switch_counts, tcp_usage

# ── Evaluation ─────────────────────────────────────────────────────
def evaluate(agent, episodes=50):
    print("\n" + "="*55)
    print("  AGENT EVALUATION")
    print("="*55)

    env = TCPEnvironment()
    agent.epsilon = 0.0  # No exploration during eval

    eval_rewards = []
    tcp_choices  = {tcp: 0 for tcp in TCP_VARIANTS}
    correct      = 0
    total        = 0

    for ep in range(episodes):
        state        = env.reset()
        total_reward = 0

        for step in range(100):
            action     = agent.get_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            tcp_choices[TCP_VARIANTS[action]] += 1
            total += 1
            state = next_state
            if done:
                break

        eval_rewards.append(total_reward)

    avg_reward = np.mean(eval_rewards)
    print(f"\n  Avg Reward (eval): {avg_reward:.3f}")
    print(f"\n  TCP Selection during evaluation:")
    total_steps = sum(tcp_choices.values())
    for tcp, count in tcp_choices.items():
        pct = count/total_steps*100 if total_steps > 0 else 0
        bar = "█" * int(pct/3)
        print(f"    {tcp:8s} {bar} {pct:.1f}%")

    return eval_rewards, tcp_choices

# ── Plots ──────────────────────────────────────────────────────────
def plot_training(episode_rewards, switch_counts, tcp_usage, eval_rewards):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Q-Learning Agent — TCP Adaptive Switching",
                 fontsize=14, fontweight='bold')

    COLORS = {
        "reno":"#e74c3c","tahoe":"#3498db","vegas":"#2ecc71",
        "cubic":"#f39c12","bbr":"#9b59b6","dctcp":"#1abc9c"
    }

    # 1. Training rewards
    ax1 = axes[0][0]
    window = 20
    smoothed = pd.Series(episode_rewards).rolling(window).mean()
    ax1.plot(episode_rewards, alpha=0.3, color="#3498db", label="Raw")
    ax1.plot(smoothed, color="#e74c3c", linewidth=2, label=f"Smoothed ({window}ep)")
    ax1.set_title("Training Reward over Episodes")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. TCP usage during training
    ax2 = axes[0][1]
    total = sum(tcp_usage.values())
    tcps  = list(tcp_usage.keys())
    pcts  = [tcp_usage[t]/total*100 for t in tcps]
    bars  = ax2.bar(tcps, pcts,
                    color=[COLORS.get(t,"#95a5a6") for t in tcps])
    ax2.set_title("TCP Usage During Training")
    ax2.set_ylabel("Usage (%)")
    ax2.grid(True, alpha=0.3, axis='y')
    for b, p in zip(bars, pcts):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                 f"{p:.1f}%", ha='center', fontsize=8)

    # 3. Switch count over training
    ax3 = axes[1][0]
    sw_smooth = pd.Series(switch_counts).rolling(window).mean()
    ax3.plot(switch_counts, alpha=0.3, color="#2ecc71")
    ax3.plot(sw_smooth, color="#e74c3c", linewidth=2)
    ax3.set_title("TCP Switches per Episode")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Number of Switches")
    ax3.grid(True, alpha=0.3)

    # 4. Evaluation rewards
    ax4 = axes[1][1]
    ax4.hist(eval_rewards, bins=20, color="#9b59b6", alpha=0.8, edgecolor='white')
    ax4.axvline(np.mean(eval_rewards), color='red', linestyle='--',
                linewidth=2, label=f"Mean: {np.mean(eval_rewards):.3f}")
    ax4.set_title("Evaluation Reward Distribution")
    ax4.set_xlabel("Total Reward")
    ax4.set_ylabel("Frequency")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rl_results.png", dpi=150, bbox_inches='tight')
    print("\n  ✅ Plots saved → rl_results.png")

# ── Save results JSON ──────────────────────────────────────────────
def save_rl_results(episode_rewards, eval_rewards, tcp_usage, switch_counts):
    total = sum(tcp_usage.values())
    results = {
        "training": {
            "episodes":        len(episode_rewards),
            "avg_reward":      round(float(np.mean(episode_rewards)), 4),
            "best_reward":     round(float(np.max(episode_rewards)),  4),
            "final_avg":       round(float(np.mean(episode_rewards[-50:])), 4),
            "avg_switches":    round(float(np.mean(switch_counts)),   2),
        },
        "evaluation": {
            "episodes":    len(eval_rewards),
            "avg_reward":  round(float(np.mean(eval_rewards)), 4),
            "std_reward":  round(float(np.std(eval_rewards)),  4),
        },
        "tcp_usage": {
            tcp: round(count/total*100, 2)
            for tcp, count in tcp_usage.items()
        }
    }
    with open("rl_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  ✅ RL results saved → rl_results.json")
    return results

# ── Demo: Real-time TCP switching ─────────────────────────────────
def demo_realtime(agent):
    print("\n" + "="*55)
    print("  DEMO: REAL-TIME TCP SWITCHING")
    print("="*55)

    env           = TCPEnvironment(app_type="streaming")
    state         = env.reset()
    agent.epsilon = 0.0

    print(f"\n  App type: streaming")
    print(f"  Network profile: {env.profile}")
    print(f"\n  {'Step':>4} | {'TCP':>8} | {'RTT(ms)':>8} | "
          f"{'Loss':>8} | {'TP(Mbps)':>8} | {'Reward':>8}")
    print("  " + "-"*55)

    prev_tcp = None
    for step in range(15):
        action = agent.get_action(state)
        tcp    = TCP_VARIANTS[action]
        next_state, reward, done = env.step(action)

        switch = "⟳" if prev_tcp and tcp != prev_tcp else " "
        print(f"  {step+1:4d} | {tcp:>8s} {switch} | "
              f"{state[1]:8.1f} | "
              f"{state[2]:8.4f} | "
              f"{state[0]:8.3f} | "
              f"{reward:8.4f}")

        prev_tcp = tcp
        state    = next_state
        if done:
            break

# ── Main ───────────────────────────────────────────────────────────
def main():
    # Train
    agent, episode_rewards, switch_counts, tcp_usage = train(
        episodes=1000, max_steps=100
    )

    # Evaluate
    eval_rewards, tcp_choices = evaluate(agent, episodes=50)

    # Plot
    plot_training(episode_rewards, switch_counts, tcp_usage, eval_rewards)

    # Save results
    results = save_rl_results(
        episode_rewards, eval_rewards, tcp_usage, switch_counts
    )

    # Demo
    demo_realtime(agent)

    print("\n" + "="*55)
    print("  ✅ RL Training Complete!")
    print("  Files saved:")
    print("    → rl_model.pkl    (trained Q-table)")
    print("    → rl_results.png  (training plots)")
    print("    → rl_results.json (metrics)")
    print("="*55)

if __name__ == "__main__":
    main()