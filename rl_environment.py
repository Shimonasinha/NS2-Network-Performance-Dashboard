# rl_environment.py
# TCP Congestion Control Environment for RL Agent
# State: network conditions | Action: switch TCP variant | Reward: performance

import numpy as np
import pandas as pd
import random

random.seed(42)
np.random.seed(42)

# ── TCP Variants ───────────────────────────────────────────────────
TCP_VARIANTS = ["reno", "tahoe", "vegas", "cubic", "bbr", "dctcp"]

# ── Realistic network profiles ─────────────────────────────────────
NETWORK_PROFILES = {
    "low_load": {
        "throughput_range": (7.0, 10.0),
        "rtt_range":        (20.0, 80.0),
        "loss_range":       (0.001, 0.01),
        "description":      "Low congestion network"
    },
    "medium_load": {
        "throughput_range": (4.0, 7.0),
        "rtt_range":        (80.0, 150.0),
        "loss_range":       (0.01, 0.05),
        "description":      "Moderate congestion"
    },
    "high_load": {
        "throughput_range": (1.0, 4.0),
        "rtt_range":        (150.0, 300.0),
        "loss_range":       (0.05, 0.15),
        "description":      "High congestion network"
    },
    "datacenter": {
        "throughput_range": (8.0, 12.0),
        "rtt_range":        (0.1, 2.0),
        "loss_range":       (0.0001, 0.002),
        "description":      "Datacenter low-latency"
    },
}

# ── Best TCP per condition (ground truth) ─────────────────────────
def get_optimal_tcp(state, app_type):
    throughput, rtt, loss, _, _ = state
    if rtt < 2.0 and loss < 0.002:
        return TCP_VARIANTS.index("dctcp")
    elif app_type == "streaming" and throughput > 7.0 and rtt < 40:
        return TCP_VARIANTS.index("bbr")
    elif rtt < 80 and loss < 0.02:
        return TCP_VARIANTS.index("cubic")
    elif loss > 0.06:
        return TCP_VARIANTS.index("tahoe")
    elif rtt < 140 and loss < 0.03:
        return TCP_VARIANTS.index("vegas")
    else:
        return TCP_VARIANTS.index("reno")

class TCPEnvironment:
    """
    RL Environment for TCP Variant Selection
    
    State  : [throughput, rtt, loss_rate, network_load, app_type_enc]
    Action : Select TCP variant (0-5)
    Reward : Based on performance improvement
    """

    def __init__(self, app_type="streaming"):
        self.app_type       = app_type
        self.app_types      = ["streaming", "io", "sort"]
        self.n_actions      = len(TCP_VARIANTS)
        self.n_states       = 5
        self.current_state  = None
        self.current_tcp    = 0
        self.step_count     = 0
        self.max_steps      = 100
        self.profile        = "medium_load"
        self.history        = []

    def _generate_state(self):
        profile = NETWORK_PROFILES[self.profile]
        tp      = random.uniform(*profile["throughput_range"])
        rtt     = random.uniform(*profile["rtt_range"])
        loss    = random.uniform(*profile["loss_range"])
        load    = random.uniform(0.1, 1.0)
        app_enc = self.app_types.index(self.app_type) / len(self.app_types)

        # Add noise to simulate real network variability
        tp   = max(0.1, tp   + random.gauss(0, tp*0.05))
        rtt  = max(0.1, rtt  + random.gauss(0, rtt*0.05))
        loss = max(0.0, loss + random.gauss(0, loss*0.05))

        return np.array([tp, rtt, loss, load, app_enc])

    def reset(self):
        # Randomly change network profile each episode
        self.profile       = random.choice(list(NETWORK_PROFILES.keys()))
        self.app_type      = random.choice(self.app_types)
        self.current_state = self._generate_state()
        self.current_tcp   = random.randint(0, self.n_actions-1)
        self.step_count    = 0
        self.history       = []
        return self.current_state

    def step(self, action):
        """
        Take action (select TCP variant) and return next state + reward
        """
        self.step_count += 1

        # Calculate reward based on action
        reward = self._calculate_reward(action, self.current_state)

        # Record history
        self.history.append({
            "step":       self.step_count,
            "state":      self.current_state.copy(),
            "action":     action,
            "tcp":        TCP_VARIANTS[action],
            "reward":     reward,
            "app_type":   self.app_type,
            "profile":    self.profile,
        })

        # Evolve network conditions slightly
        self.current_state = self._evolve_state(self.current_state)
        self.current_tcp   = action

        done = self.step_count >= self.max_steps
        return self.current_state, reward, done

    def _calculate_reward(self, action, state):
        """
        Reward function:
        - Streaming: minimize latency
        - IO:        maximize throughput
        - Sort:      minimize completion time (balance both)
        """
        throughput, rtt, loss, load, _ = state
        tcp = TCP_VARIANTS[action]

        # Base performance scores per TCP variant
        perf = {
            "reno":  {"tp": 0.65, "rtt": 0.60, "loss_tol": 0.05},
            "tahoe": {"tp": 0.50, "rtt": 0.55, "loss_tol": 0.10},
            "vegas": {"tp": 0.70, "rtt": 0.75, "loss_tol": 0.02},
            "cubic": {"tp": 0.80, "rtt": 0.65, "loss_tol": 0.04},
            "bbr":   {"tp": 0.90, "rtt": 0.85, "loss_tol": 0.03},
            "dctcp": {"tp": 0.95, "rtt": 0.95, "loss_tol": 0.001},
        }

        p = perf[tcp]

        # Penalty if loss exceeds tolerance
        loss_penalty = max(0, (loss - p["loss_tol"]) * 10)

        # Reward based on app type
        if self.app_type == "streaming":
            # Minimize latency, maintain throughput
            reward = (p["rtt"] * 0.6 + p["tp"] * 0.4) - loss_penalty
        elif self.app_type == "io":
            # Maximize throughput
            reward = (p["tp"] * 0.7 + p["rtt"] * 0.3) - loss_penalty
        else:  # sort
            # Balance both
            reward = (p["tp"] * 0.5 + p["rtt"] * 0.5) - loss_penalty

        # Bonus for selecting optimal TCP
        optimal = get_optimal_tcp(state, self.app_type)
        if action == optimal:
            reward += 0.3

        # Penalty for unnecessary switching
        if action != self.current_tcp:
            reward -= 0.05

        return round(float(reward), 4)

    def _evolve_state(self, state):
        """Simulate network condition changes over time"""
        tp, rtt, loss, load, app_enc = state

        # Random walk with mean reversion
        profile = NETWORK_PROFILES[self.profile]
        tp_mean   = np.mean(profile["throughput_range"])
        rtt_mean  = np.mean(profile["rtt_range"])
        loss_mean = np.mean(profile["loss_range"])

        tp   = tp   + 0.1*(tp_mean - tp)   + random.gauss(0, 0.2)
        rtt  = rtt  + 0.1*(rtt_mean - rtt) + random.gauss(0, 2.0)
        loss = loss + 0.1*(loss_mean-loss)  + random.gauss(0, 0.002)
        load = np.clip(load + random.gauss(0, 0.05), 0.1, 1.0)

        tp   = max(0.1, tp)
        rtt  = max(0.1, rtt)
        loss = max(0.0, min(loss, 0.3))

        return np.array([tp, rtt, loss, load, app_enc])

    def get_state_description(self, state):
        tp, rtt, loss, load, _ = state
        return {
            "throughput_mbps": round(float(tp),   3),
            "rtt_ms":          round(float(rtt),  3),
            "loss_rate":       round(float(loss),  5),
            "network_load":    round(float(load),  2),
            "app_type":        self.app_type,
            "profile":         self.profile,
        }

if __name__ == "__main__":
    print("="*50)
    print("  TCP RL ENVIRONMENT TEST")
    print("="*50)

    env   = TCPEnvironment(app_type="streaming")
    state = env.reset()

    print(f"\n  Initial State:")
    desc = env.get_state_description(state)
    for k, v in desc.items():
        print(f"    {k:20s}: {v}")

    print(f"\n  Running 5 steps...")
    for i in range(5):
        action = random.randint(0, len(TCP_VARIANTS)-1)
        next_state, reward, done = env.step(action)
        print(f"  Step {i+1}: TCP={TCP_VARIANTS[action]:6s} | "
              f"Reward={reward:.4f} | "
              f"RTT={next_state[1]:.1f}ms")

    print(f"\n  ✅ Environment working correctly!")
    print("="*50)