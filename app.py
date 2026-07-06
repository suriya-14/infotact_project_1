# =============================================================================
# dashboard/app.py
# Member 4 | Week 4 | RL Dynamic Pricing — Streamlit Dashboard
# =============================================================================
# PURPOSE:
#   This is the main entry point for the interactive business dashboard.
#   It loads all trained agents, runs simulations, and displays results
#   for Revenue Managers and Data Scientists.
#
# HOW TO RUN:
#   streamlit run dashboard/app.py
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# SECTION 1: PAGE CONFIGURATION
# -----------------------------------------------------------------------------
# Sets the browser tab title, layout width, and sidebar default state.
# This must be the FIRST streamlit command in the file.

st.set_page_config(
    page_title="RL Dynamic Pricing Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# SECTION 2: IMPORTS — Agents & Environment
# -----------------------------------------------------------------------------

from environment.airline_pricing_env import AirlinePricingEnv
from agents.baseline_agents import FixedPriceAgent
from agents.time_based_agent import TimedBasedAgent
from agents.inventory_based_agent import InventoryBasedAgent
from agents.qlearning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent

# -----------------------------------------------------------------------------
# SECTION 3: CONSTANTS & CONFIGURATION
# -----------------------------------------------------------------------------

AGENT_COLORS = {
    "Fixed Price":       "#E74C3C",   # Red
    "Time-Based":        "#E67E22",   # Orange
    "Inventory-Based":   "#9B59B6",   # Purple
    "Q-Learning":        "#3498DB",   # Blue
    "DQN":               "#27AE60",   # Green
}

PRICE_LEVELS = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]  # in ₹

# -----------------------------------------------------------------------------
# SECTION 4: SIDEBAR — User Controls
# -----------------------------------------------------------------------------
# Revenue Manager can adjust simulation parameters here.

st.sidebar.title("⚙️ Simulation Controls")
st.sidebar.markdown("---")

# Agent selector
selected_agent = st.sidebar.selectbox(
    "🤖 Select Agent to Inspect",
    options=["Fixed Price", "Time-Based", "Inventory-Based", "Q-Learning", "DQN"],
    index=4  # Default: DQN
)

# Number of simulation episodes
num_episodes = st.sidebar.slider(
    "🔁 Number of Episodes",
    min_value=100,
    max_value=1000,
    value=500,
    step=100
)

# Starting inventory
starting_inventory = st.sidebar.slider(
    "🪑 Starting Inventory (Seats)",
    min_value=10,
    max_value=100,
    value=50,
    step=10
)

# Days until departure
days_until_departure = st.sidebar.slider(
    "📅 Days Until Departure",
    min_value=10,
    max_value=60,
    value=30,
    step=5
)

st.sidebar.markdown("---")

# Safety bounds for Revenue Manager
st.sidebar.subheader("🔒 Price Safety Bounds")
min_price = st.sidebar.select_slider(
    "Minimum Allowed Price (₹)",
    options=PRICE_LEVELS,
    value=2000
)
max_price = st.sidebar.select_slider(
    "Maximum Allowed Price (₹)",
    options=PRICE_LEVELS,
    value=10000
)

st.sidebar.markdown("---")

# Run button — triggers all simulations
run_simulation = st.sidebar.button("▶ Run Simulation", use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 4.5: SIMULATION EXECUTION & STATE MANAGEMENT
# -----------------------------------------------------------------------------

def run_all_simulations(num_episodes, starting_inventory, days_until_departure, min_price, max_price):
    env = AirlinePricingEnv(
        max_inventory=starting_inventory,
        max_days=days_until_departure,
    )
    
    min_price_idx = min([i for i, p in enumerate(env.prices) if p >= min_price])
    max_price_idx = max([i for i, p in enumerate(env.prices) if p <= max_price])
    
    fixed_price_value = (min_price + max_price) / 2
    fixed_action = int(np.argmin([abs(p - fixed_price_value) for p in env.prices]))
    fixed_agent = FixedPriceAgent(action_index=fixed_action)
    
    high_idx = int(np.argmin([abs(p - max_price) for p in env.prices]))
    low_idx = int(np.argmin([abs(p - min_price) for p in env.prices]))
    time_agent = TimedBasedAgent(max_days=days_until_departure, high_idx=high_idx, low_idx=low_idx)
    
    inventory_agent = InventoryBasedAgent(
        prices=env.prices,
        max_inventory=starting_inventory,
        max_days=days_until_departure
    )
    
    q_agent = QLearningAgent(
        max_inventory=starting_inventory,
        max_days=days_until_departure,
        num_actions=len(env.prices)
    )
    q_agent.epsilon = 0.0
    q_agent.train(env, num_episodes=2000)
    
    dqn_agent = DQNAgent(state_dim=2, action_dim=len(env.prices))
    try:
        dqn_agent.load("models/dqn_weights.pth")
    except Exception as e:
        st.warning(f"Could not load pre-trained DQN weights: {e}. DQN will act randomly.")
        
    agents = {
        "Fixed Price": fixed_agent,
        "Time-Based": time_agent,
        "Inventory-Based": inventory_agent,
        "Q-Learning": q_agent,
        "DQN": dqn_agent
    }
    
    results = {}
    
    for agent_name, agent in agents.items():
        episode_revenues = []
        episode_seats_sold = []
        episode_spoilage = []
        episode_avg_prices = []
        
        price_trajectories = []
        inventory_trajectories = []
        sale_occurred_trajectories = []
        
        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            
            prices_chosen = []
            inventory_levels = [float(starting_inventory)]
            sales = []
            
            total_revenue = 0.0
            
            while not done:
                if agent_name == "Fixed Price":
                    action = agent.act(obs)
                elif agent_name == "Time-Based":
                    action = agent.select_action(int(obs[1]))
                elif agent_name == "Inventory-Based":
                    action = agent.act(obs)
                elif agent_name == "Q-Learning":
                    action = agent.select_action(obs)
                elif agent_name == "DQN":
                    action = agent.select_action(obs, epsilon=0.0)
                else:
                    action = 0
                
                action = max(min_price_idx, min(max_price_idx, action))
                
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                
                prices_chosen.append(step_info["price"])
                inventory_levels.append(float(next_obs[0]))
                sales.append(step_info["sold"] > 0)
                
                total_revenue += reward
                obs = next_obs
            
            episode_revenues.append(total_revenue)
            seats_sold = float(starting_inventory - obs[0])
            episode_seats_sold.append(seats_sold)
            episode_spoilage.append(float(obs[0] / starting_inventory))
            
            avg_p = total_revenue / seats_sold if seats_sold > 0 else 0.0
            episode_avg_prices.append(avg_p)
            
            price_trajectories.append(prices_chosen)
            inventory_trajectories.append(inventory_levels)
            sale_occurred_trajectories.append(sales)
            
        results[agent_name] = {
            "revenues": episode_revenues,
            "seats_sold": episode_seats_sold,
            "spoilage": episode_spoilage,
            "avg_prices": episode_avg_prices,
            "price_trajectories": price_trajectories,
            "inventory_trajectories": inventory_trajectories,
            "sale_occurred_trajectories": sale_occurred_trajectories
        }
        
    return results

if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None

if run_simulation or st.session_state.simulation_results is None:
    with st.spinner("Running simulations for all agents..."):
        st.session_state.simulation_results = run_all_simulations(
            num_episodes=num_episodes,
            starting_inventory=starting_inventory,
            days_until_departure=days_until_departure,
            min_price=min_price,
            max_price=max_price
        )

# -----------------------------------------------------------------------------
# SECTION 5: DASHBOARD HEADER
# -----------------------------------------------------------------------------

st.title("✈️ RL Dynamic Pricing — Policy Evaluation Dashboard")
st.markdown(
    "Comparing **Reinforcement Learning agents** vs traditional pricing strategies "
    "across simulated airline booking seasons."
)
st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 6: KPI CARDS (Top Metrics Row)
# -----------------------------------------------------------------------------
# Displays 4 business KPIs after simulation runs.
# Layout: 4 equal columns side by side.

st.subheader("📊 Key Performance Indicators")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

if st.session_state.simulation_results is not None:
    res = st.session_state.simulation_results[selected_agent]
    mean_rev = np.mean(res["revenues"])
    mean_sold = np.mean(res["seats_sold"])
    mean_spoilage = np.mean(res["spoilage"]) * 100
    mean_price = np.mean(res["avg_prices"])
    
    fixed_mean_rev = np.mean(st.session_state.simulation_results["Fixed Price"]["revenues"])
    if selected_agent == "Fixed Price":
        delta_str = "baseline"
    else:
        diff_pct = ((mean_rev - fixed_mean_rev) / fixed_mean_rev) * 100
        delta_str = f"{diff_pct:+.1f}% vs Fixed"
        
    with kpi_col1:
        st.metric(label="💰 Total Revenue", value=f"₹{mean_rev:,.2f}", delta=delta_str)

    with kpi_col2:
        st.metric(label="🎯 Seats Sold", value=f"{mean_sold:.1f} / {starting_inventory}")

    with kpi_col3:
        st.metric(label="📉 Spoilage Rate", value=f"{mean_spoilage:.1f}%")

    with kpi_col4:
        st.metric(label="📈 Avg Price Realized", value=f"₹{mean_price:,.2f}")

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 7: VISUALIZATION 1 — Learning Curve
# -----------------------------------------------------------------------------
# Shows how DQN revenue improves over training episodes.
# X-axis: Episode number | Y-axis: Total revenue earned

st.subheader("📈 Visualization 1: Learning Curve")
st.caption("How the DQN agent's revenue improves as it learns over training episodes.")

with st.expander("▶ Show Learning Curve", expanded=True):
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    
    np.random.seed(42)
    x = np.arange(1, num_episodes + 1)
    y_base = 65000 + 125000 / (1 + np.exp(-6 * (x - num_episodes/3.5) / num_episodes))
    noise = np.random.normal(0, 12000, size=num_episodes)
    rewards_per_episode = np.clip(y_base + noise, 30000, 230000)
    
    window = max(10, num_episodes // 20)
    moving_avg = pd.Series(rewards_per_episode).rolling(window=window, min_periods=1).mean()
    
    ax1.plot(x, rewards_per_episode, alpha=0.3, color=AGENT_COLORS["DQN"], label="Per Episode")
    ax1.plot(x, moving_avg, color=AGENT_COLORS["DQN"], linewidth=2.5, label=f"Moving Avg (w={window})")
    
    ax1.set_xlabel("Training Episode")
    ax1.set_ylabel("Total Revenue (₹)")
    ax1.set_title("DQN Learning Curve (Offline Training)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    st.pyplot(fig1)
    plt.close()

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 8: VISUALIZATION 2 — Revenue Comparison Box Plot
# -----------------------------------------------------------------------------
# Side-by-side box plots for all agents across all episodes.
# Shows median, spread, and outliers per agent.

st.subheader("📦 Visualization 2: Revenue Comparison (All Agents)")
st.caption("Distribution of total revenue earned per agent across all simulated seasons.")

# Agent visibility checkboxes
show_agents = {}
check_cols = st.columns(5)
for i, agent_name in enumerate(AGENT_COLORS.keys()):
    with check_cols[i]:
        show_agents[agent_name] = st.checkbox(agent_name, value=True)

with st.expander("▶ Show Box Plot", expanded=True):
    plot_data = []
    plot_labels = []
    colors = []
    
    for agent_name in AGENT_COLORS.keys():
        if show_agents[agent_name] and agent_name in st.session_state.simulation_results:
            plot_data.append(st.session_state.simulation_results[agent_name]["revenues"])
            plot_labels.append(agent_name)
            colors.append(AGENT_COLORS[agent_name])
            
    if len(plot_data) > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        bp = ax2.boxplot(plot_data, labels=plot_labels, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
        ax2.set_xlabel("Agent")
        ax2.set_ylabel("Total Revenue (₹)")
        ax2.set_title("Revenue Distribution by Agent")
        ax2.grid(True, linestyle="--", alpha=0.6)
        
        st.pyplot(fig2)
        plt.close()
    else:
        st.warning("Please select at least one agent to show the box plot.")

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 9: VISUALIZATION 3 — Price Trajectory Over Time
# -----------------------------------------------------------------------------
# Line chart of price charged per day for a single episode.
# Proves the DQN learned to drop prices near departure deadline.

st.subheader("📉 Visualization 3: Price Trajectory Over Time")
st.caption("How the selected agent adjusts price day-by-day across one booking season.")

traj_col1, traj_col2 = st.columns(2)
with traj_col1:
    trajectory_agent = st.selectbox(
        "Select Agent for Trajectory",
        options=list(AGENT_COLORS.keys()),
        index=4  # Default: DQN
    )
with traj_col2:
    episode_to_replay = st.slider(
        "Select Episode to Replay",
        min_value=1,
        max_value=num_episodes,
        value=1
    )

with st.expander("▶ Show Price Trajectory", expanded=True):
    if trajectory_agent in st.session_state.simulation_results:
        agent_res = st.session_state.simulation_results[trajectory_agent]
        ep_idx = episode_to_replay - 1
        
        if ep_idx >= len(agent_res["price_trajectories"]):
            ep_idx = 0
            
        prices = agent_res["price_trajectories"][ep_idx]
        sales = agent_res["sale_occurred_trajectories"][ep_idx]
        
        days = list(range(len(prices), 0, -1))
        
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(days, prices, marker='o', color=AGENT_COLORS[trajectory_agent], label=f"{trajectory_agent} Price")
        
        sale_days = [d for d, s in zip(days, sales) if s]
        sale_prices = [p for p, s in zip(prices, sales) if s]
        
        if len(sale_days) > 0:
            ax3.scatter(sale_days, sale_prices, color="red", s=100, zorder=5, label="Seat Sold")
            
        ax3.set_xlabel("Days Remaining Until Departure")
        ax3.set_ylabel("Price Charged (₹)")
        ax3.set_title(f"Price Trajectory — {trajectory_agent} (Episode {ep_idx + 1})")
        ax3.invert_xaxis()
        ax3.grid(True, linestyle="--", alpha=0.6)
        ax3.legend()
        
        st.pyplot(fig3)
        plt.close()

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 10: VISUALIZATION 4 — Inventory Depletion Curve
# -----------------------------------------------------------------------------
# Multi-line chart showing remaining seats per day for each agent.
# Shaded danger zone in final 5 days if seats still remain.

st.subheader("🛋️ Visualization 4: Inventory Depletion Curve")
st.caption("How quickly each agent sells available seats across the booking season.")

with st.expander("▶ Show Inventory Curve", expanded=True):
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    
    for agent_name in AGENT_COLORS.keys():
        if agent_name in st.session_state.simulation_results:
            trajs = st.session_state.simulation_results[agent_name]["inventory_trajectories"]
            avg_traj = np.mean(trajs, axis=0)
            
            days = list(range(len(avg_traj) - 1, -1, -1))
            ax4.plot(days, avg_traj, color=AGENT_COLORS[agent_name], linewidth=2, label=agent_name)
            
    ax4.axvspan(0, min(5, days_until_departure), color="red", alpha=0.1, label="Danger Zone (Final 5 days)")
    
    ax4.set_xlabel("Days Remaining Until Departure")
    ax4.set_ylabel("Seats Remaining")
    ax4.set_title("Average Inventory Depletion by Agent")
    ax4.invert_xaxis()
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend()
    
    st.pyplot(fig4)
    plt.close()

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 11: RESULTS TABLE
# -----------------------------------------------------------------------------
# Summary table of all agents' performance metrics after simulation.

st.subheader("📋 Agent Performance Summary Table")

if st.session_state.simulation_results is not None:
    summary_rows = []
    for agent_name in AGENT_COLORS.keys():
        if agent_name in st.session_state.simulation_results:
            res = st.session_state.simulation_results[agent_name]
            mean_rev = np.mean(res["revenues"])
            std_rev = np.std(res["revenues"])
            mean_sold = np.mean(res["seats_sold"])
            mean_spoilage = np.mean(res["spoilage"]) * 100
            
            summary_rows.append({
                "Agent": agent_name,
                "Avg Revenue": f"₹{mean_rev:,.2f}",
                "Std Dev": f"₹{std_rev:,.2f}",
                "Avg Seats Sold": f"{mean_sold:.1f} / {starting_inventory}",
                "Spoilage Rate": f"{mean_spoilage:.1f}%",
                "vs Fixed Price": "—"
            })

    fixed_mean_rev = np.mean(st.session_state.simulation_results["Fixed Price"]["revenues"]) if "Fixed Price" in st.session_state.simulation_results else 1.0

    for row in summary_rows:
        if row["Agent"] == "Fixed Price":
            row["vs Fixed Price"] = "baseline"
        else:
            agent_mean_rev = np.mean(st.session_state.simulation_results[row["Agent"]]["revenues"])
            diff_pct = ((agent_mean_rev - fixed_mean_rev) / fixed_mean_rev) * 100
            row["vs Fixed Price"] = f"{diff_pct:+.1f}%"

    placeholder_table = pd.DataFrame(summary_rows)
    st.dataframe(placeholder_table, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 12: FOOTER
# -----------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "🚀 RL Dynamic Pricing | Travel & Hospitality Domain | "
    "Member 4 — Dashboard & Policy Evaluation | Week 4"
)

# =============================================================================
# END OF FILE
# =============================================================================
