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
# Uncomment these once the other members share their code files.
# All agents follow the same interface: agent.run_episode(env) → (revenue, prices, inventory)

# from environment.airline_pricing_env import AirlinePricingEnv
# from agents.fixed_price_agent import FixedPriceAgent
# from agents.time_based_agent import TimeBasedAgent
# from agents.inventory_based_agent import InventoryBasedAgent
# from agents.qlearning_agent import QLearningAgent
# from agents.dqn_agent import DQNAgent

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

with kpi_col1:
    # TODO: Replace 0 with: total_revenue from simulation results
    st.metric(label="💰 Total Revenue", value="₹0", delta="vs Fixed Price")

with kpi_col2:
    # TODO: Replace 0/50 with: seats_sold / starting_inventory
    st.metric(label="🎯 Seats Sold", value=f"0 / {starting_inventory}")

with kpi_col3:
    # TODO: Replace 0% with: spoilage_rate from simulation results
    st.metric(label="📉 Spoilage Rate", value="0%")

with kpi_col4:
    # TODO: Replace 0 with: avg_price from simulation results
    st.metric(label="📈 Avg Price Realized", value="₹0")

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 7: VISUALIZATION 1 — Learning Curve
# -----------------------------------------------------------------------------
# Shows how DQN revenue improves over training episodes.
# X-axis: Episode number | Y-axis: Total revenue earned

st.subheader("📈 Visualization 1: Learning Curve")
st.caption("How the DQN agent's revenue improves as it learns over training episodes.")

# TODO: Replace this placeholder with actual training reward history from DQN agent
# Expected data shape: rewards_per_episode = [float, float, ...] length = num_episodes

with st.expander("▶ Show Learning Curve", expanded=True):
    fig1, ax1 = plt.subplots(figsize=(10, 4))

    # --- PLACEHOLDER CHART (remove after real data is available) ---
    ax1.text(0.5, 0.5, "⏳ Run simulation to see Learning Curve",
             ha='center', va='center', fontsize=14, color='gray',
             transform=ax1.transAxes)
    ax1.set_xlabel("Training Episode")
    ax1.set_ylabel("Total Revenue (₹)")
    ax1.set_title("DQN Learning Curve")
    # ----------------------------------------------------------------

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

# TODO: Replace placeholder with: revenue_data dict {agent_name: [revenues list]}
# Each agent runs num_episodes times → list of total revenues

with st.expander("▶ Show Box Plot", expanded=True):
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    # --- PLACEHOLDER CHART (remove after real data is available) ---
    ax2.text(0.5, 0.5, "⏳ Run simulation to see Revenue Comparison",
             ha='center', va='center', fontsize=14, color='gray',
             transform=ax2.transAxes)
    ax2.set_xlabel("Agent")
    ax2.set_ylabel("Total Revenue (₹)")
    ax2.set_title("Revenue Distribution by Agent")
    # ----------------------------------------------------------------

    st.pyplot(fig2)
    plt.close()

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

# TODO: Replace placeholder with: price_trajectory list [price_on_day_1, ..., price_on_day_N]
# Annotate days where a sale occurred with a dot marker

with st.expander("▶ Show Price Trajectory", expanded=True):
    fig3, ax3 = plt.subplots(figsize=(10, 4))

    # --- PLACEHOLDER CHART (remove after real data is available) ---
    ax3.text(0.5, 0.5, "⏳ Run simulation to see Price Trajectory",
             ha='center', va='center', fontsize=14, color='gray',
             transform=ax3.transAxes)
    ax3.set_xlabel("Days Remaining Until Departure")
    ax3.set_ylabel("Price Charged (₹)")
    ax3.set_title(f"Price Trajectory — {trajectory_agent}")
    ax3.invert_xaxis()  # Day 30 on left → Day 1 on right
    # ----------------------------------------------------------------

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

# TODO: Replace placeholder with: inventory_data dict {agent_name: [inventory_per_day list]}
# Add shaded region for final 5 days as "danger zone"

with st.expander("▶ Show Inventory Curve", expanded=True):
    fig4, ax4 = plt.subplots(figsize=(10, 4))

    # --- PLACEHOLDER CHART (remove after real data is available) ---
    ax4.text(0.5, 0.5, "⏳ Run simulation to see Inventory Depletion",
             ha='center', va='center', fontsize=14, color='gray',
             transform=ax4.transAxes)
    ax4.set_xlabel("Days Remaining Until Departure")
    ax4.set_ylabel("Seats Remaining")
    ax4.set_title("Inventory Depletion by Agent")
    ax4.invert_xaxis()
    # ----------------------------------------------------------------

    st.pyplot(fig4)
    plt.close()

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 11: RESULTS TABLE
# -----------------------------------------------------------------------------
# Summary table of all agents' performance metrics after simulation.

st.subheader("📋 Agent Performance Summary Table")

# TODO: Replace with real simulation results after running all agents
placeholder_table = pd.DataFrame({
    "Agent":         ["Fixed Price", "Time-Based", "Inventory-Based", "Q-Learning", "DQN"],
    "Avg Revenue":   ["—", "—", "—", "—", "—"],
    "Std Dev":       ["—", "—", "—", "—", "—"],
    "Avg Seats Sold":["—", "—", "—", "—", "—"],
    "Spoilage Rate": ["—", "—", "—", "—", "—"],
    "vs Fixed Price":["baseline", "—", "—", "—", "—"],
})

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
