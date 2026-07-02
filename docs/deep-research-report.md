# Environment Design

The airline dynamic-pricing environment is modelled as a Markov Decision Process (MDP) and implemented as a custom Gym-like `Env` class. In this framework, the **environment** encapsulates all airline logistics (inventory, time, demand simulation) and defines the interface to the learning agent (action input, state observation, reward output). Key components include:

- **State representation:** a complete description of the current booking period. For a simple single-route sale, this may include at least the *remaining inventory* (seats) and *time left* (selling periods). Additional features (e.g. previous price, current load factor, time-of-day) could be added to capture seasonality or customer arrival trends. To maintain the Markov property, the state must include all information needed to predict future demand (for instance, if demand tends to increase as departure nears, include “days remaining”). In one RL pricing model, state was defined as the current arrival-rate signal and remaining inventory. In general, we choose a compact state (e.g. `[inventory, days_remaining]`) that suffices to predict demand given price. 

- **Action space:** the set of pricing decisions the agent can make. Typically, actions are *discrete price points* or *discount levels*. For example, the agent might choose among prices from £20 to £100 in £2 increments (40 actions), or select a percentage discount off a base price. Alternatively, a continuous action (price multiplier) space can be used. Each action `a_t` is interpreted as the fare charged in period `t`. The environment must enforce valid actions (e.g. within price bounds) and often quantises continuous actions into bins if needed.  

- **State transitions:** defined by inventory depletion and time progression. After an action (price) is taken, the environment simulates sales according to the *demand function* (see Demand Function below), yielding some number of seats sold. The environment then updates:  
  \[
    \text{inventory}_{t+1} = \text{inventory}_t - \text{sales}_t,\quad
    \text{days_remaining}_{t+1} = \text{days_remaining}_t - 1.
  \]  
  If sales would exceed inventory, they are clamped to the remaining seats (no overselling). This transition probability, $P(s_{t+1}\mid s_t,a_t)$, captures the stochastic demand given the price. Liang *et al.* (2023) formalise this transition in their MDP description.  

- **Reward signals:** after each step, the environment returns a scalar reward. The natural choice is *immediate profit* or revenue: 
  \[
    r_t = p_t \times \text{sales}_t \ (\text{minus any per-seat cost if modelled}).
  \] 
  Thus higher prices can yield larger per-unit revenue but usually reduce sales. Positive reward thus corresponds to actual sales revenue. Additional shaping rewards or penalties can be added (see Reward Function).  

- **Action→Environment interface:** At each time step, the agent picks a price action; the environment “steps” by simulating demand and updating state. It returns the new state (observation), the reward, a done flag, and info. This loop continues until termination.  

Overall, the environment mimics a simplified airline selling process: in each period, an agent-set price and a stochastic demand model determine how many seats are sold. Design choices – such as whether to include competitor actions (in multi-agent contests) or richer time dynamics – can be guided by project scope, but the core structure remains: **state** (inventory, time, etc.), **action** (price), **transition** (sell and decrement), and **reward** (revenue). 

**Constraints and Best Practices:** The environment must enforce realistic constraints (e.g. inventory never negative, prices in allowed range) and provide all relevant information for the agent to make decisions. Use of a standard Gym interface (defining `action_space` and `observation_space`) is recommended for clarity. Keep state dimensionality as low as possible to reduce complexity, but include any features needed for Markovian demand prediction. A well-designed environment will be deterministic given its random seed and agent action (aside from intended demand randomness), ensuring reproducible episodes. 

# State & Action Space

## State Space

Each **state** $s_t$ must capture all information needed to decide pricing. In a basic airline pricing model, a minimal state could be: 
- $I_t$: remaining inventory (seats) at time $t$,
- $T_t$: time (or periods) remaining until departure.  

Optionally, one can include additional features, such as:
- Previous period’s price or sales (if demand has inertia or hidden trends),
- Current *load factor* (= sold/initial seats) or booking velocity,
- Time-of-day or day-of-week indicators for seasonality (encoded, e.g., via sine/cosine),
- Competitor prices (in a competitive scenario),
- External demand signals (marketing, events).  

For example, in Liang *et al.* the state combined an arrival-rate signal and inventory level. In an RL taxi pricing example, the observation included hour, day, recent demand and competitor-price ratios. In our simple case, a 2D state $(I_t,T_t)$ is usually sufficient (since knowing “days remaining” and “seats left” lets the agent infer booking urgency). 

**State space representation:** We can treat $(I,T)$ as discrete (integers) or continuous; typically one uses discrete values since seats and days are integer. The observation vector fed to the agent should be normalized (e.g., seats scaled to [0,1]) if using neural nets.  

**Design Rationale:** Including only invariant, necessary information keeps the state small. Extra information like previous actions can be omitted if transitions are deterministic given $(I,T)$ and price. However, if demand depends on latent factors, one might include history (e.g. *demand estimation* state). The goal is to satisfy the Markov property: future (distribution of) demand depends only on the current state and action, not deeper history. 

**Challenges and Trade-offs:** More complex states (e.g. multi-route, multi-class seats) explode state space, making learning harder (“curse of dimensionality”). We must balance realism versus tractability. In practice, start simple: for example, assume demand depends only on *days remaining* (higher demand closer to departure), and ignore competitor or season if not critical. One can always augment state later if performance suffers. 

## Action Space

The **action** $a_t$ is the price the agent sets at time $t$. It can be represented in various ways:

- **Discrete actions:** choose among a finite set of prices. For example, an agent might have 40 price options from £20 to £100 (step £2), or a set of percentage discounts (0%, 10%, …, 50%). Liang *et al.* model pricing via discrete discount levels. Discretization simplifies learning (finite Q-table or discrete DQN). 

- **Continuous actions:** choose any price within a range. In some implementations, agents output a real number (price multiplier). This allows finer control but requires continuous-action RL methods (e.g. DDPG, PPO).  

For clarity and stability, we typically use a discrete set of price actions. For instance, if the ticket list price is £100, we could define actions as percentages of that (100%, 90%, …, 50%). The environment should map each action to an actual price $p_t$. The action space is then $A = \{p^{(1)}, p^{(2)}, \dots, p^{(N)}\}$, and the agent selects an index or value in each step.

**Action Constraints:** Actions must respect feasible pricing. That means no negative or zero prices, and typically an upper bound based on market (e.g. £max the highest price a customer might pay). If a chosen action would violate constraints (e.g. price below cost or above max), the environment should clamp or reject it (e.g. give zero sales if price is outrageous). 

**State-Action Dependencies:** The transition $s_{t+1}=f(s_t,a_t)$ depends on the chosen price. For example, higher $p_t$ generally yields lower sales (due to price elasticity) and thus a smaller drop in inventory. This relationship can be encoded via a *demand function*. The agent’s action thus directly controls the stochastic transition of inventory. In all cases, we enforce that sales $\le$ inventory, so $I_{t+1}=\max(0,I_t-\text{sales}_t)$, and $T_{t+1}=T_t-1$. 

## Examples and Best Practices

- **State Example:** If initially 80 seats and 100 days, initial state might be $(I=80,T=100)$. If after one day we sell 2 seats, next state is $(I=78,T=99)$.  
- **Action Example:** Suppose actions are {£20, £25, …, £100}. If agent chooses £40 at $t$, price $p_t=40$ is served.  
- **Environment Class:** Implement a `step(action)` method that: simulates demand given the action (see Demand Function), computes sales, updates state, computes reward, and sets `done` if $I=0$ or $T=0$. A `reset()` method should reinitialise inventory and time to start a new episode. 

Defining clear `observation_space` and `action_space` (as in Gym) helps integrate with RL libraries. Always handle edge cases, e.g. if inventory is already 0, immediately set `done=True`. Validate that actions falling outside defined bins are not accepted.  

# Reward Function Design

The reward function guides the agent toward profitable pricing. In a revenue management context, the primary objective is to **maximize total sales revenue (or profit)** over the selling horizon. Accordingly, the natural immediate reward at each step is the *revenue from that period*:

\[
   r_t = p_t \times \text{sales}_t \quad \text{(revenue at time }t).
\]

If there is a variable cost per seat (e.g. operational cost), one could use profit = $(p_t - c)\times\text{sales}_t$. For simplicity, we often assume cost is zero or fixed and optimize revenue. Accumulating these rewards (possibly discounted) approximates total revenue. 

**Reward Shaping:** In addition to raw revenue, reward shaping can accelerate learning. Common strategies include:

- **Penalise unsold inventory:** To discourage holding seats to the end, one can impose a negative reward for leftover seats. For example, at episode end, add a penalty $-\lambda\cdot I_{\text{final}}$ (since unsold inventory is lost revenue). In perishable inventory RL, authors subtract disposal and lost-sales costs from profit. Similarly, a final negative term can push the agent to sell all seats.  
- **Time pressure cues:** Give a small bonus for selling early or a penalty for selling too fast/slow. For example, if staying unsold till the last period yields $-1$ per seat, the agent learns to avoid spoilage. 
- **No-sale penalty:** In some designs, if an action results in zero sales while inventory remains, one might give a slight penalty to discourage overpricing. 

Any shaping rewards must not obscure the main goal. For example, encouraging higher prices (if irrational) could conflict with revenue maximization. Reward shaping should always preserve the optimal policy. In practice, we often start with the simple reward = revenue and add penalties only if the learning agent shows pathological behaviour (like pricing so low or high that it never sells or sells out too early). 

**Success Metrics:** Ultimately, the performance metric is the cumulative episode reward (total revenue). The reward design should align with this. For instance, if we reward short-term revenue too strongly without penalising leftover seats, the agent might underprice early and sell out long before the horizon (forgoing future revenue). A balanced reward or final penalty avoids this “spill or spoil” issue. 

**Positive/Negative Rewards:** Positive rewards come from sales revenue. Negative rewards could include:
- Inventory spoilage (unsold seats at end),
- Violating soft constraints (e.g. extremely low load factor), 
- Encouragement of exploration (e.g. small noise reward to avoid being stuck). 

However, avoid making prices themselves negative rewards – pricing low is not a mistake per se, it may be optimal. The agent should learn the trade-off. 

**Design Rationale:** A well-shaped reward accelerates convergence. For example, de Moor *et al.* used reward shaping in a perishable inventory problem to speed up DQN learning. In pricing, one could shape by comparing revenue to a baseline strategy (giving extra reward when beating a fixed heuristic). But care must be taken: shaped rewards must not create local optima. Keeping rewards sparse (only revenue) ensures the agent optimizes true profits, while occasional shaping terms guide it gently. 

**Challenges:** Too much shaping can mislead. For instance, if we reward setting a high price regardless of sales, the agent may learn to always price high and get small negative penalties for unsold seats, which is suboptimal. Conversely, only rewarding revenue can make learning slow if episodes are long and signal is sparse. A trade-off is to use *discounted* revenue (so early sales count more) or small intermediate shaping. Testing different reward formulations and monitoring behavior is key. 

# Episode Lifecycle

An episode simulates one “selling period” (e.g. one flight). The **lifecycle** is as follows:

1. **Episode Initialization:** When `env.reset()` is called, set inventory to its initial value (e.g. 80 seats) and set time to the full horizon (e.g. 100 periods). Clear any past episode data. The initial state might simply be $(I=80, T=100)$.  

2. **Step Logic:** Each time step (while the episode is not done) proceeds:
   - The agent observes state $s_t$ and chooses an action $a_t$ (price).
   - The environment computes demand and sales given $a_t$ (see Demand Function), and assigns reward $r_t$ (revenue).  
   - The state is updated to $s_{t+1}$ with decreased inventory and time.  
   - The environment returns $(s_{t+1}, r_t, \text{done}, \text{info})$.  
   - Increment a time counter or internal step. 

3. **Time Progression:** We decrement the remaining time each step. After each pricing decision, one “period” elapses. 

4. **Termination Conditions:** The episode ends (`done=True`) if **either**:
   - Time runs out ($T=0$ after a step), meaning the selling horizon has ended.  
   - Inventory is depleted ($I=0$), meaning all seats sold before departure. In this case, no further sales can occur even if time remains. After sell-out, one can either end the episode immediately or continue stepping with zero sales; both amount to termination.  

   In practice, most designs end early on sell-out since no more revenue is possible. 

5. **Edge Cases:**  
   - If the agent attempts an invalid action (price out of bounds), the environment should handle it gracefully, e.g. by clamping to valid range or returning no sales.  
   - If inventory is already zero at reset (degenerate case), immediately end the episode.  
   - If extreme demand occurs (due to randomness), ensure sales $\le I$ by construction.  
   - If demand is zero (e.g. price too high), the episode still progresses normally. 

Throughout the episode, keep records of cumulative reward, seats sold, etc. At termination, you may compute final metrics (total revenue, load factor). 

**Best Practices:** Use a consistent seeding of randomness for reproducibility. Log the evolution of inventory and prices to analyze agent behaviour (e.g. to check if it sold too early or held too long). Ensure the environment’s `done` logic prevents any action after termination. Also, support multi-episode training by proper resets. 

# Demand Function

The demand function is the core stochastic engine: it maps the agent’s price $p_t$ (and possibly other factors) to the number of seats sold in period $t$. This can be modelled in many ways, but key considerations are:

- **Price dependence:** *Higher prices yield lower demand.* This inverse relationship (price elasticity) is fundamental. Quantitatively, demand might follow a parametric curve. Common simple models include:
  - **Linear demand:** $D(p)=\max(0, a + b\,p)$. For example, as in one dynamic pricing algorithm: $d(p)=b + a\,p$. Here $a<0$ to reflect negative slope.  
  - **Exponential demand:** $D(p)=A e^{-\beta p}$ for constants $A,\beta>0$.  
  - **Logistic (sigmoid) demand:** $D(p)=\frac{I}{1+\exp(\alpha(p - p_0))}$, scaling with remaining inventory $I$ and some reference $p_0$.  
  - **Constant elasticity:** $D(p)=c\,p^{-\epsilon}$ for elasticity $\epsilon>0$. 

  The chosen form should reflect how sensitively sales drop with price. For instance, in a linear example, revenue-optimal price is $p^*=-b/(2a)$, illustrating how $a,b$ determine elasticity. 

- **Time dependence:** Demand often changes as departure approaches. A typical assumption is that *demand gradually increases over time* (more last-minute bookings). In practice, one could model a baseline demand $\lambda(t)$ that rises as days remaining $T_t$ shrinks. For example, set a higher base arrival rate in later periods. Liang *et al.* refer to “arrival rate” varying with period. If not modelling explicitly, one can simply let the stochastic demand process implicitly capture this by parameter choices (e.g. draw mean demand from a larger value when $T$ is small).

- **Randomness:** To simulate uncertainty, actual sales can be drawn from a probability distribution around the mean demand. Common choices:
  - **Binomial/Multinomial:** If we think of each seat being sold with some probability (given price), then $\text{sales}_t\sim \text{Binomial}(I_t,p_{\text{sale}})$, where $p_{\text{sale}}=f(p_t)$ is decreasing in $p_t$.  
  - **Poisson:** Let $\text{sales}_t\sim\text{Poisson}(\lambda_t)$, where $\lambda_t$ depends on price (and time). A Poisson model is popular for count data and is a natural fit for independent demand arrivals.  
  - **Stochastic noise addition:** Compute a deterministic demand value $d(p_t)$ as above, then add zero-mean noise (e.g. Gaussian or uniform) to represent random fluctuations. 

  For example, one might use 
  $$\lambda_t = \max\bigl(0,\,\alpha - \beta p_t + \gamma g(T_t)\bigr),$$ 
  and then draw $\text{sales}_t\sim\text{Poisson}(\lambda_t)$. Here $\alpha,\beta,\gamma$ are parameters and $g(T)$ is an increasing function of time urgency. 

- **Capacity constraint:** Ensure that if the drawn sales exceed inventory, we cap it: $\text{sales}_t = \min(\text{drawn}, I_t)$. This enforces no overselling. 

- **Demand Variables:** Variables influencing demand can include:
  - *Price ($p_t$):* The key control variable.  
  - *Days to departure:* Captured through a time factor $g(T_t)$ in the model.  
  - *Base trend or seasonality:* Could be a deterministic function or random shift (e.g. higher demand on certain days or flight events).  
  - *Competitor price:* In multi-agent settings, a lower competitor price can steal demand. In a single-agent model, this is omitted.  
  - *Customer classes:* Advanced models may include different customer types with varying willingness to pay. 

- **Mathematical Formulation (Example):** A simple demand function might be:  
  $$\text{expected sales}_t = I_t \times \frac{1}{1+\exp[\alpha(p_t - \theta)]},$$  
  where $\theta$ is a “fair price” and $\alpha>0$ controls sensitivity. Then actual $\text{sales}_t$ is sampled from a Binomial$(I_t,\,1/(1+e^{\alpha(p_t-\theta)}))$. This logistic form ensures a smooth decline in sales as price increases.  

Alternatively, a linear form could be: $d(p_t)=\max(0,\,\beta_0 - \beta_1 p_t)$; if drawn as Poisson($d(p_t)$), we ensure nonnegativity. For instance, Grid Dynamics illustrates using linear demand $d(p)=b+a p$ with revenue-optimal price $-b/(2a)$.  

**Assumptions and Behaviour:** We assume demand is **price-elastic** (negatively correlated with price), and often that it *increases as time runs out* (so unsold tickets become more valuable). Under this model:
- If $p_t$ is high (above most customers’ willingness), $\text{sales}_t$ will often be zero or very low, even early in the horizon.  
- If $p_t$ is low, $\text{sales}_t$ may be high, potentially selling out inventory. 
- Randomness introduces exploration: even a given price may yield slightly different sales each episode, encouraging robust policies.  

**Trade-offs and Best Practices:** The demand model must balance realism and simplicity. Too simple (e.g. fully deterministic demand) may let the agent overfit; too complex (multimodal demand with many factors) makes learning hard. In practice, one chooses a parameterised stochastic demand and tunes parameters so that the agent sees non-trivial trade-offs. 

Calibration: If possible, use historical data to set elasticity. Otherwise, trial and error or domain knowledge can guide parameter choices. For example, setting $\alpha$ so that a 10% price drop yields roughly 5–15% more demand (reflecting moderate elasticity). Checking behaviour under extremes is useful: e.g., if price is zero, demand should max out at inventory (or a little above if one allows overselling in distribution); if price equals a reservation price threshold, demand should approach zero.

**Example Scenario:** Suppose $I=50$ seats remain and $T=30$ days. We choose $p=£50$. Our demand model might be $\text{sales}\sim \text{Binomial}(50,\;p_{\text{sale}})$ with $p_{\text{sale}}=1/(1+\exp[0.1(50-40)])\approx0.38$. We sample that and perhaps get 20 seats sold. Then $I\gets30$, $T\gets29$, and $r=50\times20=1000$. 

**Citing Practice:** Like the ride-sharing example, we incorporate “the physics of pricing” so that elasticity controls demand. Also, time-dependence (e.g. slight demand rise) matches observations in competition settings. As a sanity check, Liang *et al.* note that state includes arrival rates which differ by period, implying demand should vary with time in the simulation.

