import arviz as az
import pymc as pm
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import matplotlib.pyplot as plt

# ------------------------------
# Toy domain
states = ["light", "medium", "heavy"]
weights = np.array([2.0, 5.0, 9.0])
utterances = ["say_light", "say_medium", "say_heavy"]
x_values = np.array([4.0, 6.0, 8.0])  # thresholds for 'heavy'

# Salience and cost (toy)
Sal = np.array([0.4, 0.35, 0.25])
C = np.array([0.1, 0.15, 0.2])
V = np.array([1.0, 0.6, 0.2])

alpha = 1.5
lambda_info = 0.7


# ------------------------------
# Literal semantics function
def literal_listener(u_idx, x_idx):
    s_vals = weights
    if u_idx == 0:  # say_light
        truth = (s_vals <= 4).astype(float)
    elif u_idx == 1:  # say_medium
        truth = ((s_vals > 4) & (s_vals <= 7)).astype(float)
    else:  # say_heavy
        _x = pt.switch(
            pt.eq(x_idx, 0),
            x_values[0],
            pt.switch(pt.eq(x_idx, 1), x_values[1], x_values[2]),
        )
        truth = (s_vals > _x).astype(float)
    return pt.switch(
        pt.gt(truth.sum(), 0),  # condition: truth.sum() > 0
        truth / truth.sum(),  # if True
        pt.ones_like(truth) / truth.shape[0],  # if False
    )


# ------------------------------
# PyMC model
with pm.Model() as rsa_model:
    # Priors
    P_s = pm.Dirichlet("P_s", a=np.ones(len(states)))
    P_x = pm.Dirichlet("P_x", a=np.ones(len(x_values)))

    # Epistemic state prior (toy)
    P_e = np.array([0.4, 0.4, 0.9])

    # Threshold index
    x = pm.Categorical("x", p=P_x)

    # True state index
    s_true = pm.Categorical("s_true", p=P_s)

    # Epistemic mixture
    p = pt.switch(pt.eq(s_true, 0), P_e[0], pt.switch(pt.eq(s_true, 1), P_e[1], P_e[2]))
    is_exact = pm.Bernoulli("is_exact", p=p)

    # Belief
    belief = pt.switch(
        pt.eq(is_exact, 1), pt.eye(len(states))[s_true], np.array([0.5, 0.5, 0.0])
    )

    # Utility computation
    def compute_utilities(x_idx, belief):
        U = []
        for u_idx in range(len(utterances)):
            L0 = literal_listener(u_idx, x_idx)
            inf = pt.sum(belief * pt.log(L0 + 1e-12))
            soc = pt.sum(belief * pt.dot(L0, V))
            u_val = lambda_info * inf + (1 - lambda_info) * soc
            U.append(u_val)
        return pt.as_tensor_variable(U)

    U = pm.Deterministic("U", compute_utilities(x, belief))

    logits = pt.log(Sal + 1e-12) + alpha * U - C
    u = pm.Categorical("u", logit_p=logits)

    # Sampling
    trace = pm.sample(
        draws=8000, tune=2000, chains=4, target_accept=0.95, random_seed=42
    )

# ------------------------------
# Inspect samples
print(az.summary(trace, var_names=["s_true", "x", "u"]))
# plot = az.plot_trace(trace, var_names=["s_true", "x", "u"])
# plt.savefig("trace_plot.png", dpi=300, bbox_inches='tight')
# plt.close()  # optional: close to free memory

# ------------------------------
# Compute P_S1(u | s)

# Extract samples
s_samples = trace.posterior["s_true"].stack(draws=("chain", "draw")).values
u_samples = trace.posterior["u"].stack(draws=("chain", "draw")).values

# Compute P_S1(u|s)
P_S1 = np.zeros((len(states), len(utterances)))

for s_idx in range(len(states)):
    mask = s_samples == s_idx  # samples with this state
    u_given_s = u_samples[mask]
    # Count occurrences of each utterance
    for u_idx in range(len(utterances)):
        P_S1[s_idx, u_idx] = np.sum(u_given_s == u_idx) / len(u_given_s)

# Pretty print as DataFrame
df_P_S1 = pd.DataFrame(P_S1, index=states, columns=utterances)
print("======================================")
print("Posterior P_S1(u | s):")
print(df_P_S1)

# ------------------------------
# Compute P_L1(s,x | u='say_heavy')

# Extract samples
s_samples = trace.posterior["s_true"].stack(draws=("chain", "draw")).values
x_samples = trace.posterior["x"].stack(draws=("chain", "draw")).values
u_samples = trace.posterior["u"].stack(draws=("chain", "draw")).values

# Utterance of interest
u_idx = utterances.index("say_heavy")

# Filter samples where u == 'say_heavy'
mask = u_samples == u_idx
s_given_u = s_samples[mask]
x_given_u = x_samples[mask]

# Compute empirical joint distribution P(s, x | u='say_heavy')
joint_counts = np.zeros((len(states), len(x_values)))
for s_val, x_val in zip(s_given_u, x_given_u):
    joint_counts[s_val, x_val] += 1

P_L1 = joint_counts / joint_counts.sum()

# Pretty print as DataFrame
df_P_L1 = pd.DataFrame(P_L1, index=states, columns=[f"x={i}" for i in x_values])
print("======================================")
print("Posterior P_L1(s, x | u='say_heavy'):")
print(df_P_L1)


# ------------------------------
# Compute P_L1(s,x | u='medium')

# Utterance of interest
u_idx = utterances.index("say_medium")
# Filter samples where u == 'say_medium'
mask = u_samples == u_idx
s_given_u = s_samples[mask]
x_given_u = x_samples[mask]

# Compute empirical joint distribution P(s, x | u='say_medium')
joint_counts = np.zeros((len(states), len(x_values)))
for s_val, x_val in zip(s_given_u, x_given_u):
    joint_counts[s_val, x_val] += 1
P_L1 = joint_counts / joint_counts.sum()
# Pretty print as DataFrame
df_P_L1 = pd.DataFrame(P_L1, index=states, columns=[f"x={i}" for i in x_values])
print("======================================")
print("Posterior P_L1(s, x | u='say_medium'):")
print(df_P_L1)

# ------------------------------
# Compute P_L1(s,x | u='light')

# Utterance of interest
u_idx = utterances.index("say_light")
# Filter samples where u == 'say_light'
mask = u_samples == u_idx
s_given_u = s_samples[mask]
x_given_u = x_samples[mask]
# Compute empirical joint distribution P(s, x | u='say_light')
joint_counts = np.zeros((len(states), len(x_values)))
for s_val, x_val in zip(s_given_u, x_given_u):
    joint_counts[s_val, x_val] += 1
P_L1 = joint_counts / joint_counts.sum()
# Pretty print as DataFrame
df_P_L1 = pd.DataFrame(P_L1, index=states, columns=[f"x={i}" for i in x_values])
print("======================================")
print("Posterior P_L1(s, x | u='say_light'):")
print(df_P_L1)
