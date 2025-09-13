import pymc as pm
import numpy as np
import pytensor.tensor as pt
from tqdm import tqdm
# ------------------------------
# Toy domain
states = ['light', 'medium', 'heavy']
weights = np.array([2., 5., 9.])
utterances = ['say_light', 'say_medium', 'say_heavy']
x_values = np.array([4., 6., 8.])  # thresholds for 'heavy'

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
        _x = pt.switch(pt.eq(x_idx, 0), x_values[0],
                       pt.switch(pt.eq(x_idx, 1), x_values[1], x_values[2])
                       )
        truth = (s_vals > _x).astype(float)
    return pt.switch(
        pt.gt(truth.sum(), 0),       # condition: truth.sum() > 0
        truth / truth.sum(),         # if True
        pt.ones_like(truth) / truth.shape[0]  # if False
    )


# ------------------------------
# PyMC model
with pm.Model() as rsa_model:
    # Priors
    P_s = pm.Dirichlet('P_s', a=np.ones(len(states)))
    P_x = pm.Dirichlet('P_x', a=np.ones(len(x_values)))

    # Epistemic state prior (toy)
    P_e = np.array([0.4, 0.4, 0.9])

    # Threshold index
    x = pm.Categorical('x', p=P_x)

    # True state index
    s_true = pm.Categorical('s_true', p=P_s)

    p = pt.switch(pt.eq(s_true, 0), P_e[0],
                  pt.switch(pt.eq(s_true, 1), P_e[1], P_e[2])
                  )

    # Epistemic mixture
    is_exact = pm.Bernoulli('is_exact', p=p)

    # Belief
    belief = pt.switch(
        pt.eq(is_exact, 1),
        pt.eye(len(states))[s_true],
        np.array([0.5, 0.5, 0.0])
    )

    # Utility computation
    def compute_utilities(x_idx, belief):
        U = []
        for u_idx in range(len(utterances)):
            L0 = literal_listener(u_idx, x_idx)
            inf = pt.sum(belief * pt.log(L0 + 1e-12))
            soc = pt.sum(belief * pt.dot(L0, V))
            u_val = lambda_info * inf + (1 - lambda_info) * soc - C[u_idx]
            U.append(u_val)
        return pt.as_tensor_variable(U)

    U = pm.Deterministic('U', compute_utilities(x, belief))

    logits = pt.log(Sal + 1e-12) + alpha * U
    u = pm.Categorical('u', logit_p=logits)

    # Sampling
    trace = pm.sample(draws=1000, tune=500, chains=2, target_accept=0.9)

# ------------------------------
# Inspect samples
# import arviz as az
# print(az.summary(trace, var_names=['s_true', 'x', 'u']))
