# Fixed version (removed premature P_s_given_e). Re-run the demo.
# import caas_jupyter_tools as tools
import math
import pandas as pd
import numpy as np

states = ['light(2)', 'medium(5)', 'heavy(9)']
weights = {'light(2)': 2, 'medium(5)': 5, 'heavy(9)': 9}
P_s = {s: 1/3 for s in states}
x_values = [4, 6, 8]
P_x = {x: 1/len(x_values) for x in x_values}
utterances = ['say_light', 'say_heavy', 'say_polite']


def truth(u, s, x):
    w = weights[s]
    if u == 'say_light':
        return 1.0 if w <= 4 else 0.0
    if u == 'say_heavy':
        return 1.0 if w > x else 0.0
    if u == 'say_polite':
        return 1.0
    return 0.0


Sal = {'say_light': 0.4, 'say_heavy': 0.35, 'say_polite': 0.25}
C = {'say_light': 0.1, 'say_heavy': 0.2, 'say_polite': 0.05}
V = {'light(2)': 1.0, 'medium(5)': 0.6, 'heavy(9)': 0.2}

alpha = 1.5
info_weight = 0.7

P_s_given_e_amb = {'light(2)': 0.5, 'medium(5)': 0.5, 'heavy(9)': 0.0}
P_e_given_s = {'light(2)': 0.4, 'medium(5)': 0.4, 'heavy(9)': 0.9}


def literal_listener(u, x):
    numerators = {s: truth(u, s, x) * P_s[s] for s in states}
    Z = sum(numerators.values())
    if Z == 0:
        return {s: 1/len(states) for s in states}
    return {s: numerators[s]/Z for s in states}


def speaker_belief(e, true_state_override=None):
    if e == 'ambiguous':
        return dict(P_s_given_e_amb)
    else:
        if true_state_override is None:
            raise ValueError(
                "Need true_state_override for exact speaker belief")
        return {s: (1.0 if s == true_state_override else 0.0) for s in states}


def pragmatic_speaker_given_true_s(s_true):
    numerators = {u: 0.0 for u in utterances}
    utilities_by_e = {}
    for e in ['exact', 'ambiguous']:
        def speaker_belief_local(e_local=e):
            return speaker_belief(e_local, true_state_override=s_true) if e_local == 'exact' else speaker_belief(e_local)
        utilities = {}
        for u in utterances:
            inf_term = 0.0
            for s in states:
                p_s_e = speaker_belief_local()[s]
                expected_log = 0.0
                for x in x_values:
                    L0 = literal_listener(u, x)
                    p = max(L0[s], 1e-12)
                    expected_log += P_x[x] * math.log(p)
                inf_term += p_s_e * expected_log
            soc_term = 0.0
            for s in states:
                p_s_e = speaker_belief_local()[s]
                expected_V = 0.0
                for x in x_values:
                    L0 = literal_listener(u, x)
                    ev = sum(L0[s2]*V[s2] for s2 in states)
                    expected_V += P_x[x] * ev
                soc_term += p_s_e * expected_V
            utilities[u] = info_weight * inf_term + \
                (1 - info_weight) * soc_term
        numerators_e = {
            u: Sal[u] * math.exp(alpha * utilities[u] - C[u]) for u in utterances}
        Z_e = sum(numerators_e.values())
        S1_e = {u: numerators_e[u]/Z_e for u in utterances}
        utilities_by_e[e] = utilities
        for u in utterances:
            numerators[u] += S1_e[u] * P_e_given_s[s_true]
    Z = sum(numerators.values())
    return {u: numerators[u]/Z for u in utterances}, utilities_by_e


def pragmatic_listener_L1(u):
    S1_by_s = {}
    utilities_by_s = {}
    for s in states:
        S1_by_s[s], utilities_by_s[s] = pragmatic_speaker_given_true_s(s)
    numerators = {}
    for s in states:
        for x in x_values:
            numerators[(s, x)] = S1_by_s[s][u] * P_s[s] * P_x[x]
    Z = sum(numerators.values())
    return {k: v/Z for k, v in numerators.items()}, S1_by_s, utilities_by_s


rows = []
for s in states:
    S1_dist, util_by_e = pragmatic_speaker_given_true_s(s)
    for u in utterances:
        rows.append({'true_state': s, 'utterance': u, 'P_S1(u|s)': S1_dist[u]})
df_speaker = pd.DataFrame(rows)

L1_dist, S1_by_s, utils_by_s = pragmatic_listener_L1('say_heavy')
L1_table = pd.DataFrame(
    [{'state': k[0], 'x_threshold': k[1], 'P_L1': v} for k, v in L1_dist.items()])

print("Speaker choices P_S1(u|s)", df_speaker)
util_rows = []
for s in states:
    for e, utils in utils_by_s[s].items():
        for u, val in utils.items():
            util_rows.append(
                {'true_state': s, 'epistemic_state': e, 'utterance': u, 'utility': val})
df_utils = pd.DataFrame(util_rows)
print("Speaker utility components (by true state & epistemic e)", df_utils)
L1_table_sorted = L1_table.sort_values(
    'P_L1', ascending=False).reset_index(drop=True)
print("Listener posterior P_L1(s,x | u='say_heavy')", L1_table_sorted)

summary = f"Summary:\n- Speaker choice distributions P_S1(u|s) shown in table 'Speaker choices P_S1(u|s)'.\n" \
          f"- Listener posterior P_L1(s,x | u='say_heavy') shown in 'Listener posterior P_L1(s,x | u=\\'say_heavy\\')'.\n" \
          f"- The model integrates: meaning-inference via thresholds x, epistemic uncertainty via P(e|s) and P(s|e),\n" \
          f"  complex utility mixing informativeness & social value (info_weight={info_weight}), and utterance salience.\n"
print(summary)
