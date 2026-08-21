"""
generate_data.py
=================
Simulates the data for the case study:

    "Does the new Feed Ranking model (v2) increase Meaningful Social
    Interactions (MSI) without hurting Time Spent or Retention?"
    -- a NorthStar Social (fictional composite of a Meta/Google-style
       social feed product) experimentation case study.

Two datasets are produced:

1. cluster_experiment.csv
   The PRIMARY experiment. Users are grouped into 200 "friend clusters"
   (connected communities within the social graph). Whole clusters are
   randomized to treatment/control (cluster-randomized design), which is
   the standard fix used by Meta/LinkedIn/etc. when a feature can spill
   over between friends (SUTVA violation). Each user is observed daily
   for 21 days: a 7-day PRE period (before launch, used for CUPED
   variance reduction) and a 14-day POST period (feature live).

2. individual_randomization_demo.csv
   A SECOND, smaller synthetic experiment used only to illustrate what
   would have happened if the team had (incorrectly) randomized at the
   individual level instead of the cluster level. It bakes in a
   network-spillover mechanism so the analysis script can quantify the
   contamination bias that cluster randomization was designed to avoid.

All effects (treatment lift, novelty decay, heterogeneity, spillover)
are injected with known ground-truth parameters so the analysis script's
recovered estimates can be checked against them -- exactly like you'd
validate an analysis pipeline with simulation before trusting it on
real production data.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------
# 1. PRIMARY EXPERIMENT: cluster-randomized, 21-day panel
# ---------------------------------------------------------------------

N_CLUSTERS = 200
USERS_PER_CLUSTER = 60
N_USERS = N_CLUSTERS * USERS_PER_CLUSTER
PRE_DAYS = 7
POST_DAYS = 14
TOTAL_DAYS = PRE_DAYS + POST_DAYS

COUNTRIES = np.array(["US", "UK", "IN", "BR"])
COUNTRY_P = np.array([0.35, 0.15, 0.30, 0.20])

SEGMENTS = np.array(["power", "casual", "dormant"])
SEGMENT_P = np.array([0.15, 0.55, 0.30])

# --- cluster-level randomization -------------------------------------
cluster_ids = np.arange(N_CLUSTERS)
cluster_treated = rng.binomial(1, 0.5, size=N_CLUSTERS)  # 50/50 split

# --- user attributes ---------------------------------------------------
user_id = np.arange(N_USERS)
user_cluster = np.repeat(cluster_ids, USERS_PER_CLUSTER)
user_treated = cluster_treated[user_cluster]

user_segment = rng.choice(SEGMENTS, size=N_USERS, p=SEGMENT_P)
user_country = rng.choice(COUNTRIES, size=N_USERS, p=COUNTRY_P)
user_tenure_days = rng.gamma(shape=2.2, scale=260, size=N_USERS).clip(1, 3000).astype(int)

# baseline (untreated, steady-state) daily MSI depends on segment
base_msi_mean = {"power": 14.0, "casual": 5.5, "dormant": 1.4}
base_msi_sd = {"power": 4.0, "casual": 2.2, "dormant": 1.0}
base_time_mean = {"power": 38.0, "casual": 22.0, "dormant": 8.0}
base_time_sd = {"power": 9.0, "casual": 7.0, "dormant": 4.0}

user_base_msi = np.array([rng.normal(base_msi_mean[s], base_msi_sd[s]) for s in user_segment]).clip(0)
user_base_time = np.array([rng.normal(base_time_mean[s], base_time_sd[s]) for s in user_segment]).clip(0)

# a user-level random effect (persistent individual "flavor") so pre-period
# is correlated with post-period -> this is what makes CUPED useful
user_re_msi = rng.normal(0, 1.6, size=N_USERS)
user_re_time = rng.normal(0, 4.0, size=N_USERS)

# a CLUSTER-level shared shock (friend groups share regional trends, group
# culture, correlated news events, etc.) -> this is what creates the
# intra-cluster correlation that makes naive (non-cluster-robust) SEs wrong.
cluster_re_msi = rng.normal(0, 1.9, size=N_CLUSTERS)
cluster_re_time = rng.normal(0, 5.5, size=N_CLUSTERS)
user_cluster_re_msi = cluster_re_msi[user_cluster]
user_cluster_re_time = cluster_re_time[user_cluster]

# ground-truth treatment effect (per-day), by segment, at "steady state"
# (this is the effect once novelty has fully worn off)
steady_state_lift = {"power": 3.2, "casual": 0.9, "dormant": -0.1}
# novelty multiplier on day 1 of treatment vs steady state (novelty bump)
novelty_bump_mult = 2.6
# exponential decay time-constant (days) for novelty wearing off
decay_tau = 4.5

# time-spent trade-off: treatment nudges people from passive scrolling
# toward active interactions -> small negative effect on time spent,
# proportionally larger for power users (they are the ones interacting more)
time_spent_effect = {"power": -2.6, "casual": -0.8, "dormant": -0.1}

rows = []
for day in range(TOTAL_DAYS):
    is_post = day >= PRE_DAYS
    post_day_idx = day - PRE_DAYS  # 0..13 during post period

    if is_post:
        decay_factor = np.exp(-post_day_idx / decay_tau)  # 1 -> 0 over post period
    else:
        decay_factor = 0.0

    day_noise_msi = rng.normal(0, 1.3, size=N_USERS)
    day_noise_time = rng.normal(0, 5.5, size=N_USERS)

    seg_steady = np.array([steady_state_lift[s] for s in user_segment])
    seg_time_eff = np.array([time_spent_effect[s] for s in user_segment])

    treat_effect_msi = np.where(
        user_treated & is_post,
        seg_steady * (1 + (novelty_bump_mult - 1) * decay_factor),
        0.0,
    )
    treat_effect_time = np.where(user_treated & is_post, seg_time_eff, 0.0)

    # mild weekly seasonality (weekend bump), shared by everyone
    weekday = day % 7
    weekend_bump_msi = 0.6 if weekday >= 5 else 0.0
    weekend_bump_time = 3.0 if weekday >= 5 else 0.0

    msi = (user_base_msi + user_re_msi + user_cluster_re_msi + weekend_bump_msi
           + treat_effect_msi + day_noise_msi).clip(0)
    time_spent = (user_base_time + user_re_time + user_cluster_re_time + weekend_bump_time
                  + treat_effect_time + day_noise_time).clip(0)

    # next-day return probability, mildly boosted by treatment for
    # power/casual (more meaningful interactions -> stickier) and
    # mildly hurt for dormant users (algorithm change is confusing them)
    base_return_p = {"power": 0.93, "casual": 0.74, "dormant": 0.34}
    ret_seg = np.array([base_return_p[s] for s in user_segment])
    ret_treat_adj = np.where(
        user_treated & is_post,
        np.where(user_segment == "dormant", -0.02, 0.015) * (0.4 + 0.6 * decay_factor if is_post else 0),
        0.0,
    )
    return_p = np.clip(ret_seg + ret_treat_adj, 0.01, 0.99)
    returned_next_day = rng.binomial(1, return_p)

    rows.append(pd.DataFrame({
        "user_id": user_id,
        "cluster_id": user_cluster,
        "day": day,
        "period": np.where(is_post, "post", "pre"),
        "treatment": user_treated,
        "segment": user_segment,
        "country": user_country,
        "tenure_days": user_tenure_days,
        "msi_score": msi.round(2),
        "time_spent_min": time_spent.round(2),
        "returned_next_day": returned_next_day,
    }))

experiment_df = pd.concat(rows, ignore_index=True)
experiment_df.to_csv("submission/data/cluster_experiment.csv", index=False)
print(f"cluster_experiment.csv -> {experiment_df.shape[0]:,} rows, "
      f"{experiment_df['user_id'].nunique():,} users, {N_CLUSTERS} clusters")

# ---------------------------------------------------------------------
# 2. DEMO: naive INDIVIDUAL-level randomization with network spillover
#    (used only to demonstrate contamination bias in the analysis)
# ---------------------------------------------------------------------

N_USERS_2 = 6000
N_CLUSTERS_2 = 150
USERS_PER_CLUSTER_2 = N_USERS_2 // N_CLUSTERS_2

u_cluster2 = np.repeat(np.arange(N_CLUSTERS_2), USERS_PER_CLUSTER_2)
u_id2 = np.arange(N_USERS_2)
u_segment2 = rng.choice(SEGMENTS, size=N_USERS_2, p=SEGMENT_P)

# INDIVIDUAL-level (not cluster-level) coin flip -> friends are a mix
u_treated2 = rng.binomial(1, 0.5, size=N_USERS_2)

# fraction of each user's own cluster that is treated (their "friends")
cluster_treat_share = pd.Series(u_treated2).groupby(u_cluster2).transform("mean").values

base_msi2 = np.array([rng.normal(base_msi_mean[s], base_msi_sd[s]) for s in u_segment2]).clip(0)
seg_steady2 = np.array([steady_state_lift[s] for s in u_segment2])

# TRUE direct effect only applies to users who are themselves treated.
direct_effect = np.where(u_treated2 == 1, seg_steady2, 0.0)

# SPILLOVER: a control user surrounded by treated friends still picks up
# part of the effect (e.g., they see more comments/shares in their feed
# from treated friends, prompting them to engage more too).
# spillover_strength = 0 -> no interference (would make individual RCT valid)
# spillover_strength = 1 -> full interference (control indistinguishable from treated)
SPILLOVER_STRENGTH = 0.55
spillover_effect = np.where(
    u_treated2 == 0,
    SPILLOVER_STRENGTH * seg_steady2 * cluster_treat_share,
    0.0,
)

noise2 = rng.normal(0, 1.5, size=N_USERS_2)
post_msi2 = (base_msi2 + direct_effect + spillover_effect + noise2).clip(0)
pre_msi2 = (base_msi2 + rng.normal(0, 1.5, size=N_USERS_2)).clip(0)

demo_df = pd.DataFrame({
    "user_id": u_id2,
    "cluster_id": u_cluster2,
    "segment": u_segment2,
    "treatment": u_treated2,
    "cluster_treated_share": cluster_treat_share.round(3),
    "pre_period_msi": pre_msi2.round(2),
    "post_period_msi": post_msi2.round(2),
})
demo_df.to_csv("submission/data/individual_randomization_demo.csv", index=False)
print(f"individual_randomization_demo.csv -> {demo_df.shape[0]:,} rows, "
      f"{N_CLUSTERS_2} clusters, true steady-state lift per segment = {steady_state_lift}")

# Save ground truth for validation / grading of the analysis script
ground_truth = {
    "steady_state_lift_msi": steady_state_lift,
    "novelty_bump_multiplier": novelty_bump_mult,
    "novelty_decay_tau_days": decay_tau,
    "time_spent_effect": time_spent_effect,
    "spillover_strength_demo_dataset": SPILLOVER_STRENGTH,
}
pd.Series(ground_truth).to_json("submission/data/ground_truth_params.json")
print("Saved ground_truth_params.json (for validating the analysis pipeline)")
