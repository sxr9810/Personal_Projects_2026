# 🔎 Sanctions Screening & Fuzzy Matching

**NLP | Entity Resolution | Fuzzy Matching | Model Evaluation | Python**

## Situation

Sanctions screening systems cannot rely on exact name matching.

A legitimate payment message might contain **"Mohammed Al Rashid"**, while a sanctions list contains **"Muhammad Rashid"**. Typos, missing name tokens, reordered names, transliteration differences, and honorifics can cause exact matching systems to miss potential sanctions hits.

At the same time, overly aggressive matching can generate excessive false positives and create unnecessary analyst workload.

The goal of this project was therefore not simply to determine whether two names match, but to quantify **how strong the evidence is that two names represent the same entity**.

## Task

Build a data-driven sanctions screening pipeline that:

* Handles noisy and inconsistent entity names
* Reduces the number of candidate comparisons
* Combines multiple similarity signals
* Produces an explainable composite matching score
* Evaluates the ability to separate sanctions hits from clean transactions
* Identifies an appropriate screening threshold while considering operational tradeoffs

## Data

The reference data was sourced from **OpenSanctions**, including entity, name, and sanctions information.

A controlled synthetic transaction dataset was created for evaluation.

### Dataset

* **1.29M** entities in the target dataset
* **3.07M** name variants available
* **2.13M** sanctions target name variants
* **10,000** synthetic transactions screened
* **9,800** clean synthetic transactions
* **200** labeled sanctions hits
* **2%** initial injected hit rate

To simulate realistic screening conditions, sanctioned names were deliberately perturbed using:

* Character-level typos
* First/last name reordering
* Missing name tokens
* Common transliteration variations
* Titles and honorifics

## Action

### 1. Normalize

Transaction and reference names were standardized by:

* Converting text to lowercase
* Removing punctuation
* Removing honorifics
* Standardizing whitespace

### 2. Block

Rather than comparing every transaction against every reference entity, **Soundex-based blocking** was applied using the first name token to reduce the candidate search space.

### 3. Generate Similarity Signals

Five complementary similarity measures were calculated:

| Similarity Signal | Weight | Purpose                           |
| ----------------- | -----: | --------------------------------- |
| Levenshtein ratio |    25% | Character-level similarity        |
| Token-set ratio   |    25% | Robustness to token differences   |
| Token-sort ratio  |    20% | Robustness to reordered names     |
| Jaro-Winkler      |    20% | Short names and prefix similarity |
| Phonetic matching |    10% | Sound-based similarity            |

These signals were combined into a single **composite similarity score**.

### 4. Evaluate

The composite score was evaluated using:

* Precision
* Recall
* F1-score
* Accuracy
* AUC
* KS statistic
* Alert-volume / operational metrics

Threshold analysis was used to understand the tradeoff between sanctions detection and false-positive workload.

## Results

The composite score showed clear separation between sanctions hits and clean transactions.

| Metric                             |           Result |
| ---------------------------------- | ---------------: |
| Average score — true hits          |        **94.58** |
| Average score — clean transactions |        **83.67** |
| Score separation gap               | **10.91 points** |
| Selected threshold                 |         **0.85** |
| Recall                             |         **100%** |
| Precision                          |          **80%** |
| Accuracy                           |          **85%** |

The threshold of **0.85** was selected based on maximizing F1.

At this threshold, the model captured **100% of the labeled sanctions hits** while achieving **80% precision**.

## Feature Analysis

Individual similarity signals were also evaluated to understand their discriminatory power.

* **Token-sort similarity:** AUC **0.9808**
* **Levenshtein ratio:** AUC **0.9787**
* **Jaro-Winkler:** AUC **0.9380**
* **Phonetic matching:** AUC **0.50**

The results demonstrate why combining multiple signals can be more effective than relying on a single matching technique.

Interestingly, phonetic matching alone provided little discriminatory power in this dataset, despite being useful as a complementary signal within the overall framework.

## Tradeoffs

Threshold selection is not simply a statistical optimization problem.

A lower threshold can improve detection and recall but may generate a large number of false alerts.

A higher threshold can reduce false positives and analyst workload but increases the risk of missing potential sanctions matches.

Therefore, an effective screening threshold needs to balance:

**Detection performance ↔ False-positive volume ↔ Analyst workload ↔ Missed-sanctions risk**

## Key Takeaway

The key lesson from this project is that **good sanctions screening is not simply about finding similar names**.

The challenge is determining:

* Which similarity signals matter
* How much weight each signal should receive
* How much evidence is sufficient to trigger an alert
* What the operational cost of false positives is
* What the risk of missed matches looks like

The project demonstrates an end-to-end approach to entity resolution: starting with messy names, engineering complementary similarity signals, reducing the search space through blocking, creating an explainable composite score, and evaluating the model from both a statistical and operational perspective.

## Next Steps

A production-grade screening system could be extended by incorporating additional entity attributes such as:

* Date of birth
* Nationality
* Address
* Identifiers
* Country information

Thresholds could then be calibrated using **real investigator outcomes and historical alert dispositions** rather than synthetic labels alone.

## Technologies

**Python · Pandas · RapidFuzz · Jellyfish · Scikit-learn · Statistical Testing · NLP · Fuzzy Matching**
