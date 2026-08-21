1. LLM Experimentation & Context Engineering
This could build directly on your existing experimentation work.
Research:
Does giving an LLM more context improve the quality of business decisions?
Create four experimental conditions:
Experiment 1
Prompt only
Experiment 2
Prompt + raw data
Experiment 3
Prompt + summarized data
Experiment 4
Prompt + structured analytical context
Then evaluate:
Accuracy
Decision quality
Hallucinations
Consistency
Business utility
Token cost
You could even conduct an A/B test of prompting strategies.
Your final result might be:
Adding structured analytical context improved decision accuracy by 14%, but additional raw context produced no statistically significant improvement.

2. LLM-as-a-Judge
This is the most intellectually interesting one.
Suppose you have 500 business questions and answers.
Have:
Human evaluator
GPT evaluator
Claude evaluator
Rule-based evaluator
score the responses.
Then test:
Agreement
Human vs GPT
Human vs Claude
GPT vs Claude
Using:
Cohen's Kappa
Spearman correlation
Agreement rate
Then investigate evaluation bias.
For example:
Does the LLM judge prefer longer responses?
Does confidence influence perceived quality?
Does changing formatting change the score?
Does the judge rank two equivalent answers differently?
Your research question:
Can LLM-based evaluation be treated as a reliable measurement system?
This is a very strong data science + GenAI project.

