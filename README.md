# domain-specific-llm-eval
Domain Specific LLM Agents Evaluation Dynamic Keyword Metric with Human Feedback

# Intro

This metric is part of the auto-eval framework Romantic-Rush:

![alt text](graph/auto-eval-framework.png)

This project aims to deal with the real domain-specific LLM agents response problem, which cannot be solely solve by the LLM based Multi turn metrics (see Fig1). I design a customized metrics for domain specific LLM agents evaluation combining independent contextual keyword & metric gates, reference-based method scoring & reference-free metric alignment; the dynamic approach uses human feedback to fine-tune each gate & their confidence threshold, both applying the active learning method using uncertainty sampling compliance.

My metric structure goes as the follow:

![alt text](graph/base-metric.png)

![alt text](graph/contextual-keyword-gate.png)

![alt text](graph/dynamic-metric.png)


This project is still updating.

To know more of about my projects, see: ([Jason YY Lin](https://a-one-and-a-two.notion.site/Jason-YY-Lin-9c867799194b4c0abf124d55209a5f1e?pvs=4))

###### Fig1 (by RAGAS doc)

![alt text](graph/metric-category.png.png)
