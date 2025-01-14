# CriSPO: Multi-Aspect Critique-Suggestion-guided Automatic Prompt Optimization for Text Generation

Codes for our AAAI 2025 paper "[CriSPO: Multi-Aspect Critique-Suggestion-guided Automatic Prompt Optimization for Text Generation](https://arxiv.org/abs/2410.02748)".


Authors: 
[Han He](https://www.emorynlp.org/doctorates/han-he)\*,
[Qianchu Liu](https://qianchu.github.io/)\*,
[Lei Xu](https://leixx.io/)\* 
(\* equal contribution),
[Chaitanya Shivade](https://cshivade.github.io/),
Yi Zhang, 
Sundararajan Srinivasan, 
Katrin Kirchhoff



## Introduction
Existing automatic prompt engineering methods are typically designed for discriminative tasks, where new task prompts are iteratively refined with limited feedback from a single metric reflecting a single aspect. However, these approaches are suboptimal for generative tasks, which require more nuanced guidance beyond a single numeric metric to improve the prompt and optimize multiple aspects of the generated text. To address these challenges, we propose a novel multi-aspect Critique-Suggestion-guided automatic Prompt Optimization (CriSPO) approach. CriSPO introduces a critique-suggestion module as its core component. This module spontaneously discovers aspects, and compares generated and reference texts across these aspects, providing specific suggestions for prompt modification. These clear critiques and actionable suggestions guide a receptive optimizer module to make more substantial changes, exploring a broader and more effective search space. To further improve CriSPO with multi-metric optimization, we introduce an Automatic Suffix Tuning (AST) extension to enhance the performance of task prompts across multiple metrics. We evaluate CriSPO on 4 state-of-the-art LLMs across 4 summarization and 5 QA datasets. Extensive experiments show 3-4% ROUGE score improvement on summarization and substantial improvement of various metrics on QA.

## Install

```bash
pip install -e .
```

- To run scripts in `experiments` , install with `pip install -e '.[experiments]'`

## Experiments

Experiment scripts to reproduce the results in our paper are located in `experiments`. We provide scripts for the following datasets

- **Text Summarization Datasets**: 
[ACI-Bench clinical note generation](https://github.com/amazon-science/CriSPO/tree/main/experiments/clinical_note_sum),
[CNN/DailyMail](https://github.com/amazon-science/CriSPO/tree/main/experiments/cnn),
[MeetingBank](https://github.com/amazon-science/CriSPO/tree/main/experiments/meeting_bank),
[PubMed](https://github.com/amazon-science/CriSPO/tree/main/experiments/pubmed),
[SamSUM](https://github.com/amazon-science/CriSPO/tree/main/experiments/samsum),

- **QA Datasets**:
[MedMCQA](https://github.com/amazon-science/CriSPO/tree/main/experiments/gsm),
[NarrativeQA](https://github.com/amazon-science/CriSPO/tree/main/experiments/narrativeqa),
[Natural Questions](https://github.com/amazon-science/CriSPO/tree/main/experiments/nq),
[SQuAD](https://github.com/amazon-science/CriSPO/tree/main/experiments/squad)

- **Other Datasets**
[GSM](https://github.com/amazon-science/CriSPO/tree/main/experiments/gsm),
[WebNLG](https://github.com/amazon-science/CriSPO/tree/main/experiments/webnlg)

## Implement a New Task

Use `experiments/summarization` as the starter kit for a new task.

### example.py

**Optionally**, create `example.py` to define a data point of your task as an `Example` subclass. Extend it with any additional data properties other than the default `x` and `y` required by your new task. E.g., you may want to track the ID of a data point.

### task_prompt.py

Create this file to define the task prompt template. The template will be filled up with `x` of each `Example`. It also handles how to parse a LLM generated text to `y`.

### meta_prompt.py

Create this file to define the CriSPO meta prompt template. The template will be filled up with previous prompt-score pairs and few-shot examples. It also handles how to parse a generated text from LLM to a `TaskPrompt`.

This is where you tune the prompt. Feel free to create many versions of meta prompts with different `fill` and `parse` variations. Possible variations to consider:

- Positions of prompt-score pairs and few-shot examples
- Formating of them
- A more specific system prompt that defines what the assistant will do

### metric.py

**Optionally**, if your new task needs a novel metric beyond the metrics in `crispo/metrics`, you can implement one using `crispo/metrics/accuracy.py` as the boilerplate code.

### [experiment].py

Lastly, create an entrypoint script which initiates a `Trainer` that tunes your task prompt on your dataset using your meta prompt. Let's steer clear of using complex config files like `yaml` and stick to simpler Python codes instead. Please give a semanticly meaningful file name to your script when you have multiple experiments.

Please save the trainer outputs like `save_dir='data/experiments/{dataset_name}/{exp_name}'`. You can find useful artifacts there:

- `prompts.json`: every task prompt and their scores sorted decending on the training set or the dev set if provided
- `stepwise_scores.png`: a boxplot of the scores of newly generated task prompts at each step
- `step-xxx`: the input, output, score of task prompts on each training example at each step
- `log.md` the log of this experiment which is more verbosed than the console outputs

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the CC-BY-NC-4.0 License.

