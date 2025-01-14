# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import abc
import os
import random
import statistics
from collections import defaultdict
from typing import List, Sequence, Union, Any, Literal, Set, Optional, Dict, Tuple

import pandas as pd
from crispo.llms import LargeLanguageModel
from tqdm import tqdm

from crispo.metrics.floats import FloatDict
from crispo.metrics.metric import Metric, MetricDict
from crispo.optimizer.meta_prompt import MetaPrompt
from crispo.task.critique import CritiquePrompt
from crispo.task.example import Example
from crispo.task.prompt import TaskPrompt
from crispo.utilities.log_util import init_logger
from crispo.utilities.plot_util import boxplot
from crispo.utilities.time_util import CountdownTimer


class Trainer(abc.ABC):
    def __init__(self, save_dir: str) -> None:
        super().__init__()
        self.logger = init_logger(name="log.md", save_dir=save_dir, mode="a")
        self.save_dir = save_dir

    def fit(
        self,
        train: List[Example],
        dev: Optional[List[Example]],
        initial_task_prompts: Set[TaskPrompt],
        meta_prompt: MetaPrompt,
        meta_llm: LargeLanguageModel,
        metric: Metric,
        task_llm: LargeLanguageModel = None,
        critique_prompt: CritiquePrompt = None,
        num_examples_in_critique=10,
        max_score_of_example_in_critique=None,
        critique_example_selection_criteria: Literal[
            "random", "lowest_score"
        ] = "random",
        num_search_steps=200,
        num_new_prompts_in_each_step=8,
        few_shot_selection_criteria: Literal[
            "random", "current_lowest_score", "accumulative_lowest_score"
        ] = "current_lowest_score",
        num_few_shot_examples_in_meta_prompt=3,
        num_task_prompts_in_meta_prompt=20,
        dev_evaluation_per_n_steps=5,
        dev_evaluation_threshold=None,
        max_retry=5,
        refresh_meta_prompt_each_attempt=True,
        save_every_n_steps=1,
    ) -> (Dict[TaskPrompt, float], str, float):
        timer = CountdownTimer(num_search_steps + 1)
        prompt_score_pairs = dict()
        prompt_score_pairs_dev = dict()
        best_prompt = None
        best_prompt_dev, best_prompt_dev_step = None, 0
        prompt_steps = dict()
        accumulative_few_shot_example_scores = [[] for _ in range(len(train))]
        stepwise_scores = {"train": [], "dev": []}
        critiques = defaultdict(str)

        if task_llm is None:
            task_llm = meta_llm

        new_prompts = dict.fromkeys(initial_task_prompts)

        for step in range(num_search_steps + 1):
            self.logger.info(f"[yellow]## Step {step}/{num_search_steps}[/yellow]")
            step_save_dir = f"{self.save_dir}/step-{step:03}"
            os.makedirs(step_save_dir, exist_ok=True)
            step_few_shot_example_scores = [[] for _ in range(len(train))]
            stepwise_scores["train"].append([])
            stepwise_scores["dev"].append([])

            for pi, task_prompt in enumerate(new_prompts):
                # train
                overall_score, scores, predictions = self.evaluate(
                    task_prompt=task_prompt,
                    dataset=train,
                    task_llm=task_llm,
                    metric=metric,
                    desc=f"Evaluating {pi + 1}/{len(new_prompts)} prompts on train",
                    save_path=f"{step_save_dir}/train-prompt-{pi + 1:03}.csv",
                    # we use dev set to select in context examples for train. This argument will be used with Trainer_fewshot class.
                    prompt_score_pairs=prompt_score_pairs,
                )

                prompt_score_pairs[task_prompt] = overall_score
                prompt_steps[task_prompt] = step
                stepwise_scores["train"][-1].append(overall_score)
                for i, score in enumerate(scores):
                    step_few_shot_example_scores[i].append(score)
                    accumulative_few_shot_example_scores[i].append(score)

                # report in log
                if (
                    best_prompt is None
                    or metric.key(overall_score) > prompt_score_pairs[best_prompt]
                ):
                    best_score = overall_score
                    best_prompt = task_prompt
                    msg = (
                        f"New best prompt `[light_magenta]{str(best_prompt)}[/light_magenta]` score "
                        f"on train: [light_cyan]{overall_score:.4f}[/light_cyan]"
                    )
                else:
                    msg = (
                        f"`{task_prompt.short_str()}` score on train: [light_cyan]{overall_score:.4f}"
                        f"[/light_cyan], worse than the best score [red]{best_score:.4f}"
                        f"[/red] at step [light_yellow]{prompt_steps[best_prompt]}[/light_yellow]"
                    )
                self.logger.info(msg)

                # critique
                if critique_prompt:
                    critiques = self.update_critique(
                        scores=scores,
                        critique_prompt=critique_prompt,
                        task_prompt=task_prompt,
                        predictions=predictions,
                        train=train,
                        critiques=critiques,
                        critique_example_selection_criteria=critique_example_selection_criteria,
                        max_score_of_example_in_critique=max_score_of_example_in_critique,
                        meta_llm=meta_llm,
                        num_examples_in_critique=num_examples_in_critique,
                        metric=metric,
                    )

            # collect the prompt history
            prompt_score_pairs = self.sort_prompts_best_to_worst(
                prompt_score_pairs, metric
            )
            best_prompt, best_score = list(prompt_score_pairs.items())[0]

            timer.log(
                f"Best prompt on train `[light_magenta]{best_prompt.short_str()}[/light_magenta]` "
                f"scored [light_red]{best_score:.4f}[/light_red] at "
                f"step [light_yellow]{prompt_steps[best_prompt]}[/light_yellow]",
                logger=self.logger,
                newline=True,
                ratio_percentage=False,
                ratio=False,
            )

            # run evaluate on best train set prompt
            if step % dev_evaluation_per_n_steps == 0 and dev:
                prompts_to_eval_dev = new_prompts
                if best_prompt not in prompt_score_pairs_dev:
                    prompts_to_eval_dev[best_prompt] = None

                for pi, prompt in enumerate(prompts_to_eval_dev):
                    if dev_evaluation_threshold is not None and metric.key(
                        prompt_score_pairs[prompt]
                    ) < metric.key(dev_evaluation_threshold):
                        continue
                    overall_score_dev, _, _ = self.evaluate(
                        task_prompt=prompt,
                        dataset=dev,
                        task_llm=task_llm,
                        metric=metric,
                        desc=f"Evaluating {pi + 1}/{len(prompts_to_eval_dev)} prompts on dev",
                        save_path=f"{step_save_dir}/dev-prompt-{pi + 1:03}.csv",
                        # we use train set to select in context examples for dev. This argument will be used with Trainer_fewshot class.
                        prompt_score_pairs=prompt_score_pairs_dev,
                    )

                    stepwise_scores["dev"][-1].append(overall_score_dev)
                    prompt_score_pairs_dev[prompt] = overall_score_dev
                    prompt_score_pairs_dev = self.sort_prompts_best_to_worst(
                        prompt_score_pairs_dev, metric
                    )
                    if (
                        best_prompt_dev is None
                        or metric.key(overall_score_dev)
                        > prompt_score_pairs_dev[best_prompt_dev]
                    ):
                        best_prompt_score_dev, best_prompt_dev, best_prompt_dev_step = (
                            overall_score_dev,
                            prompt,
                            step,
                        )
                        msg = (
                            f"New best prompt `[light_magenta]{str(best_prompt_dev)}[/light_magenta]` score "
                            f"on dev: [light_cyan]{overall_score_dev:.4f}[/light_cyan]"
                        )
                    else:
                        msg = (
                            f"`{prompt.short_str()}` score on dev: [light_cyan]{overall_score_dev:.4f}"
                            f"[/light_cyan], worse than the best score [red]{best_prompt_score_dev:.4f}"
                            f"[/red] at step [light_yellow]{best_prompt_dev_step}[/light_yellow]"
                        )
                    self.logger.info(msg)

            if step == num_search_steps:
                break
            # self.logger.debug('### New Prompts\n')
            # self.logger.debug(self.convert_prompt_scores_to_df(
            #     dict((k, v) for k, v in prompt_score_pairs.items() if k in new_prompts), prompt_steps).to_markdown(
            #     index=False) + '\n')

            # Generate new task prompts
            new_prompts = self.generating_new_prompts(
                meta_prompt=meta_prompt,
                prompt_score_pairs=prompt_score_pairs,
                train=train,
                step_few_shot_example_scores=step_few_shot_example_scores,
                accumulative_few_shot_example_scores=accumulative_few_shot_example_scores,
                critiques=critiques,
                step_save_dir=step_save_dir,
                meta_llm=meta_llm,
                metric=metric,
                few_shot_selection_criteria=few_shot_selection_criteria,
                num_few_shot_examples_in_meta_prompt=num_few_shot_examples_in_meta_prompt,
                num_task_prompts_in_meta_prompt=num_task_prompts_in_meta_prompt,
                num_new_prompts_in_each_step=num_new_prompts_in_each_step,
                refresh_meta_prompt_each_attempt=refresh_meta_prompt_each_attempt,
                max_retry=max_retry,
            )

            # save every n steps
            if (step + 1) % save_every_n_steps == 0:
                # self.logger.debug('### Final Prompts\n')
                prompts_df = self.convert_prompt_scores_to_df(
                    prompt_score_pairs, prompt_steps
                )
                # self.logger.debug(prompts_df.to_markdown(index=False) + '\n')
                prompts_df.to_json(
                    f"{self.save_dir}/prompts.json", indent=2, orient="records"
                )

                prompts_df_dev = self.convert_prompt_scores_to_df(
                    prompt_score_pairs_dev, prompt_steps
                )
                # self.logger.debug(prompts_df_dev.to_markdown(index=False) + '\n')
                prompts_df_dev.to_json(
                    f"{self.save_dir}/prompts_dev.json", indent=2, orient="records"
                )

                for _split, _scores in stepwise_scores.items():
                    fig = boxplot(
                        _scores, xlabel="Step", ylabel=metric.__class__.__name__
                    )
                    fig.savefig(f"{self.save_dir}/stepwise_scores_{_split}.png")

        return prompt_score_pairs, best_prompt_dev

    def update_critique(
        self,
        scores,
        critique_prompt,
        task_prompt,
        predictions,
        train,
        critiques,
        critique_example_selection_criteria,
        max_score_of_example_in_critique,
        meta_llm,
        num_examples_in_critique,
        metric,
    ):
        if critique_example_selection_criteria == "random":
            indices = list(range(len(scores)))
            random.shuffle(indices)
        else:
            indices = sorted(range(len(scores)), key=lambda _i: metric.key(scores[_i]))
        if max_score_of_example_in_critique is not None:
            indices = [
                _i
                for _i in indices
                if metric.key(scores[_i]) < max_score_of_example_in_critique
            ]

        _cp = critique_prompt.fill(
            str(task_prompt),
            [predictions[_i] for _i in indices][:num_examples_in_critique],
            [train[_i] for _i in indices][:num_examples_in_critique],
        )
        _gen = meta_llm.generate(_cp)
        critique = critique_prompt.parse(_gen)
        # noinspection PyUnboundLocalVariable
        critiques[task_prompt] = critique
        self.logger.info(f"Critique: [white]{critique}[/white]")
        return critiques

    def fill_in_meta_prompt(
        self,
        train,
        step_few_shot_example_scores,
        accumulative_few_shot_example_scores,
        prompt_score_pairs,
        meta_prompt,
        critiques,
        meta_llm,
        metric,
        few_shot_selection_criteria,
        num_few_shot_examples_in_meta_prompt,
        num_task_prompts_in_meta_prompt,
    ):
        if num_few_shot_examples_in_meta_prompt:
            # self.logger.debug('### Few Shot Examples\n')
            few_shot_examples = self.select_few_shot_examples(
                train,
                num_few_shot_examples_in_meta_prompt,
                few_shot_selection_criteria,
                step_few_shot_example_scores,
                accumulative_few_shot_example_scores,
            )
            # self.logger.debug(
            #     self.convert_examples_to_df(few_shot_examples, remove_newline=True, truncate=50).to_markdown(
            #         index=False) + '\n')
        else:
            few_shot_examples = None
        # noinspection PyTypeChecker
        promising_task_prompt_score_pairs = list(reversed(prompt_score_pairs.items()))[
            -num_task_prompts_in_meta_prompt:
        ]
        if hasattr(metric, "get_description"):
            promising_task_prompt_score_pairs = [
                (prompt, metric.get_description(score))
                for prompt, score in promising_task_prompt_score_pairs
            ]
        meta_prompt_text = meta_prompt.fill(
            promising_task_prompt_score_pairs,
            few_shot_examples,
            critiques=[
                critiques.get(p, None) for p, _ in promising_task_prompt_score_pairs
            ],
        )
        # self.logger.debug('### Meta Prompt\n')
        # self.logger.debug('> ' + meta_prompt_text.replace('\n', '\n> '))
        return meta_prompt_text

    def generating_new_prompts(
        self,
        meta_prompt,
        prompt_score_pairs,
        train,
        step_few_shot_example_scores,
        accumulative_few_shot_example_scores,
        critiques,
        step_save_dir,
        meta_llm,
        metric,
        few_shot_selection_criteria,
        num_few_shot_examples_in_meta_prompt,
        num_task_prompts_in_meta_prompt,
        num_new_prompts_in_each_step,
        refresh_meta_prompt_each_attempt,
        max_retry=5,
    ):
        new_prompts = dict()
        meta_prompt_text = None
        with tqdm(
            total=num_new_prompts_in_each_step, desc="Generating new prompts"
        ) as pbar:
            table = []
            counter = 0
            while len(new_prompts) < num_new_prompts_in_each_step:
                if counter >= max_retry:
                    self.logger.warning(
                        f"[red]Reached maximum retry {max_retry}.[/red]"
                    )
                    break
                if not meta_prompt_text or refresh_meta_prompt_each_attempt:
                    meta_prompt_text = self.fill_in_meta_prompt(
                        train=train,
                        step_few_shot_example_scores=step_few_shot_example_scores,
                        accumulative_few_shot_example_scores=accumulative_few_shot_example_scores,
                        prompt_score_pairs=prompt_score_pairs,
                        meta_prompt=meta_prompt,
                        critiques=critiques,
                        meta_llm=meta_llm,
                        metric=metric,
                        few_shot_selection_criteria=few_shot_selection_criteria,
                        num_few_shot_examples_in_meta_prompt=num_few_shot_examples_in_meta_prompt,
                        num_task_prompts_in_meta_prompt=num_task_prompts_in_meta_prompt,
                    )

                generations = meta_llm.batch_generate(
                    [meta_prompt_text]
                    * (num_new_prompts_in_each_step - len(new_prompts)),
                    desc=None,
                )
                for generation in generations:
                    new_prompt = meta_prompt.parse(generation)
                    if (
                        new_prompt
                        and new_prompt not in prompt_score_pairs
                        and new_prompt not in new_prompts
                    ):
                        new_prompts[new_prompt] = None
                        pbar.update()
                        table.append(
                            {
                                "meta_prompt": meta_prompt_text,
                                "generation": generation,
                                "new_prompt": new_prompt,
                            }
                        )
                    else:
                        counter += 1
            pd.DataFrame(table).to_csv(f"{step_save_dir}/meta-prompts.csv", index=False)
        return new_prompts

    def evaluate(
        self,
        task_prompt: TaskPrompt,
        dataset: List[Example],
        task_llm: LargeLanguageModel,
        metric: Metric,
        desc: str = None,
        save_path=None,
        prompt_score_pairs=None,
    ) -> Tuple[float, List[float], list]:
        xs, ys = [example.x for example in dataset], [example.y for example in dataset]
        prompts, generations, predictions = self.predict(
            xs, task_prompt, task_llm, desc
        )
        scores = [
            metric.score(p, e.y, x=e.x)
            for p, e in tqdm(
                zip(predictions, dataset),
                total=len(dataset),
                desc=metric.__class__.__name__,
            )
        ]
        overall_score = metric.aggregate(scores)
        if prompt_score_pairs is not None:
            prompt_score_pairs[task_prompt] = overall_score
            if (
                isinstance(overall_score, FloatDict)
                and isinstance(metric, MetricDict)
                and metric.primary == "rank"
            ):
                score_pairs = list(prompt_score_pairs.values())
                # Update the average rank of each prompt
                metric_scores = defaultdict(list)
                for prompt_scores in score_pairs:
                    assert isinstance(prompt_scores, FloatDict)
                    prompt_scores.scores.pop("rank", None)
                    for name, score in prompt_scores.scores.items():
                        metric_scores[name].append(score)
                for prompt_scores in score_pairs:
                    ranks = []
                    for name, _scores in metric_scores.items():
                        ranks.append(
                            sorted(_scores, key=metric[name].key, reverse=True).index(
                                prompt_scores.scores[name]
                            )
                        )
                    prompt_scores.scores["rank"] = (
                        statistics.mean(ranks) if ranks else 0
                    )
                for prompt, prompt_scores in list(zip(prompt_score_pairs, score_pairs)):
                    prompt_score_pairs[prompt] = FloatDict(
                        prompt_scores.scores["rank"], **prompt_scores.scores
                    )
                overall_score = prompt_score_pairs[task_prompt]

        self.save_evaluation_info(
            xs, ys, predictions, scores, prompts, generations, save_path
        )
        self.logger.info(f"{save_path} overall score: {overall_score}")
        return overall_score, scores, predictions

    def save_evaluation_info(
        self, xs, ys, predictions, scores, prompts, generations, save_path
    ):
        if save_path:
            table = []
            for x, y, prediction, score, prompt, generation in zip(
                xs, ys, predictions, scores, prompts, generations
            ):
                info = {
                    "x": x,
                    "y": y,
                    "prediction": prediction,
                    "prompt": prompt,
                    "generation": generation,
                    "score": score,
                }
                if isinstance(score, dict):
                    info.update(score)
                table.append(info)
            df = pd.DataFrame(table)
            df = df.sort_values("score")
            df.to_csv(save_path)

    def select_few_shot_examples(
        self,
        train,
        num_few_shot_examples_in_meta_prompt,
        few_shot_selection_criteria,
        step_few_shot_example_scores,
        accumulative_few_shot_example_scores,
    ):
        few_shot_example_indices = []
        if num_few_shot_examples_in_meta_prompt:
            if few_shot_selection_criteria == "random":
                few_shot_example_indices = random.sample(
                    list(range(len(train))), k=num_few_shot_examples_in_meta_prompt
                )
            elif few_shot_selection_criteria == "current_lowest_score":
                few_shot_example_indices = sorted(
                    enumerate(step_few_shot_example_scores), key=lambda x: sum(x[1])
                )[:num_few_shot_examples_in_meta_prompt]
                few_shot_example_indices = [x[0] for x in few_shot_example_indices]
            else:
                few_shot_example_indices = sorted(
                    enumerate(accumulative_few_shot_example_scores),
                    key=lambda x: sum(x[1]),
                )[:num_few_shot_examples_in_meta_prompt]
                few_shot_example_indices = [x[0] for x in few_shot_example_indices]
        few_shot_examples = [train[i] for i in few_shot_example_indices]
        return few_shot_examples

    def predict(
        self,
        xs: Sequence[Union[str, Any]],
        task_prompt: TaskPrompt,
        task_llm: LargeLanguageModel,
        desc: str = None,
    ) -> Sequence[Union[str, Any]]:
        prompts = [task_prompt.fill(x) for x in xs]
        generations = task_llm.batch_generate(prompts, desc=desc)
        predictions = [
            task_prompt.parse(generation) for generation, x in zip(generations, xs)
        ]
        return prompts, generations, predictions

    def convert_prompt_scores_to_df(
        self,
        prompt_scores: Dict[TaskPrompt, Union[float, dict]],
        prompt_steps: Dict[TaskPrompt, int],
    ):
        table = []
        for prompt, score in prompt_scores.items():
            info = {"step": prompt_steps[prompt], "prompt": str(prompt)}

            if isinstance(score, dict):
                info.update(score)
            else:
                info["score"] = score
            table.append(info)
        return pd.DataFrame(table)

    def convert_examples_to_df(
        self, examples: List[Example], remove_newline=False, truncate=None
    ):
        table = []
        for example in examples:
            x = example.x
            y = example.y
            if truncate:
                x = str(x)
                y = str(y)
                x = x[:truncate]
                y = y[:truncate]
            if remove_newline:
                x = str(x)
                x = x.replace("\n", " ")
                y = y.replace("\n", " ")
            table.append({"x": x, "y": y})
        return pd.DataFrame(table)

    @staticmethod
    def sort_prompts_best_to_worst(prompt_score_pairs, metric):
        return dict(
            sorted(
                prompt_score_pairs.items(), key=lambda x: metric.key(x[1]), reverse=True
            )
        )
