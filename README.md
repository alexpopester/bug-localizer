# Bug Localizer: GraphCodeBERT ranking for LLM bug fixing

Takes a GitHub issue and finds the functions in a Python codebase that most likely need changing. It does this by fine-tuning GraphCodeBERT to match bug reports to code and adding that ranking step to the [Agentless](https://github.com/OpenAutoCoder/Agentless) bug-fixing pipeline.

![Localization results on SWE-bench Lite](benchmark_comparison_v2_gemini.png)

## What I built

Agentless fixes GitHub issues in three phases: localize the bug, generate patches, then validate them. Localization is where it most often fails. If the right function is never found, no patch can be correct. This project replaces the LLM-only localization with a learned semantic ranking:

- **Fine-tuned retriever** (`classification/train_localizer.py`): fine-tunes `microsoft/graphcodebert-base` with sentence-transformers on (issue, fixed function, unrelated function) triplets built from the SWE-bench *train* split.
- **Semantic scoring** (`agentless/fl/semantic_scoring.py`): parses each repository with Python's `ast`, embeds every function, and ranks them by cosine similarity to the issue text.
- **More model backends** (`agentless/util/model.py`): adds Gemini, a local OpenAI-compatible server, and llama.cpp to the existing OpenAI, Anthropic, and DeepSeek clients.
- **Evaluation** (`classification/*_results.py`, `visualize_logs.py`): Hit@k, precision, recall, and F1 against the gold patches.

## Results

Evaluated on all 300 SWE-bench Lite issues (function-level localization):

| Method | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|
| Agentless LLM localization, Gemini | 12.3% | 12.3% | 12.3% |
| Agentless LLM localization, GPT-4o (original Agentless run) | 46.7% | 51.3% | 51.3% |
| GraphCodeBERT, not fine-tuned | 2.3% | 7.3% | 9.7% |
| **GraphCodeBERT, fine-tuned (2 epochs)** | **19.3%** | **32.0%** | **40.0%** |

After two epochs of fine-tuning, the 125M-parameter retriever beat the Gemini-driven pipeline at every cutoff, and its Hit@5 came within about 11 points of the stronger LLM baseline. Embedding the functions is a one-time cost, and ranking a new issue against them takes a single embedding pass with no API calls.

## Running it

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. **Model:** download the [fine-tuned model](https://drive.google.com/file/d/1B9X52hsL7xnTmj-Ale8w2pCVWGxjDUhR/view?usp=sharing) and unzip it in the repository root. Without it, the base GraphCodeBERT is used, which is faster to try but much less accurate.
2. **Localization results to rank:** download Agentless's [pre-generated SWE-bench Lite run](https://github.com/OpenAutoCoder/Agentless/releases/download/v1.5.0/agentless_swebench_lite.zip).
3. **Score:** in `run_semantic_scoring_alone.sh`, set `in_dir` to the folder containing `file_level/`, `file_level_combined/`, and so on. Then run it. Embeddings are written to `embeddings/`, and the rankings to `out_dir/out_file`.

To score the whole dataset, use the "all instances" section of that script. Pointing `PROJECT_FILE_LOC` at a local copy of the [SWE-bench Lite repositories](https://drive.usercontent.google.com/download?id=15-4XjTmY48ystrsc_xcvtOkMs3Fx8RoW&export=download&authuser=0&confirm=t&uuid=927974f7-294c-45ab-b5ce-dd8dd816aba2&at=ALWLOp55XuTjV6RIYrtvNMwuAzhl%3A1763766440739) (3.5 GB) saves downloading them again.

To regenerate the LLM localization yourself, set your API key and `RESULTS_FOLDER`, then run `full_workflow/localization_commands_with_semantic_scoring.sh`.

## Built with

Python, PyTorch, Hugging Face Transformers, sentence-transformers, scikit-learn, and the SWE-bench datasets.

Originally a final project for CS 6640. It builds on [Agentless](https://github.com/OpenAutoCoder/Agentless) (MIT, © 2024 OpenAutoCoder), whose original README is below.

<details>
<summary>Original Agentless README</summary>

# 😺 Agentless

<p align="center">
    <a href="https://arxiv.org/abs/2407.01489"><img src="https://img.shields.io/badge/📃-Arxiv-b31b1b?style=for-the-badge"></a>
    <a href="https://github.com/OpenAutoCoder/Agentless/blob/master/LICENSE"><img src="https://forthebadge.com/images/badges/license-mit.svg" style="height: 28px"></a>
</p>

<p align="center">
    <big><a href="#-news">😽News</a></big> |
    <big><a href="#-setup">🐈Setup</a></big> |
    <big><a href="#-comparison">🧶Comparison</a></big> | 
    <big><a href="#-artifacts">🐈‍⬛Artifacts</a></big> |
    <big><a href="#-citation">📝Citation</a></big> |
    <big><a href="#-acknowledgement">😻Acknowledgement</a></big>
</p>

## 😽 News 

- *Dec 2nd, 2024*: We integrated Agentless with Claude 3.5 Sonnet to achieve 40.7% and 50.8% solve rate on SWE-bench lite and verified 
- *Oct 28th, 2024*: We just released OpenAutoCoder-Agentless 1.5! 
- *July 1st, 2024*: We just released OpenAutoCoder-Agentless 1.0! **Agentless** currently is the best open-source approach on SWE-bench lite with 82 fixes (27.3%) and costing on average $0.34 per issue.

## 😺 About 

**Agentless** is an *agentless* approach to automatically solve software development problems. To solve each issue, **Agentless** follows a simple three phase process: localization, repair, and patch validation.
- 🙀 **Localization**: Agentless employs a hierarchical process to first localize the fault to specific files, then to relevant classes or functions, and finally to fine-grained edit locations
- 😼 **Repair**: Agentless takes the edit locations and samples multiple candidate patches per bug in a simple diff format
- 😸 **Patch Validation**: Agentless selects the regression tests to run and generates additional reproduction test to reproduce the original error. Using the test results, Agentless re-ranks all remaining patches to selects one to submit

## 🐈 Setup

First create the environment 

```shell
git clone https://github.com/OpenAutoCoder/Agentless.git
cd Agentless

conda create -n agentless python=3.11 
conda activate agentless
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

<details><summary>⏬ Developer Setup</summary>
<div>

```shell
# for contribution, please install the pre-commit hook.
pre-commit install  # this allows a more standardized code style
```

</div>
</details>

Then export your OpenAI API key 
```shell
export OPENAI_API_KEY={key_here}
```

Now you are ready to run **Agentless** on the problems in SWE-bench! 

> [!NOTE]
> 
> To reproduce the full SWE-bench lite experiments and follow our exact setup as described in the paper. Please see this [README](https://github.com/OpenAutoCoder/Agentless/blob/main/README_swebench.md)

## 🧶 Comparison

Below shows the comparison graph between **Agentless** and the best open-source agent-based approaches on SWE-bench lite

<p align="center">
<img src="./resources/comparison_graph.png" style="width:75%; margin-left: auto; margin-right: auto;">
</p>

## 🐈‍⬛ Artifacts

You can download the complete artifacts of **Agentless** in our [v1.5.0 release](https://github.com/OpenAutoCoder/Agentless/releases/tag/v1.5.0):
- 🐈‍⬛ agentless_swebench_lite: complete Agentless run on SWE-bench Lite
- 🐈‍⬛ agentless_swebench_verified: complete Agentless run on SWE-bench Verified
- 🐈‍⬛ swebench_repo_structure: preprocessed structure information for each SWE-Bench problem

You can also checkout `classification/` folder to obtain our manual classifications of SWE-bench-lite as well as our filtered SWE-bench-lite-*S* problems.

## 📝 Citation

```bibtex
@article{agentless,
  author    = {Xia, Chunqiu Steven and Deng, Yinlin and Dunn, Soren and Zhang, Lingming},
  title     = {Agentless: Demystifying LLM-based Software Engineering Agents},
  year      = {2024},
  journal   = {arXiv preprint},
}
```

> [!NOTE]
> 
> The first two authors contributed equally to this work, with author order determined via [*Nigiri*](https://senseis.xmp.net/?Nigiri)

## 😻 Acknowledgement 

* [SWE-bench](https://www.swebench.com/)
* [Aider](https://github.com/paul-gauthier/aider)
* [SWE-bench-docker](https://github.com/aorwall/SWE-bench-docker)

</details>
