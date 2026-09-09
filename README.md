# CE7457 Assignment

Companion code for the CE7457 Reinforcement Learning assignment. See the assignment PDF for
the question descriptions; this README covers how the codebase is organised and how to run it.

## 1. Install uv

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python version, virtual
environment, and dependencies in one step. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
```

(See the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/) for
Windows and other options. Learn more at https://docs.astral.sh/uv/.)

## 2. Get the code

```bash
git clone https://github.com/ntu-agents/ce7457-assignment.git
cd ce7457-assignment
```

## 3. Install dependencies

```bash
uv sync
```

This creates a `.venv/` and installs the dependencies specified in `pyproject.toml`, at the exact
versions pinned in `uv.lock`.

## 4. Repository structure

```
submission/
├── answer_sheet.py   # write-up / multiple-choice answers referenced by each question
├── constants.py       # fixed hyperparameters and environment settings — do not edit
├── question1/          # Question 1 — Dynamic Programming
├── question2/          # Question 2 — Tabular Reinforcement Learning
├── question3/          # Question 3 — Deep Reinforcement Learning
├── question4/           # Question 4 — Advanced Policy Gradient Methods (PPO)
├── question5/           # Question 5 — Fine-tuning the Algorithms
└── util/                # shared helpers used by question2-5
tests/                  # public self-checks — run these before you submit
```

The assignment PDF tells you exactly which functions in which file to implement for each question; 
everything else in `submission/` should be left unchanged. See assignment PDF for more detail.

## 5. Implement each question

Open the relevant file and fill in the functions marked in the assignment PDF (look for
`### PUT YOUR CODE HERE ###` / `raise NotImplementedError(...)`), then run the matching script:

| Question | Edit | Run |
|---|---|---|
| 1 — Dynamic Programming | `submission/question1/mdp_solver.py` | `uv run submission/question1/mdp_solver.py` |
| 2 — Tabular RL | `submission/question2/agents.py` | `uv run submission/question2/train_q_learning.py`, `uv run submission/question2/train_monte_carlo.py` |
| 3 — Deep RL | `submission/question3/agents.py` | `uv run submission/question3/train_dqn.py`, `uv run submission/question3/train_reinforce.py` |
| 4 — Advanced Policy Gradient Methods | `submission/question4/agents.py` | `uv run submission/question4/train_ppo.py` |
| 5 — Fine-tuning the Algorithms | `submission/question5/train_ppo.py` config (and, optionally, `schedule_hyperparameters` in `submission/question4/agents.py`) | `uv run submission/question5/train_ppo.py` |

Note: using `uv run` will automatically use the `.venv`, no venv activation needed.

## 6. Check your work before submitting

```bash
uv run pytest
```

The `tests/` directory contains the public self-checks, they confirm your submission has the
right files/functions. They don't verify the *correctness* or *performance* of your implementation (nor your `answer_sheet.py`
answers) — passing them is necessary, but not sufficient, for full marks.

## 7. Build your submission zip

```bash
uv run make_submission.py
```

This zips the required files into `submission.zip`, and warns about anything missing. Upload `submission.zip` to NTULearn. Do not zip virtual environments.
