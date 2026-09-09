"""
Those are tests that will be shared with students
They should test that the code structure/return values
are of correct type/shape
"""

import os
import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def submission_dir():
    path_base = os.path.dirname(os.path.dirname(__file__))
    submission_path = os.path.join(path_base, "submission")
    return submission_path

def test_question1(submission_dir):
    ex1_path = os.path.join(submission_dir, "question1")
    init_path = os.path.join(ex1_path, "__init__.py")
    assert os.path.isfile(init_path)
    mdp_solver_path = os.path.join(ex1_path, "mdp_solver.py")
    assert os.path.isfile(mdp_solver_path)

def test_question2(submission_dir):
    ex2_path = os.path.join(submission_dir, "question2")
    init_path = os.path.join(ex2_path, "__init__.py")
    assert os.path.isfile(init_path)
    agents_path = os.path.join(ex2_path, "agents.py")
    assert os.path.isfile(agents_path)
    train_mc_path = os.path.join(ex2_path, "train_monte_carlo.py")
    assert os.path.isfile(train_mc_path)
    train_q_path = os.path.join(ex2_path, "train_q_learning.py")
    assert os.path.isfile(train_q_path)

def test_question3(submission_dir):
    ex3_path = os.path.join(submission_dir, "question3")
    init_path = os.path.join(ex3_path, "__init__.py")
    assert os.path.isfile(init_path)
    agents_path = os.path.join(ex3_path, "agents.py")
    assert os.path.isfile(agents_path)
    train_dqn_path = os.path.join(ex3_path, "train_dqn.py")
    assert os.path.isfile(train_dqn_path)
    train_reinforce_path = os.path.join(ex3_path, "train_reinforce.py")
    assert os.path.isfile(train_reinforce_path)

def test_question4(submission_dir):
    ex4_path = os.path.join(submission_dir, "question4")
    init_path = os.path.join(ex4_path, "__init__.py")
    assert os.path.isfile(init_path)
    agents_path = os.path.join(ex4_path, "agents.py")
    assert os.path.isfile(agents_path)
    train_ppo_path = os.path.join(ex4_path, "train_ppo.py")
    assert os.path.isfile(train_ppo_path)
    evaluate_ppo_path = os.path.join(ex4_path, "evaluate_ppo.py")
    assert os.path.isfile(evaluate_ppo_path)

def test_question5(submission_dir):
    ex5_path = os.path.join(submission_dir, "question5")
    init_path = os.path.join(ex5_path, "__init__.py")
    assert os.path.isfile(init_path)
    train_ppo_path = os.path.join(ex5_path, "train_ppo.py")
    assert os.path.isfile(train_ppo_path)
    evaluate_ppo_path = os.path.join(ex5_path, "evaluate_ppo.py")
    assert os.path.isfile(evaluate_ppo_path)

def test_answer_sheet(submission_dir):
    answer_sheet_path = os.path.join(submission_dir, "answer_sheet.py")
    assert os.path.isfile(answer_sheet_path)
