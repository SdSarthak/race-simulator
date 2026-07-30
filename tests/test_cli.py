"""End-to-end CLI tests. They stay headless and tiny so they run in seconds."""

import os

import pytest

import main


def _train_args(model, extra=()):
    return ["train", "--headless", "--generations", "1", "--pop", "4",
            "--max-steps", "25", "--laps", "1", "--seed", "5",
            "--model", model, "--log-dir", "", "--checkpoint-every", "0",
            "--quiet", *extra]


def test_no_command_prints_help(capsys):
    assert main.main([]) == 1
    assert "train" in capsys.readouterr().out


def test_training_writes_a_model(tmp_path, capsys):
    model = str(tmp_path / "best.pt")
    assert main.main(_train_args(model)) == 0
    assert os.path.exists(model)
    assert "done" in capsys.readouterr().out


def test_training_resumes_from_an_existing_model(tmp_path, capsys):
    model = str(tmp_path / "best.pt")
    main.main(_train_args(model))
    capsys.readouterr()
    assert main.main(_train_args(model)) == 0
    assert "resumed" in capsys.readouterr().out


def test_no_resume_starts_clean(tmp_path, capsys):
    model = str(tmp_path / "best.pt")
    main.main(_train_args(model))
    capsys.readouterr()
    main.main(_train_args(model, ["--no-resume"]))
    assert "resumed" not in capsys.readouterr().out


def test_a_mismatched_checkpoint_falls_back_to_a_fresh_run(tmp_path, capsys):
    import torch
    from ai import PolicyNet, save_checkpoint
    from config import STATE_DIM, ACTION_DIM

    torch.manual_seed(0)
    model = str(tmp_path / "wrong.pt")
    save_checkpoint(PolicyNet(STATE_DIM + 4, ACTION_DIM).state_dict(),
                    model, STATE_DIM + 4, ACTION_DIM)

    assert main.main(_train_args(model)) == 0
    assert "starting fresh" in capsys.readouterr().out


def test_evaluate_scores_a_trained_model(tmp_path, capsys):
    model = str(tmp_path / "best.pt")
    main.main(_train_args(model))
    capsys.readouterr()

    assert main.main(["evaluate", "--model", model, "--episodes", "2",
                      "--max-steps", "30", "--laps", "1", "--seed", "2",
                      "--stochastic"]) == 0
    out = capsys.readouterr().out
    assert "mean laps" in out
    assert out.count("\n") >= 4


def test_evaluate_without_a_model_explains_itself(tmp_path, capsys):
    assert main.main(["evaluate", "--model", str(tmp_path / "nope.pt")]) == 1
    assert "no model at" in capsys.readouterr().out


def test_replay_without_a_model_explains_itself(tmp_path, capsys):
    assert main.main(["replay", "--model", str(tmp_path / "nope.pt")]) == 1
    assert "no model at" in capsys.readouterr().out


def test_layout_choice_is_validated():
    with pytest.raises(SystemExit):
        main.main(["train", "--layout", "spiral"])


def test_the_oval_layout_trains_too(tmp_path):
    model = str(tmp_path / "oval.pt")
    assert main.main(_train_args(model, ["--layout", "oval"])) == 0
    assert os.path.exists(model)
