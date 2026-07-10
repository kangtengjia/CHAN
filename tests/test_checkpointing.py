import torch

from lib.checkpointing import build_checkpoint, restore_checkpoint


def test_checkpoint_round_trip_restores_optimizer_epoch_and_best_score():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    optimizer.param_groups[0]["lr"] = 3e-5
    checkpoint = build_checkpoint(
        epoch=17,
        model_state={"weights": model.state_dict()},
        optimizer=optimizer,
        best_score=42.5,
        best_metrics={"R@1": 10.0, "Rsum": 42.5},
        iterations=123,
        config={"data_name": "scanrefer"},
    )

    restored_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    state = restore_checkpoint(checkpoint, restored_optimizer)

    assert state.epoch == 17
    assert state.best_score == 42.5
    assert state.best_metrics["Rsum"] == 42.5
    assert state.iterations == 123
    assert restored_optimizer.param_groups[0]["lr"] == 3e-5


def test_legacy_checkpoint_requires_weights_only_mode():
    legacy = {"epoch": 3, "model": [], "best_rsum": 4.0, "Eiters": 9}
    optimizer = torch.optim.AdamW(torch.nn.Linear(1, 1).parameters(), lr=1e-4)

    try:
        restore_checkpoint(legacy, optimizer)
    except ValueError as error:
        assert "weights-only" in str(error)
    else:
        raise AssertionError("expected legacy resume to be rejected")

    state = restore_checkpoint(legacy, optimizer, weights_only=True)
    assert state.epoch == 0
    assert state.best_score == 0.0
    assert state.iterations == 0
