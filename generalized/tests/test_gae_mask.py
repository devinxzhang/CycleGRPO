"""Batched GAE must respect each response's own padding mask."""

import torch

from generalized.verl.trainer.core_algos import compute_gae_advantage_return


def test_gae_uses_each_response_mask_independently():
    rewards = torch.tensor([[1.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
    values = torch.zeros_like(rewards)
    response_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    advantages, returns = compute_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=response_mask,
        gamma=1.0,
        lam=1.0,
    )

    torch.testing.assert_close(returns, torch.tensor([[2.0, 1.0, 0.0], [2.0, 0.0, 0.0]]))
    assert torch.isfinite(advantages).all()
