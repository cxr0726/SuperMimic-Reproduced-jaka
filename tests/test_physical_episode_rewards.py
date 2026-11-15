import pytest
import torch

from supermimic.training.episode_rewards import PhysicalEpisodeRewards


def test_physical_reward_phases_across_logging_and_reset_boundaries():
    cap = .112
    meter = PhysicalEpisodeRewards(2, ["tracking", "penalty"], cap, "cpu")

    def step(terms, dones):
        terms = torch.tensor(terms) * cap
        meter.record(terms.sum(-1), terms, torch.tensor(dones))

    def phase(metrics, name):
        return [metrics[f"train/reward_terms/{name}/{term}"] for term in meter.term_names]

    # Env 0 terminates immediately. Env 1 crosses the logging boundary.
    step([[.6, -.1], [.4, -.1]], [True, False])
    n, m = meter.metrics(meter.consume())
    assert n == 1 and m["train/reward_terms/post_first_episodes"] == 0
    assert phase(m, "episode") == pytest.approx([.6, -.1])
    assert phase(m, "first_step") == pytest.approx([.6, -.1])
    assert "train/reward_terms/post_first_step/tracking" not in m
    assert meter.metrics(meter.consume()) == (0, {})

    step([[.8, -.2], [.9, -.1]], [False, True])
    step([[.7, -.1], [1., 0.]], [True, True])
    # Using rank sums must weight by counts, even with an empty rank.
    other = PhysicalEpisodeRewards(1, ["tracking", "penalty"], cap, "cpu")
    extra = torch.tensor([[.25, -.25]]) * cap
    other.record(extra.sum(-1), extra, torch.tensor([True]))
    totals = meter.consume() + other.consume() + other.consume()
    n, m = meter.metrics(totals)
    assert n == 4 and m["train/episode_len"] == pytest.approx(1.5)
    assert m["train/reward_terms/post_first_episodes"] == 2
    assert phase(m, "episode") == pytest.approx([(.65 + .75 + 1. + .25) / 4, -.5 / 4])
    assert phase(m, "first_step") == pytest.approx([(.4 + .8 + 1. + .25) / 4, -.55 / 4])
    assert phase(m, "post_first_step") == pytest.approx([.8, -.1])
    assert m["train/episode_norm_return"] == pytest.approx((.55 + .60 + 1.) / 4)
    assert abs(m["train/reward_terms/episode_sum_error"]) < 1e-7
