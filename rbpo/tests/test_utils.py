import math
import torch
import pytest

from rbpo.utils.wer import compute_wer, compute_wer_from_tokens, compute_batch_wer
from rbpo.utils.advantages import (
    group_relative_advantages,
    group_relative_advantages_per_utterance,
)
from rbpo.utils.clipping import clip_surrogate, length_normalize_ratio




class TestComputeWer:
    def test_identical(self):
        assert compute_wer("hello world", "hello world") == 0.0

    def test_completely_different(self):
        assert compute_wer("a b c d", "e f g h") == 1.0

    def test_empty_ref_nonempty_hyp(self):
        assert compute_wer("some words", "") == 1.0

    def test_both_empty(self):
        assert compute_wer("", "") == 0.0

    def test_empty_hyp_nonempty_ref(self):
        assert compute_wer("", "some words") == 1.0

    def test_single_substitution(self):
        assert compute_wer("the cat sat down", "the dog sat down") == 0.25

    def test_insertion(self):
        wer = compute_wer("the big cat sat", "the cat sat")
        assert wer == pytest.approx(1.0 / 3.0)

    def test_deletion(self):
        wer = compute_wer("the sat", "the cat sat")
        assert wer == pytest.approx(1.0 / 3.0)


class TestComputeWerFromTokens:
    def test_identical_tokens(self):
        assert compute_wer_from_tokens([1, 2, 3], [1, 2, 3]) == 0.0

    def test_empty_ref_nonempty_hyp(self):
        assert compute_wer_from_tokens([1, 2], []) == 1.0

    def test_both_empty(self):
        assert compute_wer_from_tokens([], []) == 0.0

    def test_known_edit_distance(self):
        # ref=[1,2,3,4], hyp=[1,5,3,4] -> 1 substitution, edit_dist=1, wer=0.25
        assert compute_wer_from_tokens([1, 5, 3, 4], [1, 2, 3, 4]) == 0.25

    def test_all_different(self):
        assert compute_wer_from_tokens([10, 20], [30, 40]) == 1.0


class TestComputeBatchWer:
    def test_batch(self):
        hyps = ["the cat sat", "hello world"]
        refs = ["the dog sat", "hello world"]
        wers = compute_batch_wer(hyps, refs)
        assert wers[0] == pytest.approx(1.0 / 3.0)
        assert wers[1] == 0.0




class TestGroupRelativeAdvantages:
    def test_sum_zero(self):
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        adv = group_relative_advantages(rewards)
        assert adv.sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_best_reward_highest_advantage(self):
        rewards = [1.0, 5.0, 3.0]
        adv = group_relative_advantages(rewards)
        assert adv.argmax().item() == 1

    def test_all_same(self):
        rewards = [2.0, 2.0, 2.0]
        adv = group_relative_advantages(rewards)
        assert torch.allclose(adv, torch.zeros(3))

    def test_single_candidate(self):
        adv = group_relative_advantages([7.0])
        assert adv.item() == pytest.approx(0.0, abs=1e-6)

    def test_tensor_input(self):
        rewards = torch.tensor([1.0, 3.0, 5.0])
        adv = group_relative_advantages(rewards)
        assert adv.sum().item() == pytest.approx(0.0, abs=1e-6)


class TestGroupRelativeAdvantagesPerUtterance:
    def test_two_utterances(self):
        rewards = torch.tensor([1.0, 3.0, 5.0, 10.0, 20.0])
        num_per_utt = [3, 2]
        adv = group_relative_advantages_per_utterance(rewards, num_per_utt)

        group1 = adv[:3]
        group2 = adv[3:]
        assert group1.sum().item() == pytest.approx(0.0, abs=1e-6)
        assert group2.sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_single_per_group(self):
        rewards = torch.tensor([5.0, 10.0])
        adv = group_relative_advantages_per_utterance(rewards, [1, 1])
        assert torch.allclose(adv, torch.zeros(2))




class TestClipSurrogate:
    def test_inside_bounds_no_clip(self):
        rho = torch.tensor([1.0, 1.1, 0.9])
        adv = torch.tensor([2.0, -1.0, 0.5])
        result = clip_surrogate(rho, adv)
        expected = rho * adv
        assert torch.allclose(result, expected)

    def test_rho_above_positive_advantage(self):
        rho = torch.tensor([2.0])
        adv = torch.tensor([1.0])
        result = clip_surrogate(rho, adv, eps_high=0.28)
        expected = torch.tensor([1.28])  # (1+0.28)*1.0
        assert torch.allclose(result, expected)

    def test_rho_below_negative_advantage(self):
        rho = torch.tensor([0.5])
        adv = torch.tensor([-1.0])
        result = clip_surrogate(rho, adv, eps_low=0.2)
        expected = torch.tensor([-0.8])  # (1-0.2)*(-1.0)
        assert torch.allclose(result, expected)

    def test_rho_one_exactly(self):
        rho = torch.tensor([1.0])
        adv = torch.tensor([3.5])
        result = clip_surrogate(rho, adv)
        assert result.item() == pytest.approx(3.5)

    def test_batch_manual_loop(self):
        torch.manual_seed(42)
        rho = torch.rand(10) * 2 + 0.1
        adv = torch.randn(10)
        result = clip_surrogate(rho, adv, eps_low=0.2, eps_high=0.28)
        for i in range(10):
            r, a = rho[i].item(), adv[i].item()
            clipped_r = max(0.8, min(1.28, r))
            expected = min(r * a, clipped_r * a)
            assert result[i].item() == pytest.approx(expected, abs=1e-5)


class TestLengthNormalizeRatio:
    def test_known_value(self):
        log_rho = torch.tensor([math.log(2.0)])
        lengths = torch.tensor([50.0])
        result = length_normalize_ratio(log_rho, lengths)
        expected = 2.0 ** (1.0 / 50.0)
        assert result.item() == pytest.approx(expected, rel=1e-4)

    def test_length_one(self):
        log_rho = torch.tensor([1.0])
        lengths = torch.tensor([1.0])
        result = length_normalize_ratio(log_rho, lengths)
        assert result.item() == pytest.approx(math.exp(1.0), rel=1e-5)
