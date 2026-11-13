"""Original mjlab math, without mutable TorchScript execution plans in BPTT."""

from types import FunctionType

import pytest
import torch
from mjlab.utils.lab_api import math as mjmath

from supermimic.kinematics import rotations

NAMES = rotations._PRIMITIVE_NAMES


def _assert_plain_dependencies(function, visited):
    if id(function) in visited:
        return
    visited.add(id(function))
    assert isinstance(function, FunctionType)
    for name in function.__code__.co_names:
        dependency = function.__globals__.get(name)
        assert not isinstance(dependency, torch.jit.ScriptFunction), name
        if isinstance(dependency, FunctionType) and dependency.__module__ == function.__module__:
            _assert_plain_dependencies(dependency, visited)


def test_rotation_clones_keep_original_bodies_and_do_not_mutate_mjlab():
    for name in NAMES:
        original = getattr(mjmath, name)
        body = getattr(original, "_torchdynamo_inline", original)
        plain = getattr(rotations, name)
        assert isinstance(original, torch.jit.ScriptFunction)
        assert plain.__code__ is body.__code__
        assert plain.__globals__ is not body.__globals__
        _assert_plain_dependencies(plain, set())
    with pytest.raises(RuntimeError, match="incompatible"):
        rotations._python_math_body(object(), {})


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_plain_rotation_values_and_all_input_gradients_match_mjlab(name, dtype, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    generator = torch.Generator(device=device).manual_seed(991)

    def normal(shape):
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    q, q2 = normal((2, 3, 4)), normal((2, 3, 4))
    q, q2 = q / q.norm(dim=-1, keepdim=True), q2 / q2.norm(dim=-1, keepdim=True)
    vector, angle = normal((2, 3, 3)), normal((2, 3))
    if name in ("quat_apply", "quat_apply_inverse"):
        inputs = q, vector
    elif name in ("quat_error_magnitude", "quat_mul", "quat_box_minus"):
        inputs = q, q2
    elif name == "quat_from_angle_axis":
        inputs = angle, vector
    else:
        inputs = (q,)
    original_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
    plain_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
    original = getattr(mjmath, name)(*original_inputs)
    plain = getattr(rotations, name)(*plain_inputs)
    weights = normal(original.shape)
    original_grad = torch.autograd.grad((original * weights).sum(), original_inputs)
    plain_grad = torch.autograd.grad((plain * weights).sum(), plain_inputs)
    tolerance = 1e-12 if dtype == torch.float64 else 3e-5
    torch.testing.assert_close(plain, original, rtol=tolerance, atol=tolerance)
    for actual, expected in zip(plain_grad, original_grad, strict=True):
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Graphs")
def test_uncompiled_checkpoint_capture_survives_external_script_profiling_history(tmp_path):
    from test_policy_update_capture import _components, _steps, _trainer

    components = _components(tmp_path, "cuda:0")
    trainers = [_trainer(components, capture=True, activation_checkpoint=True) for _ in range(2)]
    try:
        for repeat, trainer in enumerate(trainers):
            # Exercise the untouched upstream ScriptFunction before each run.
            # Its profiling/gradient plan must not affect our checkpoint graph.
            for _ in range(1 + repeat * 5):
                quat = torch.tensor([[.8, .1, .2, .3]], device="cuda:0", requires_grad=True)
                vector = torch.ones(1, 3, device="cuda:0", requires_grad=True)
                mjmath.quat_apply(quat, vector).square().sum().backward()
            for step in range(6):
                values = trainer.update(components[-1])
                assert all(torch.isfinite(torch.tensor(value)) for value in values.values())
                assert _steps(trainer.opt) == _steps(trainer.critic_opt) == {step + 1}
                assert trainer.ema.num_updates == step + 1
            assert trainer._captured_update.captured and trainer._captured_update.phase == "replay"
    finally:
        for trainer in trainers:
            trainer.reset_capture()
