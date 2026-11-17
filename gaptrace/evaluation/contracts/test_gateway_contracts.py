import pytest

from gaptrace import tokens
from gaptrace.context import ContextBudgetExceeded
from gaptrace.gateway import Gateway, ModelOutputInvalid
from gaptrace.schema import StepOutput

from .conftest import Script


def test_oversized_input_never_reaches_backend(settings):
    backend = Script([])
    gateway = Gateway(settings, {"local": backend})
    with pytest.raises(ContextBudgetExceeded):
        gateway.call("local", "x" * 60000, StepOutput)
    assert backend.calls == []


def test_retry_cannot_cross_token_boundary(settings):
    tokens.set_counter(lambda text: len(text))
    limit = settings.budget.total - settings.budget.output - settings.budget.margin
    backend = Script(["invalid"])
    with pytest.raises(ContextBudgetExceeded):
        Gateway(settings, {"local": backend}).call("local", "x" * limit, StepOutput)
    assert len(backend.calls) == 1


def test_two_invalid_replies_stop(settings):
    backend = Script(["invalid", "invalid"])
    with pytest.raises(ModelOutputInvalid):
        Gateway(settings, {"local": backend}).call("local", "question", StepOutput)
    assert len(backend.calls) == 2


@pytest.mark.parametrize("changed", ["model", "output", "context"])
def test_cache_invalidates_on_configuration_change(settings, tmp_path, changed):
    backend = Script([{"answer": "first"}, {"answer": "second"}])
    gateway = Gateway(settings, {"local": backend}, tmp_path)
    assert gateway.call("local", "Q", StepOutput)[0].answer == "first"
    assert gateway.call("local", "Q", StepOutput)[0].answer == "first"
    if changed == "model":
        settings.model.local = "different-model"
    elif changed == "output":
        settings.budget.output -= 1
    else:
        settings.model.local_num_ctx += 1
    assert gateway.call("local", "Q", StepOutput)[0].answer == "second"
    assert len(backend.calls) == 2


def test_local_only_never_invokes_cloud(settings):
    local, cloud = Script([{}]), Script([])
    settings.cloud.enabled = False
    _, tier = Gateway(settings, {"local": local, "cloud_light": cloud}).call("cloud_light", "Q", StepOutput)
    assert tier == "local" and cloud.calls == []


def test_cloud_exception_falls_back_and_is_logged(settings):
    settings.cloud.enabled = True
    local, cloud = Script([{}]), Script([RuntimeError("provider unavailable")])
    gateway = Gateway(settings, {"local": local, "cloud_light": cloud})
    _, tier = gateway.call("cloud_light", "Q", StepOutput)
    assert tier == "local" and len(local.calls) == 1
    assert gateway.log[0]["fallback"] == "cloud_light"


def test_cloud_retry_cannot_exceed_topic_call_cap(settings):
    settings.cloud.enabled = True
    settings.limits.max_cloud_calls_per_topic = 1
    cloud, local = Script(["invalid", {}]), Script([{}])
    Gateway(settings, {"cloud_light": cloud, "local": local}).call("cloud_light", "Q", StepOutput)
    assert len(cloud.calls) <= 1, "Schema retry spent a cloud call beyond the topic cap"
