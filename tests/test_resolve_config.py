import importlib

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError, ValidationError


RUNTIMES = [("verl", "projects/rl/config.yaml"), ("generalized.verl", "generalized/config.yaml")]


def resolve(runtime, config_file, *overrides):
    module = importlib.import_module(f"{runtime}.trainer.resolve_config")
    return module.resolve_config(OmegaConf.from_dotlist([f"config={config_file}", *overrides]))


@pytest.mark.parametrize("runtime,config_file", RUNTIMES)
def test_overrides_are_applied(runtime, config_file):
    config = resolve(runtime, config_file, "worker.rollout.n=6", "data.val_files=[]", "trainer.logger=[file]")
    assert config.worker.rollout.n == 6
    assert config.data.val_files == []
    assert tuple(config.trainer.logger) == ("file",)


@pytest.mark.parametrize("runtime,config_file", RUNTIMES)
def test_bad_overrides_fail_before_training(runtime, config_file):
    with pytest.raises(ConfigKeyError):
        resolve(runtime, config_file, "worker.actor.optimze_captioner=true")
    with pytest.raises(ValidationError):
        resolve(runtime, config_file, "worker.rollout.n=eight")


@pytest.mark.parametrize("runtime,config_file", RUNTIMES)
def test_missing_resume_path_is_an_error(runtime, config_file, tmp_path):
    with pytest.raises(FileNotFoundError, match="load_checkpoint_path"):
        resolve(runtime, config_file, f"trainer.load_checkpoint_path={tmp_path / 'missing'}")
    config = resolve(runtime, config_file, f"trainer.load_checkpoint_path={tmp_path}")
    assert config.trainer.load_checkpoint_path == str(tmp_path)


@pytest.mark.parametrize("runtime,config_file", RUNTIMES)
def test_missing_image_dir_is_an_error(runtime, config_file, tmp_path):
    with pytest.raises(FileNotFoundError, match="image_dir"):
        resolve(runtime, config_file, f"data.image_dir={tmp_path / 'missing'}")
    assert resolve(runtime, config_file, f"data.image_dir={tmp_path}").data.image_dir == str(tmp_path)


@pytest.mark.parametrize("runtime,config_file", RUNTIMES)
def test_missing_format_prompt_is_an_error(runtime, config_file, tmp_path):
    with pytest.raises(FileNotFoundError, match="format_prompt"):
        resolve(runtime, config_file, f"data.format_prompt={tmp_path / 'missing.jinja'}")
