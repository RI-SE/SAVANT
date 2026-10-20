"""Tests for merging a params JSON with train_yolo_obb CLI arguments."""

from trainit.train_yolo_obb import build_parser, explicit_cli_dests, merge_config_with_args


def _merge(argv, config):
    args = build_parser().parse_args(argv)
    return merge_config_with_args(args, config, explicit_cli_dests(argv))


def test_config_applies_to_options_not_given_on_cli():
    args = _merge([], {"device": "cpu", "epochs": 100, "lr0": 0.01})
    assert (args.device, args.epochs, args.lr0) == ("cpu", 100, 0.01)


def test_explicit_cli_value_wins_even_if_equal_to_default():
    default_epochs = build_parser().parse_args([]).epochs
    args = _merge(["--epochs", str(default_epochs)], {"epochs": default_epochs + 50})
    assert args.epochs == default_epochs


def test_auto_device_keeps_detected_default():
    default_device = build_parser().parse_args([]).device
    assert _merge([], {"device": "auto"}).device == default_device


def test_local_settings_disable_analytics_and_integrations(tmp_path, monkeypatch):
    import yaml

    from trainit.train_yolo_obb import setup_ultralytics_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path))  # restored after the test

    settings = yaml.safe_load(setup_ultralytics_settings().read_text())

    flags = ["sync", "clearml", "comet", "dvc", "hub", "mlflow", "neptune", "raytune", "tensorboard", "wandb"]
    assert all(settings[flag] is False for flag in flags)
