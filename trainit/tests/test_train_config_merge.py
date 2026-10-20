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
