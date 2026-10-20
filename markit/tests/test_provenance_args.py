"""Tests for the provenance record built by run_markit."""

from markit.run_markit import build_arguments_string, build_parser, provenance_inputs


def _args(*extra):
    return build_parser().parse_args(["--input", "v.mp4", "--output_json", "o.json", *extra])


def test_every_option_is_recorded_except_secrets():
    args = _args("--vlm", "--vlm-api-key", "SECRET", "--min-duration", "7")
    recorded = build_arguments_string(args)

    assert "SECRET" not in recorded and "--vlm-api-key" not in recorded
    assert "--vlm " in recorded + " " and "--min-duration 7" in recorded
    option_flags = {
        a.option_strings[0] for a in build_parser()._actions
        if a.option_strings and a.dest not in ("help", "vlm_api_key")
        and a.nargs != 0 and a.default is not None
    }
    assert all(flag in recorded for flag in option_flags)


def test_all_input_files_are_listed():
    args = _args("--exclusion-mask", "m.png", "--drone-info", "d.json", "--visual-markers", "vm.csv",
                 "--vlm", "--vlm-prompts", "p.json")
    paths, formats = provenance_inputs(args)

    assert {"v.mp4", "m.png", "d.json", "vm.csv", "p.json"} <= set(paths)
    assert len(paths) == len(formats)
