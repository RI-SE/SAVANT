from types import SimpleNamespace

from edit.frontend.states.frontend_state import FrontendState


def _issue(object_id: str, severity: str):
    return SimpleNamespace(object_id=object_id, severity=severity)


def test_confidence_issue_frames_set_split_by_severity():
    state = FrontendState()
    issues = {
        10: [_issue("obj-warn", "warning"), _issue("obj-both", "warning")],
        11: [_issue("obj-err", "error")],
        12: [_issue("obj-both", "warning"), _issue("obj-both", "error")],
    }

    state.set_confidence_issues(issues)

    assert state.warning_frames() == {10, 12}
    assert state.error_frames() == {11, 12}


def test_confidence_issue_frames_clear_when_removed():
    state = FrontendState()
    state.set_confidence_issues({5: [_issue("obj", "warning")]})
    assert state.warning_frames() == {5}

    # Removing the frame from the issues map should clear the cached sets.
    state.set_confidence_issues({})

    assert state.warning_frames() == set()
    assert state.error_frames() == set()
