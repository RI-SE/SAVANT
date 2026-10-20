"""Undo/redo stack management."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from .gateways import GatewayHolder
from .commands import UndoableCommand


@dataclass
class UndoRedoManager:
    """Simple two-stack undo/redo manager."""

    _undo_stack: list[UndoableCommand] = field(default_factory=list)
    _redo_stack: list[UndoableCommand] = field(default_factory=list)

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def execute(self, command: UndoableCommand, context: GatewayHolder) -> None:
        command.do(context)
        self._undo_stack.append(command)
        self._redo_stack.clear()

    def undo(self, context: GatewayHolder) -> Optional[UndoableCommand]:
        if not self._undo_stack:
            return None
        # Pop only after success, so a failing command stays retryable instead of vanishing
        command = self._undo_stack[-1]
        command.undo(context)
        self._redo_stack.append(self._undo_stack.pop())
        return command

    def redo(self, context: GatewayHolder) -> Optional[UndoableCommand]:
        if not self._redo_stack:
            return None
        command = self._redo_stack[-1]
        command.do(context)
        self._undo_stack.append(self._redo_stack.pop())
        return command
