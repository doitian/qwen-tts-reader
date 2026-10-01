from textual.await_complete import AwaitComplete
from textual.command import CommandPalette


class DismissOnce:
    """Ignore queued dismissals for a modal that's already off the screen stack."""

    _dismissed = False

    def dismiss(self, result=None) -> AwaitComplete:
        if (
            self._dismissed
            or not self.app.is_running
            or len(self.app.screen_stack) <= 1
            or self.app.screen is not self
        ):
            return AwaitComplete.nothing()
        self._dismissed = True
        return super().dismiss(result)


class ReaderCommandPalette(DismissOnce, CommandPalette):
    pass
