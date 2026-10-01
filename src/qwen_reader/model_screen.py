from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static

from .config import MODEL_VOICES, Settings
from .screens import DismissOnce
from .voices import VOICES

CUSTOM_VOICE = "__custom__"


class ModelScreen(DismissOnce, ModalScreen[tuple[str, str] | None]):
    BINDINGS = [("escape", "cancel", "Cancel")]
    DEFAULT_CSS = """
    ModelScreen { align: center middle; background: $background 70%; }
    #model-dialog { width: 68; max-width: 95%; height: auto; padding: 1 2;
        background: $surface; border: round $primary; }
    #model-dialog Static { height: auto; }
    #model-dialog Select, #model-dialog Input { margin-bottom: 0; }
    #voice-error { color: $error; }
    #model-buttons { height: 3; align: right middle; }
    #model-buttons Button { margin-left: 1; }
    """

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.selected_model = settings.model

    def voice_options(self, model: str) -> list[tuple[str, str]]:
        return [
            (f"Default · {MODEL_VOICES[model]}", ""),
            *[(f"{name} · {voice}", voice) for name, voice in VOICES[model]],
            ("Custom voice ID…", CUSTOM_VOICE),
        ]

    def compose(self) -> ComposeResult:
        with Vertical(id="model-dialog"):
            yield Static("Choose a streaming model")
            yield Select(
                [(model, model) for model in MODEL_VOICES],
                value=self.settings.model,
                allow_blank=False,
                id="model-choice",
            )
            yield Static("Voice")
            known_voices = {voice for _, voice in VOICES[self.settings.model]}
            custom = bool(self.settings.voice and self.settings.voice not in known_voices)
            yield Select(
                self.voice_options(self.settings.model),
                value=CUSTOM_VOICE if custom else self.settings.voice,
                allow_blank=False,
                id="voice-choice",
            )
            field = Input(
                value=self.settings.voice if custom else "",
                placeholder="Enter custom voice ID",
                id="voice-custom",
            )
            field.display = custom
            yield field
            error = Static("", id="voice-error")
            error.display = False
            yield error
            yield Static("Applying changes stops current playback. Press Read to start again.")
            with Horizontal(id="model-buttons"):
                yield Button("Cancel", id="model-cancel")
                yield Button("Apply", id="model-apply", variant="primary")

    @on(Select.Changed, "#model-choice")
    def model_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, str) and event.value != self.selected_model:
            self.selected_model = event.value
            voice = self.query_one("#voice-choice", Select)
            voice.set_options(self.voice_options(event.value))
            voice.value = ""
            self.query_one("#voice-custom", Input).value = ""
            self.query_one("#voice-custom", Input).display = False
            self.query_one("#voice-error").display = False

    @on(Select.Changed, "#voice-choice")
    def voice_changed(self, event: Select.Changed) -> None:
        custom = event.value == CUSTOM_VOICE
        field = self.query_one("#voice-custom", Input)
        field.display = custom
        self.query_one("#voice-error").display = False
        if custom:
            field.focus()

    @on(Button.Pressed, "#model-apply")
    def apply(self) -> None:
        model = self.query_one("#model-choice", Select).value
        voice = self.query_one("#voice-choice", Select).value
        if voice == CUSTOM_VOICE:
            voice = self.query_one("#voice-custom", Input).value.strip()
            if not voice:
                error = self.query_one("#voice-error", Static)
                error.update("Enter a custom voice ID or choose a listed voice.")
                error.display = True
                return
        if isinstance(model, str) and isinstance(voice, str):
            self.dismiss((model, voice))

    @on(Button.Pressed, "#model-cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)
