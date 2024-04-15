import logging
import datetime
import math

from rich.segment import Segment
from rich.style import Style
from rich.text import Text

from textual import events
from textual.app import App, ComposeResult
from textual.geometry import Offset, Region
from textual.reactive import reactive, var
from textual.widget import Widget
from textual.widgets import (
    Header,
    Footer,
    DataTable,
    TabbedContent,
    TabPane,
    DirectoryTree,
    Label,
    Input,
    Button,
    Static,
)
from textual.message import Message
from textual.containers import Container
from textual.strip import Strip

from pathlib import Path
from typing import Union

from emulator import Emulator
from memory import Memory

logging.basicConfig(
    filename=f"logs/{datetime.date.today()}.log",
    filemode="a",
    format="(%(asctime)s) %(levelname)s: %(message)s",
    level=logging.INFO,
)
logging.info("Started application")


class FileSelectorWidget(Widget):
    instruction_memory_path = reactive(None)
    data_memory_path = reactive(None)
    base_path = reactive("./uasm_files")

    class FileSelected(Message):
        """File(s) selected message."""

        def __init__(
            self, instruction_memory_path: Path, data_memory_path: Union[Path, None]
        ):
            self.instruction_memory_path = instruction_memory_path
            self.data_memory_path = data_memory_path
            super().__init__()

    def compose(self) -> ComposeResult:
        instruction_memory_directory_tree = DirectoryTree(
            self.base_path, id="instruction-memory-directory-tree"
        )
        data_memory_directory_tree = DirectoryTree(
            self.base_path, id="data-memory-directory-tree"
        )

        yield Label("Base Path")
        yield Container(
            Input(value=self.base_path, id="base-path-input"),
            Button("Load Path", id="load-path", variant="primary"),
            id="file-path-container",
        )
        yield Label("", id="load-path-label")
        yield Container(
            Container(Label("Instruction Memory"), instruction_memory_directory_tree),
            Container(Label("Data Memory"), data_memory_directory_tree),
            id="file-selector-container",
        )
        yield Button("Load Files", id="load-files-button", variant="primary")
        yield Label("", id="load-files-label")

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        if event.node.tree.id == "instruction-memory-directory-tree":
            self.instruction_memory_path = event.path
        else:
            self.data_memory_path = event.path

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "load-files-button":
            load_files_label = self.query_one("#load-files-label")

            if not self.instruction_memory_path:
                event.button.variant = "error"

                load_files_label.update(
                    Text("Instruction memory file not selected", style="red")
                )
                return

            self.post_message(
                self.FileSelected(self.instruction_memory_path, self.data_memory_path)
            )
            event.button.variant = "primary"
            load_files_label.update("")
        else:
            base_path_input = self.query_one("#base-path-input")
            load_path_label = self.query_one("#load-path-label")

            # If path doesn't exist, show an error
            if not Path(base_path_input.value).exists():
                event.button.variant = "error"
                load_path_label.update(Text("Base path does not exist", style="red"))
                return

            # If path isn't a directory, show an error
            if not Path(base_path_input.value).is_dir():
                event.button.variant = "error"
                load_path_label.update(
                    Text("Base path is not a directory", style="red")
                )
                return
            # Reset button to normal state, and update base path for directory trees
            event.button.variant = "primary"
            load_path_label.update("")

            instruction_memory_directory_tree = self.query_one(
                "#instruction-memory-directory-tree"
            )
            instruction_memory_directory_tree.path = Path(base_path_input.value)
            instruction_memory_directory_tree.reload()

            data_memory_directory_tree = self.query_one("#data-memory-directory-tree")
            data_memory_directory_tree.path = Path(base_path_input.value)
            data_memory_directory_tree.reload()


class EmulatorWidget(Widget):
    data_format = reactive(0)
    data_format_names = ["BIN", "HEX", "INT"]

    instruction_memory_path = reactive(None)
    running = False
    timer = None

    BINDINGS = [
        ("q", "step_backward", "Step Backward"),
        ("e", "step_forward", "Step Forward"),
        (
            "p",
            "toggle_automatic_execution",
            "Toggle Automatic Execution",
        ),
        ("r", "reset", "Reset"),
        ("t", "toggle_format", "Toggle Data Format"),
    ]

    class Updated(Message):
        def __init__(self, update_type: int = 0) -> None:
            # Update type 0 corresponds to change in data memory
            # Update type 1 corresponds to the program not auto-executing
            # Update type 2 corresponds to the program auto-executing
            self.update_type = update_type
            super().__init__()

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        instruction_memory = DataTable(id="instruction-memory")
        instruction_memory.cursor_type = "row"

        registers = DataTable(id="registers")
        registers.cursor_type = "row"

        data_memory = DataTable(id="data-memory")
        data_memory.cursor_type = "row"

        yield instruction_memory
        yield registers
        yield data_memory

    def on_mount(self) -> None:
        self.emulator = Emulator()
        instruction_memory = self.query_one("#instruction-memory")
        instruction_memory.add_columns(
            "IDX", "INSTR", f"VAL ({self.data_format_names[self.data_format]})"
        )
        instruction_memory.add_rows(
            [
                [idx * self.emulator.INSTR_WIDTH, "", self.format_data(val)]
                for idx, val in enumerate(self.emulator.instruction_memory.data)
            ]
        )

        registers = self.query_one("#registers")
        registers.add_columns(
            "REG", f"VAL ({self.data_format_names[self.data_format]})"
        )
        registers.add_rows(
            [
                [idx, self.format_data(val.data)]
                for idx, val in enumerate(self.emulator.registers)
            ]
        )

        data_memory = self.query_one("#data-memory")
        data_memory.add_columns(
            "IDX", f"VAL ({self.data_format_names[self.data_format]})"
        )
        data_memory.add_rows(
            [
                [idx * self.emulator.DATA_WIDTH, self.format_data(val)]
                for idx, val in enumerate(self.emulator.data_memory.data)
            ]
        )

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark

    def action_step_backward(self) -> None:
        self.emulator.step_backward()
        self.update_cursor()
        self.update_data()
        self.post_message(self.Updated())

    def action_step_forward(self) -> None:
        self.emulator.step_forward()
        self.update_cursor()
        self.update_data()
        self.post_message(self.Updated())

    def action_toggle_automatic_execution(self) -> None:
        if self.running:
            self.timer.pause()
            self.running = False
        else:
            if self.timer is None:
                self.timer = self.set_interval(0.01, self.action_step_forward)
            self.timer.resume()
            self.running = True
        self.post_message(self.Updated(self.running + 1))

    def action_reset(self):
        self.emulator.reset()
        self.update_data()
        self.post_message(self.Updated())

    def load_data(self, event: FileSelectorWidget.FileSelected):
        self.emulator.load_files(event.instruction_memory_path, event.data_memory_path)
        self.update_data(initial_data_load=True)
        self.post_message(self.Updated())

    def update_data(
        self,
        *,
        update_instruction_memory_data: bool = False,
        initial_data_load: bool = False,
    ):
        instruction_memory = self.query_one("#instruction-memory")
        data_memory = self.query_one("#data-memory")
        registers = self.query_one("#registers")

        if update_instruction_memory_data or initial_data_load:
            # Update instruction memory data
            for idx, val in enumerate(self.emulator.instruction_memory.data):
                cell_data = self.format_data(val)
                instruction_memory.update_cell_at(
                    (idx, 1),
                    f"{self.emulator.instruction_memory.unparsed_data[idx][0]}({', '.join([str(i) for i in self.emulator.instruction_memory.unparsed_data[idx][1:]]) if len(self.emulator.instruction_memory.unparsed_data[idx]) else ''})"
                    if self.emulator.instruction_memory.unparsed_data[idx]
                    else "",
                    update_width=initial_data_load,
                )
                instruction_memory.update_cell_at((idx, 2), cell_data)

            # Update instruction memory header label
            list(instruction_memory.columns.values())[
                -1
            ].label = f"VAL ({self.data_format_names[self.data_format]})"

        # Update data memory data
        for idx, val in enumerate(self.emulator.data_memory.data):
            cell_data = self.format_data(val)
            data_memory.update_cell_at((idx, 1), cell_data)

        # Update data memory header label
        list(data_memory.columns.values())[
            -1
        ].label = f"VAL ({self.data_format_names[self.data_format]})"

        # Update register data
        for idx, val in enumerate(self.emulator.registers):
            cell_data = self.format_data(val.data)
            registers.update_cell_at((idx, 1), cell_data)

        # Update register header label
        list(registers.columns.values())[
            -1
        ].label = f"VAL ({self.data_format_names[self.data_format]})"

    def update_cursor(self):
        instruction_memory = self.query_one("#instruction-memory")
        data_memory = self.query_one("#data-memory")
        registers = self.query_one("#registers")

        # Update data memory cursor
        for idx, val in enumerate(self.emulator.data_memory.data):
            cell_data = self.format_data(val)
            if data_memory.get_cell_at((idx, 1)) != cell_data:
                data_memory.move_cursor(row=idx)

        # Update register cursor
        for idx, val in enumerate(self.emulator.registers):
            cell_data = self.format_data(val.data)
            if registers.get_cell_at((idx, 1)) != cell_data:
                registers.move_cursor(row=idx)

        # Update selected row for instruction memory sections
        instruction_memory.move_cursor(
            row=self.emulator.program_counter // self.emulator.INSTR_WIDTH
        )

    def action_toggle_format(self):
        self.data_format = (self.data_format + 1) % 3
        self.update_data(update_instruction_memory_data=True)

    def format_data(self, data):
        if self.data_format == 0:
            return data.bin
        if self.data_format == 1:
            return data.hex
        return data.int


class Canvas(Widget):
    """
    Adapted from https://github.com/thomascrha/textual-game-of-life
    """

    COMPONENT_CLASSES: set = {
        "canvas--white-square",
        "canvas--black-square",
        "canvas--cursor-square",
    }

    DEFAULT_CSS: str = """
    Canvas .canvas--white-square {
        background: #FFFFFF;
    }
    Canvas .canvas--black-square {
        background: #000000;
    }
    Canvas > .canvas--cursor-square {
        background: darkred;
    }
    """
    ROW_HEIGHT: int = 2

    def __init__(
        self, data_memory: Memory = None, width: int = 64, height: int = 32
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.data_memory = data_memory

        self.canvas_matrix: "list[list[int]]" = [
            [0 for _ in range(self.width + 1)] for _ in range(self.height + 1)
        ]

    def initial_load(self) -> None:
        for row in range(self.height):
            for col in range(self.width):
                self.canvas_matrix[row][col] = self.data_memory.data[
                    (row * self.width + col) // (8 * self.data_memory.width)
                ][col % (self.data_memory.width * 8)]
        self.refresh()

    def update_output(self, debug_frame) -> None:
        address = debug_frame["data_memory"]["address"]
        row, col = divmod(8 * address, self.width)
        col //= self.data_memory.width * 8
        for idx, cell in enumerate(
            self.data_memory.data[
                address >> math.ceil(math.log2(self.data_memory.width))
            ]
        ):
            self.canvas_matrix[row][col * self.data_memory.width * 8 + idx] = cell
        self.refresh()

    @property
    def white(self) -> Style:
        return self.get_component_rich_style("canvas--white-square")

    @property
    def black(self) -> Style:
        return self.get_component_rich_style("canvas--black-square")

    def render_line(self, y: int) -> Strip:
        """Render a line of the widget. y is relative to the top of the widget."""
        row_index = y // int(self.ROW_HEIGHT / 2)

        if row_index >= self.height:
            return Strip.blank(self.size.width)

        def get_square_style(column: int, row: int) -> Style:
            """Get the cursor style at the given position on the checkerboard."""
            square_style = self.black
            # only update the squares that aren't out of range
            if len(self.canvas_matrix) > row and len(self.canvas_matrix[row]) > column:
                square_style = (
                    self.black if self.canvas_matrix[row][column] == 1 else self.white
                )

            return square_style

        segments = [
            Segment(" " * self.ROW_HEIGHT, get_square_style(column, row_index))
            for column in range(self.width)
        ]
        strip = Strip(segments)
        return strip


class EmulatorApp(App):
    """A Textual app to act as the UI for the Beta Emulator."""

    data_format = reactive(0)
    data_format_names = ["BIN", "HEX", "UINT"]

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("escape", "quit", "Quit"),
        ("1", "show_tab('file_selector')", "File Selector"),
        ("2", "show_tab('emulator')", "Emulator"),
        ("3", "show_tab('output')", "Output"),
    ]

    CSS_PATH = "emulator_app.css"

    SUB_TITLE = "Paused"

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with TabbedContent(initial="file_selector"):
            with TabPane("File Selector", id="file_selector"):
                yield FileSelectorWidget()
            with TabPane("Emulator", id="emulator"):
                yield EmulatorWidget()
            with TabPane("Output", id="output"):
                yield Canvas()
        yield Footer()

    def action_show_tab(self, tab: str) -> None:
        self.get_child_by_type(TabbedContent).active = tab

    def on_file_selector_widget_file_selected(
        self, event: FileSelectorWidget.FileSelected
    ) -> None:
        tabbed_content = self.query_one(TabbedContent)
        emulator_widget = self.query_one(EmulatorWidget)
        output_widget = self.query_one(Canvas)
        emulator_widget.load_data(event)
        tabbed_content.active = "emulator"
        output_widget.data_memory = emulator_widget.emulator.data_memory
        output_widget.initial_load()

    def on_emulator_widget_updated(self, event: EmulatorWidget.Updated) -> None:
        if event.update_type == 0:
            output_widget = self.query_one(Canvas)

            emulator_widget = self.query_one(EmulatorWidget)
            if output_widget.data_memory is None:
                output_widget.data_memory = emulator_widget.emulator.data_memory

            if (
                emulator_widget.emulator.history
                and "data_memory" in emulator_widget.emulator.history[-1]
            ):
                output_widget.update_output(emulator_widget.emulator.history[-1])
        else:
            self.sub_title = "Running" if event.update_type - 1 else "Paused"


if __name__ == "__main__":
    app = EmulatorApp()
    app.run()
