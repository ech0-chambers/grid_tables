from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from io import StringIO
from enum import Enum, auto


class Alignment(Enum):
    LEFT = -1
    CENTER = 0
    RIGHT = 1
    DEFAULT = CENTER


class Char(Enum):
    CORNER = 0
    ROW = auto()
    COL = auto()
    HEADER = auto()
    ALIGN = auto()


@dataclass
class TableGrid:
    # everything is in order anticlockwise starting from right.
    # corners are top right, top left, bottom left, bottom right.
    # t_juncts are pointing right, up, left, down
    row: str
    col: str
    header: str
    alignment_char: Optional[str] = None
    corner: Optional[str] = None
    corners: List[str | None] = field(default_factory=lambda: [None] * 4)
    t_juncts: List[str | None] = field(default_factory=lambda: [None] * 4)
    cross: Optional[str] = None
    header_corners: List[str | None] = field(default_factory=lambda: [None] * 4)
    header_t_juncts: List[str | None] = field(default_factory=lambda: [None] * 4)
    header_cross: Optional[str] = None
    outline_row: Optional[str] = None
    outline_col: Optional[str] = None
    outline_corners: List[str | None] = field(default_factory=lambda: [None] * 4)
    outline_t_juncts: List[str | None] = field(default_factory=lambda: [None] * 4)
    # outline_cross can't exist
    outline_header_corners: List[str | None] = field(default_factory=lambda: [None] * 4)
    # this is a list of two, unlike others which are lists of 4:
    outline_header_t_juncts: List[str | None] = field(
        default_factory=lambda: [None] * 2
    )

    __LD = 0
    __RD = 1
    __RU = 2
    __UL = 3
    __RUD = 0
    __RUL = 1
    __ULD = 2
    __RLD = 3

    @property
    def alignment(self) -> str:
        if self.alignment_char is None:
            return self.header
        return self.alignment_char
    
    @alignment.setter
    def alignment(self, value: str | None) -> None:
        self.alignment_char = value

    def __post_init__(self) -> None:
        self.char_map: List[str | None] = [None] * 16
        self.header_char_map: List[str | None] = [None] * 16
        self.outline_char_map: List[str | None] = [None] * 16
        self.outline_header_char_map: List[str | None] = [None] * 16
        self.calc_character_maps()

    def calc_character_maps(self):
        # character maps are binary encoded based on which `legs` they have.
        #  0     1     1     1
        #  ↑     ↑     ↑     ↑
        # down  left   up   right  --> ┴
        if len(self.outline_header_t_juncts) == 2:
            self.outline_header_t_juncts = [
                self.outline_header_t_juncts[0],
                None,
                self.outline_header_t_juncts[1],
                None,
            ]
        self.char_map = [
            " ",  # these should never be accessed, since we don't have any way to end up with just a right leg
            " ",
            self.corners[self.__RU],
            " ",
            self.row,
            self.corners[self.__UL],
            self.t_juncts[self.__RUL],
            " ",
            self.corners[self.__RD],
            self.col,
            self.t_juncts[self.__RUD],
            self.corners[self.__LD],
            self.t_juncts[self.__RLD],
            self.t_juncts[self.__ULD],
            self.cross,
        ]
        self.header_char_map = [
            " ",
            " ",
            self.header_corners[self.__RU],
            " ",
            self.header,
            self.header_corners[self.__UL],
            self.header_t_juncts[self.__RUL],
            " ",
            self.header_corners[self.__RD],
            self.col,
            self.header_t_juncts[self.__RUD],
            self.header_corners[self.__LD],
            self.header_t_juncts[self.__RLD],
            self.header_t_juncts[self.__ULD],
            self.header_cross,
        ]
        self.outline_char_map = [
            " ",
            " ",
            self.outline_corners[self.__RU],
            " ",
            self.outline_row,
            self.outline_corners[self.__UL],
            self.outline_t_juncts[self.__RUL],
            " ",
            self.outline_corners[self.__RD],
            self.outline_col,
            self.outline_t_juncts[self.__RUD],
            self.outline_corners[self.__LD],
            self.outline_t_juncts[self.__RLD],
            self.outline_t_juncts[self.__ULD],
            " ",
        ]
        self.outline_header_char_map = [
            " ",
            " ",
            self.outline_header_corners[self.__RU],
            " ",
            " ",
            self.outline_header_corners[self.__UL],
            " ",
            " ",
            self.outline_header_corners[self.__RD],
            " ",
            self.outline_header_t_juncts[self.__RUD],
            self.outline_header_corners[self.__LD],
            " ",
            self.outline_header_t_juncts[self.__ULD],
            " ",
        ]

    @staticmethod
    def get_from_char_maps(index: int, *char_maps: List[str | None]) -> str | None:
        char: str | None = None
        for c_map in char_maps:
            try:
                char = c_map[index]
            except Exception as e:
                print(f"Failed to fetch character {index} from {c_map}")
                raise e
            if char is not None:
                break
        if char is None:
            raise ValueError(f"Failed to fetch character {index} from {c_map}, got None")
        return char

    def get_intersection_char(self, directions: int) -> str | None:
        # `directions` is binary encoded to match maps
        # 0       0      1     0     1     0
        # ↑       ↑      ↑     ↑     ↑     ↑
        # outline header down  left  up    right
        self.calc_character_maps()

        is_header = (directions >> 4) & 1 == 1
        is_outline = (directions >> 5) & 1 == 1
        directions = directions & 15

        c_maps = [self.char_map]
        if is_outline:
            c_maps.insert(0, self.outline_char_map)
        if is_header:
            c_maps.insert(0, self.header_char_map)
        if is_header and is_outline:
            c_maps.insert(0, self.outline_header_char_map)

        char = self.get_from_char_maps(directions - 1, *c_maps)
        return char

    def apply(self, grid: np.ndarray, header_position: int = -1) -> None:
        # edit the grid in place

        if self.corner is not None:
            # this is a simple grid, we don't need to do too much
            grid[grid == Char.CORNER] = self.corner
            grid[grid == Char.ROW] = self.row
            grid[grid == Char.COL] = self.col
            grid[grid == Char.HEADER] = self.header
            grid[grid == Char.ALIGN] = self.alignment
            return

        grid_right = np.roll(grid, 1, axis=1)
        grid_left = np.roll(grid, -1, axis=1)
        grid_up = np.roll(grid, -1, axis=0)
        grid_down = np.roll(grid, 1, axis=0)

        char_indices = (
            grid == Char.CORNER
        ) * (  # we only care about the corner characters for this
            np.roll(  # check if the character to the right of the corner is still a row character of some kind
                (grid_right == Char.ROW)
                + (grid_right == Char.HEADER)
                + (grid_right == Char.ALIGN),
                -2,
                axis=1,
            )
            * 1
            + np.roll(grid_up == Char.COL, 2, axis=0) * 2  # character above is column
            + np.roll(  # left
                (grid_left == Char.ROW)
                + (grid_left == Char.HEADER)
                + (grid_left == Char.ALIGN),
                2,
                axis=1,
            )
            * 4
            + np.roll(grid_down == Char.COL, -2, axis=0) * 8  # below
        )

        if header_position >= 0:
            char_indices[
                header_position, char_indices[header_position, :] > 0
            ] += 0b010000

        top = char_indices[0, :]
        bottom = char_indices[-1, :]
        left = char_indices[1:-1, 0]
        right = char_indices[1:-1, -1]
        for side in [top, right, bottom, left]:
            side[side > 0] += 0b100000

        # turn the calculated character indices into the corresponding characters
        grid[char_indices > 0] = np.vectorize(self.get_intersection_char)(
            char_indices[char_indices > 0]
        )

        top = grid[0, :]
        bottom = grid[-1, :]
        left = grid[1:-1, 0]
        right = grid[1:-1, -1]
        if self.outline_row is not None:
            top[top == Char.ROW] = self.outline_row
            bottom[bottom == Char.ROW] = self.outline_row
        if self.outline_col is not None:
            left[left == Char.COL] = self.outline_col
            right[right == Char.COL] = self.outline_col

        grid[grid == Char.CORNER] = self.corner
        grid[grid == Char.ROW] = self.row
        grid[grid == Char.COL] = self.col
        grid[grid == Char.HEADER] = self.header
        grid[grid == Char.ALIGN] = self.alignment

    def copy(self) -> 'TableGrid':
        _new = TableGrid(self.row, self.col, self.header, alignment_char = self.alignment_char, corner = self.corner)
        _new.corners = self.corners
        _new.t_juncts = self.t_juncts
        _new.cross = self.cross
        _new.header_corners = self.header_corners
        _new.header_t_juncts = self.header_t_juncts
        _new.header_cross = self.header_cross
        _new.outline_row = self.outline_row
        _new.outline_col = self.outline_col
        _new.outline_corners = self.outline_corners
        _new.outline_t_juncts = self.outline_t_juncts
        _new.outline_header_corners = self.outline_header_corners
        _new.outline_header_t_juncts = self.outline_header_t_juncts

        return _new

__markdown_table = TableGrid("-", "|", "=", corner="+", alignment_char=":")

__unicode_table = TableGrid("─", "│", "━")
__unicode_table.corners = list("┐┌└┘")
__unicode_table.t_juncts = list("├┴┤┬")
__unicode_table.cross = "┼"
__unicode_table.header_corners = list("┑┍┕┙")
__unicode_table.header_t_juncts = list("┝┷┥┯")
__unicode_table.header_cross = "┿"

__double_table = __unicode_table.copy()
__double_table.header = "═"
__double_table.header_corners = list("╕╒╘╛")
__double_table.header_t_juncts = list("╞╧╡╤")
__double_table.header_cross = "╪"

__double_outline_table = __double_table.copy()
__double_outline_table.outline_corners = list("╗╔╚╝")
__double_outline_table.outline_t_juncts = list("╟╧╢╤")
__double_outline_table.outline_header_corners = list("╗╔╚╝")
__double_outline_table.outline_header_t_juncts = list("╠╣")
__double_outline_table.outline_row = "═"
__double_outline_table.outline_col = "║"

__rounded_table = __unicode_table.copy()
__rounded_table.corners = list("╮╭╰╯")

TABLES = {
    "markdown": __markdown_table,
    "unicode": __unicode_table,
    "double_header": __double_table,
    "rounded": __rounded_table,
    "outline": __double_outline_table,
}

def ceil(x: float) -> int:
    return int(x) + (x % 1 > 0)


def soft_wrap(
    text: str, width: int, alignment: Optional[str | Alignment] = None
) -> List[str]:
    if alignment is None:
        alignment = Alignment.LEFT
    if isinstance(alignment, str):
        alignment = Alignment[alignment.upper()]
    if len(text) < width:
        return [
            f"{text.rstrip(): {'<' if alignment == Alignment.LEFT else '^' if alignment == Alignment.CENTER else '>'}{width}s}"
        ]
    if width < 2:
        raise ValueError(f'Cannot wrap text "{text}" to a width of {width}')
    text = text.replace("\n", " \n ")
    words = text.split(" ")
    words = [w for w in words if w]
    lines = []
    line = ""
    while len(words) > 0:
        word = words.pop(0)
        if word == "\n":
            lines.append(line)
            line = ""
            continue
        if len(line) == width - 1:
            lines.append(line)
            line = ""
            words.insert(0, word)
            continue
        if len(word) > width:
            if len(line) > 0:
                line += " "
            start, end = word[: width - len(line)], word[width - len(line) :]
            line += start
            words.insert(0, end)
            continue
        if len(word) + ((len(line) + 1) if len(line) > 0 else 0) > width:
            lines.append(line)
            line = ""
            words.insert(0, word)
            continue
        line += (" " if len(line) > 0 else "") + word
    lines.append(line)
    lines = [
        f"{line.rstrip(): {'<' if alignment == Alignment.LEFT else '^' if alignment == Alignment.CENTER else '>'}{width}s}"
        for line in lines
    ]
    return lines


def regex_escaped(c: str) -> str:
    # Escape characters that have special meaning in regex
    out = StringIO()
    for char in c:
        if char in r".^$*+?{}[]\|()":
            out.write("\\")
        out.write(char)
    return out.getvalue()


def text_width_wrapped(text, num_rows):
    """
    Calculates the minimum horizontal width needed to display the given text on
    at most the specified number of rows by wrapping at spaces.

    Args:
        text: The text to wrap.
        num_rows: The maximum number of rows allowed.

    Returns:
        The minimum horizontal width required.
    """

    # Handle trivial cases (empty text or one row allowed)
    if not text or num_rows == 1:
        return len(text)

    max_word_length = max([len(s) for s in text.replace("\n", " \n ").split(" ")])
    min_width = ceil(len(text) / num_rows) + max_word_length

    lines = len(soft_wrap(text, width=min_width))

    while lines <= num_rows:
        min_width -= 1
        if min_width < max_word_length or min_width < 2:
            break
        lines = len(soft_wrap(text, width=min_width))

    min_width += 1

    return min_width


@dataclass
class Cell:
    text: str
    col_span: int = 1
    row_span: int = 1
    min_width: int = -1
    min_height: int = -1
    width: int = -1
    height: int = -1
    start_col: int = -1
    start_row: int = -1

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            self.text = str(self.text)

    @property
    def end_col(self):
        return self.start_col + self.col_span - 1

    @property
    def end_row(self):
        return self.start_row + self.row_span - 1

    def is_in_col(self, col_num: int) -> bool:
        return self.start_col <= col_num <= self.end_col

    def is_in_row(self, row_num: int) -> bool:
        return self.start_row <= row_num <= self.end_row

    def recalc_width(self, col_widths: List[int]) -> None:
        width = (
            sum(col_widths[self.start_col : self.end_col + 1]) + (self.col_span - 1) * 3
        )
        self.width = width

    def recalc_height(self, row_heights: List[int]) -> None:
        height = sum(row_heights[self.start_row : self.end_row + 1]) + (
            self.row_span - 1
        )
        self.height = height

    def as_array(
        self, padded: bool = False, alignment: Optional[str | Alignment] = None
    ) -> np.ndarray:
        wrapped = np.array(
            [list(line) for line in soft_wrap(self.text, self.width, alignment)]
        )

        if padded:
            blank_col = np.zeros((wrapped.shape[0], 1), dtype=np.dtype("<U1"))
            blank_col[:, :] = " "
            wrapped = np.concatenate((blank_col, wrapped, blank_col), axis=1)

        if len(wrapped) == self.height - 1:
            blank_line = np.zeros((1, wrapped.shape[1]), dtype=np.dtype("<U1"))
            blank_line[:, :] = " "
            wrapped = np.concatenate((wrapped, blank_line), axis=0)
        elif len(wrapped) < self.height - 1:
            lines_before = (self.height - len(wrapped)) // 2
            lines_after = (self.height - len(wrapped) + 1) // 2
            blank_before = np.zeros(
                (lines_before, wrapped.shape[1]), dtype=np.dtype("<U1")
            )
            blank_before[:, :] = " "
            blank_after = np.zeros(
                (lines_after, wrapped.shape[1]), dtype=np.dtype("<U1")
            )
            blank_after[:, :] = " "
            wrapped = np.concatenate((blank_before, wrapped, blank_after), axis=0)

        return wrapped


def n_column_width(col_width: int, col_span: int) -> int:
    return col_width * col_span + (col_span - 1) * 3


def text_width_to_col_width(text_width: int, col_span: int) -> int:
    return ceil((text_width - (col_span - 1) * 3) / col_span)


def n_row_height(row_height: int, row_span: int) -> int:
    return row_height * row_span + (row_span - 1)


def text_height_to_row_height(text_height: int, row_span: int) -> int:
    return ceil((text_height - (row_span - 1)) / row_span)


def grid_table(
    table: List[str | Tuple[str, int, int]],
    header_rows: int = 1,
    max_cell_width: int = 15,
    alignments: Optional[Alignment | List[Alignment] | str | List[str]] = None,
    min_cell_width: int = 1,
    min_cell_height: int = 1,
    align_text_in_output: bool = False,  # disabled by default since it wouldn't be correctly interpreted by markdown
    table_grid: Optional[TableGrid] = None,
) -> str:
    if alignments is None:
        alignments = Alignment.CENTER
    if isinstance(alignments, str):
        try:
            alignments = Alignment[alignments.upper()]
        except:
            raise ValueError(
                f'Unrecognised alignment "{alignments}". Acceptable values are "left", "center", and "right".'
            )

    if table_grid is None:
        table_grid = __markdown_table

    _table: List[List[Cell]] = [
        [
            Cell(*cell) if isinstance(cell, (list, tuple)) else Cell(str(cell))
            for cell in row
        ]
        for row in table
    ]

    num_columns = sum(cell.col_span for cell in _table[0])
    if isinstance(alignments, Alignment):
        alignments = [alignments] * num_columns
    elif isinstance(alignments, (list, tuple)):
        alignments = [
            Alignment[a.upper()] if isinstance(a, str) else a for a in alignments
        ]

    if not len(alignments) == num_columns:
        raise ValueError(
            f"Found {num_columns} columns but only {len(alignments)} alignments."
        )

    table_cells = [j for sub in _table for j in sub]
    cells: List[Cell] = []
    # track multirow cells -- if skip_columns[col] > 0, it's occupied by a cell from a row above.
    skip_columns = [0] * num_columns

    r = 0
    while len(table_cells) > 0:
        for c in range(num_columns):
            if skip_columns[c] > 0:
                continue
            if len(table) == 0:
                # if the last cell in the last row spans multiple columns, this is possible
                continue
            cell = table_cells.pop(0)
            cell.start_row = r
            cell.start_col = c
            cells.append(cell)
            for ci in range(c, c + cell.col_span):
                skip_columns[ci] = cell.row_span
        skip_columns = [max(0, s - 1) for s in skip_columns]
        r += 1

    num_rows = sum(cell.row_span for cell in cells if cell.is_in_col(0))

    # calculate the max cell width for all cells
    max_col_width = 0
    for cell in cells:
        text_width = text_width_wrapped(cell.text, cell.row_span * 2 - 1)
        max_col_width = max(
            max_col_width, ceil((text_width - (cell.col_span - 1) * 3) / cell.col_span)
        )

    max_col_width = max(min(max_col_width, max_cell_width), min_cell_width)

    # calculate the max cell height for all cells
    max_row_height = 1
    for cell in cells:
        cell.width = n_column_width(max_col_width, cell.col_span)
        if len(cell.text) < max_col_width:
            continue  # this will be 1 line, which can never be more than the max height
        text_height = len(soft_wrap(cell.text, cell.width))
        max_row_height = max(
            max_row_height, ceil((text_height - cell.row_span + 1) / (cell.row_span))
        )

    max_row_height = max(max_row_height, min_cell_height)

    for cell in cells:
        cell.height = n_row_height(max_row_height, cell.row_span)

    # now calculate the cells minimum height
    for cell in cells:
        if cell.row_span == 1:
            if len(cell.text) <= n_column_width(max_col_width, cell.col_span):
                cell.min_height = n_row_height(1, cell.row_span)
                cell.min_width = len(cell.text)
                continue
            wrapped = soft_wrap(cell.text, n_column_width(max_col_width, cell.col_span))
            cell.min_height = max(len(wrapped), n_row_height(1, cell.row_span))
            cell.min_width = max(len(line.rstrip()) for line in wrapped)
            continue
        min_height = n_row_height(1, cell.row_span)
        width = text_width_wrapped(cell.text, min_height)
        if width > n_column_width(max_col_width, cell.col_span):
            wrapped = soft_wrap(cell.text, n_column_width(max_col_width, cell.col_span))
            cell.min_height = max(len(wrapped), n_row_height(1, cell.row_span))
            cell.min_width = max(len(line.rstrip()) for line in wrapped)
            continue
        cell.min_height = min_height
        cell.min_width = width

    col_widths = [max_col_width] * num_columns
    row_heights = [max_row_height] * num_rows

    # Now shrink rows and columns as much as possible.
    for c in range(num_columns):
        col_widths[c] = max(
            max(
                text_width_to_col_width(cell.min_width, cell.col_span)
                for cell in cells
                if cell.is_in_col(c)
            ),
            min_cell_width,
        )

    for r in range(num_rows):
        row_heights[r] = max(
            max(
                text_height_to_row_height(cell.min_height, cell.row_span)
                for cell in cells
                if cell.is_in_row(r)
            ),
            min_cell_height,
        )

    # TODO: try shrinking each column, which would necessitate increasing some rows, but might reduce overall area (i.e, avoid short but wide tables in favour of more square ones.)

    # Readjust cell dimensions based on new column widths etc.
    for cell in cells:
        cell.recalc_width(col_widths)
        cell.recalc_height(row_heights)

    grid = construct_grid(col_widths, row_heights, header_rows, alignments)

    # we now have an empty grid with all row and column separators, including those which would go through multirow or multicolumn cells.
    # overwrite the relevant areas with each cell.
    # Since we pad the cells to their full dimensions, this will remove any unwanted row and column separators

    for cell in cells:
        start_x = sum(col_widths[: cell.start_col]) - 1 + (cell.start_col * 3 + 2)
        start_y = sum(row_heights[: cell.start_row]) + cell.start_row + 1
        alignment = Alignment.LEFT
        if align_text_in_output:
            if cell.col_span == 1:
                alignment = alignments[cell.start_col]
            else:
                alignment = get_multicol_alignment(
                    alignments[cell.start_col : cell.start_col + cell.col_span]
                )
        cell_array: np.ndarray = cell.as_array(padded=True, alignment=alignment)
        shape = cell_array.shape
        grid[start_y : start_y + shape[0], start_x : start_x + shape[1]] = cell_array

    if header_rows > 0:
        row_positions = np.cumsum([0] + row_heights)
        row_positions[1:] += [i + 1 for i in range(len(row_heights))]
        header_position = row_positions[header_rows]
    else:
        header_position = -1

    table_grid.apply(grid, header_position=header_position)

    return "\n".join("".join(c for c in row) for row in grid)


def construct_grid(
    col_widths: List[int],
    row_heights: List[int],
    header_rows: int,
    alignments: List[Alignment],
) -> np.ndarray:
    total_width = (
        sum(col_widths) + 3 * len(col_widths) + 1
    )  # 3 characters on each column separator (3 * len(col_widths)), one more column separator than column (+3), only two characters for the leftmost and rightmost column (-2)
    total_height = sum(row_heights) + len(row_heights) + 1

    grid = np.zeros((total_height, total_width), dtype=np.dtype("object"))
    grid[:, :] = " "

    col_positions = np.cumsum([0] + col_widths)
    col_positions[1:] += [3 * (i + 1) for i in range(len(col_widths))]

    row_positions = np.cumsum([0] + row_heights)
    row_positions[1:] += [i + 1 for i in range(len(row_heights))]

    for col in col_positions:
        grid[:, col] = Char.COL

    for row in row_positions:
        grid[row, :] = Char.ROW
        grid[row, col_positions] = Char.CORNER

    if header_rows > 0:
        header_position = row_positions[header_rows]
        grid[header_position, :] = Char.HEADER
        grid[header_position, col_positions] = Char.CORNER

        for c, alignment in enumerate(alignments):
            if alignment == Alignment.CENTER:
                grid[header_position, col_positions[c] + 1] = Char.ALIGN
                grid[header_position, col_positions[c + 1] - 1] = Char.ALIGN
            elif alignment == Alignment.LEFT:
                grid[header_position, col_positions[c] + 1] = Char.ALIGN
            elif alignment == Alignment.RIGHT:
                grid[header_position, col_positions[c + 1] - 1] = Char.ALIGN
            else:
                raise ValueError(
                    f'Unknown alignment type "{alignment}" for column {c+1}.'
                )

    grid[0, [0, -1]] = Char.CORNER
    grid[-1, [0, -1]] = Char.CORNER

    return grid


def get_multicol_alignment(alignments: List[Alignment]) -> Alignment:
    if len(alignments) == 2 and alignments[0] == Alignment.LEFT:
        return Alignment.LEFT
    
    als = [al.value for al in alignments]
    alignment = sum(als)
    if alignment > 0:
        return Alignment.RIGHT
    if alignment < 0:
        return Alignment.LEFT
    return Alignment.CENTER
