# grid_tables

A simple package for formatting markdown grid tables with multi-row and multi-column cells.

## Examples

The default behaviour is to output markdown-style grid tables with a single header row and centre-aligned columns. These are fully compatible with pandoc.

```python
import grid_tables as gt

table = [
    [("Grid Tables", 2, 1), ("Are Beautiful", 2, 1)],
    [("Easy to read", 2, 1), ("In code and docs", 2, 1)],
    [("Exceptionally flexible and powerful", 4, 1)],
    [f"Col {i+1}" for i in range(4)]
]

print(gt.grid_table(table))
```

```markdown
+---------+---------+---------+---------+
| Grid Tables       | Are Beautiful     |
+:=======:+:=======:+:=======:+:=======:+
| Easy to read      | In code and docs  |
+---------+---------+---------+---------+
| Exceptionally flexible and powerful   |
+---------+---------+---------+---------+
| Col 1   | Col 2   | Col 3   | Col 4   |
+---------+---------+---------+---------+
```

Other table styles are also available for pretty printing. Line wrapping is automatic for all outputs and respects newlines in the input.

```python
import grid_tables as gt

table = [
    [("Property", 2, 1), "Earth"],
    [("Temperature 1961-1990", 1, 3), "Min", "-89.2 °C"],
    ["Mean", "14 °C"],
    ["Max", "56.7 °C"]
]

print(gt.grid_table(
    table,
    alignments = ["left", "right", "left"],
    table_grid = gt.TABLES["outline"],
    min_cell_width = 6,
    align_text_in_output = True
))
```

```
╔══════════════════════╤══════════╗
║ Property             │ Earth    ║
╠═════════════╤════════╪══════════╣
║             │    Min │ -89.2 °C ║
║ Temperature ├────────┼──────────╢
║ 1961-1990   │   Mean │ 14 °C    ║
║             ├────────┼──────────╢
║             │    Max │ 56.7 °C  ║
╚═════════════╧════════╧══════════╝
```

### TableGrid Types

#### Markdown (Default)

```markdown
+-------------+--------+----------+
| Property             | Earth    |
+:============+=======:+:=========+
|             |    Min | -89.2 °C |
+ Temperature +--------+----------+
| 1961-1990   |   Mean | 14 °C    |
+             +--------+----------+
|             |    Max | 56.7 °C  |
+-------------+--------+----------+
```

#### Unicode

```
┌──────────────────────┬──────────┐
│ Property             │ Earth    │
┝━━━━━━━━━━━━━┯━━━━━━━━┿━━━━━━━━━━┥
│             │    Min │ -89.2 °C │
│ Temperature ├────────┼──────────┤
│ 1961-1990   │   Mean │ 14 °C    │
│             ├────────┼──────────┤
│             │    Max │ 56.7 °C  │
└─────────────┴────────┴──────────┘
```

#### Double Header

```
┌──────────────────────┬──────────┐
│ Property             │ Earth    │
╞═════════════╤════════╪══════════╡
│             │    Min │ -89.2 °C │
│ Temperature ├────────┼──────────┤
│ 1961-1990   │   Mean │ 14 °C    │
│             ├────────┼──────────┤
│             │    Max │ 56.7 °C  │
└─────────────┴────────┴──────────┘
```

#### Rounded

```
╭──────────────────────┬──────────╮
│ Property             │ Earth    │
┝━━━━━━━━━━━━━┯━━━━━━━━┿━━━━━━━━━━┥
│             │    Min │ -89.2 °C │
│ Temperature ├────────┼──────────┤
│ 1961-1990   │   Mean │ 14 °C    │
│             ├────────┼──────────┤
│             │    Max │ 56.7 °C  │
╰─────────────┴────────┴──────────╯
```

#### Outline

```
╔══════════════════════╤══════════╗
║ Property             │ Earth    ║
╠═════════════╤════════╪══════════╣
║             │    Min │ -89.2 °C ║
║ Temperature ├────────┼──────────╢
║ 1961-1990   │   Mean │ 14 °C    ║
║             ├────────┼──────────╢
║             │    Max │ 56.7 °C  ║
╚═════════════╧════════╧══════════╝
```

## Usage

The `grid_table` function takes the following arguments:

`table: List[List[str | Tuple[str, int, int]]]`
: The table to format. This should be a nested list, with each inner list representing one row of the table. Each element in the row should be either a string (for a single column, single row cell), or a tuple of the format `(text, column_span, row_span)`. Cells should be included in the first row in which they appear *only* -- you do not need to repeat a multi-row cell in the next row, nor should you leave an empty cell like one might with a $\LaTeX$ table for example.

`header_rows: int` (Optional, default 1)
: The number of rows which should form the table header. Set to `-1` for no header.

`max_cell_width: int` (Optional, default 15)
: The maximum width a single-column cell can take. Longer cell contents will automatically be wrapped to this width (or smaller).

`alignments: Alignment | List[Alignment] | str | List[str]` (Optional, default `Alignment.CENTER`)
: The alignment of each column. If a single value is given, all columns will have the same alignment. If multiple values are given, this must match the number of columns. Acceptable values are `"left"`, "`center`" (note the US spelling), and "`right`". Additionally, an enum is provided with these values, `Alignment.LEFT`, `Alignment.CENTER`, and `Alignment.RIGHT`.

`min_cell_width: int` (Optional, default 1)
: Cells will be shrunk to fit their contents. Setting this value sets a minimum width for a single-column cell, beyond which the columns will not be shrunk.

`min_cell_height: int` (Optional, default 1)
: Like `min_cell_width`, this sets the minimum height for a single-row cell.

`align_text_in_output: bool` (Optional, default `False`)
: If this is set `True`, the cell contents will be aligned based on the column alignment, making an educated guess at the best alignment for cells which span multiple columns. This is not recommended for markdown output, as it can interfere with how the cell contents is interpreted since grid tables can contain arbitrary markdown blocks.

`table_grid: TableGrid` (Optional, default `TABLES["markdown"]`)
: This determines the format for the table dividers. See [TableGrid Types](#tablegrid-types) above for examples of the builtin types. 