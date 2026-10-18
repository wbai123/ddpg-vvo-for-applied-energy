"""Format research results as compact escaped HTML tables in notebooks."""

import html

import numpy as np
from IPython.display import HTML, display


def format_value(value):
    """Return a compact display string for a scalar or sequence."""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    if isinstance(value, (list, tuple, np.ndarray)):
        return "[" + ", ".join(format_value(item) for item in value) + "]"
    return str(value)


def show_table(rows, columns=None, *, title=None, max_height=None):
    """Display rows as an escaped HTML table, optionally in a bounded viewport."""
    rows = list(rows)
    if not rows:
        display(HTML(f"<p><em>{html.escape(title or 'No rows')}</em></p>"))
        return
    columns = list(columns or rows[0].keys())
    caption = f"<caption>{html.escape(title)}</caption>" if title else ""
    head = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(format_value(row.get(column, '')))}</td>" for column in columns
        )
        + "</tr>"
        for row in rows
    )
    table = (
        f'<table class="data-table">{caption}<thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )
    style = f"max-height:{max_height}px;overflow:auto;" if max_height else "overflow-x:auto;"
    display(HTML(f'<div style="{style}">{table}</div>'))
