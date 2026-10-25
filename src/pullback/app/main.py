"""The desktop app: a window over `pullback.session.Session`.

Every panel is rendered by `panels` (plain data, no Qt); this module only
puts those values into widgets, and does no arithmetic.

    python3 -m pullback.app                 # the library
    python3 -m pullback.app sessions/theta.json

Verification here is headless (`QT_QPA_PLATFORM=offscreen`; see
`tests/test_app_headless.py`): the window is constructed, sessions opened,
and every panel's contents compared with the engine's values.  There is no
display on the build machine, so the first real GUI check is the author
opening it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..session import Session
from . import panels

ROOT = Path(__file__).resolve().parents[3]


def _mono_font():
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)


class MainWindow(QMainWindow):
    def __init__(self, root=None):
        super().__init__()
        self._root = Path(root) if root is not None else ROOT
        self._paths = {}
        self.session = None
        self.setWindowTitle("Pullback — graph virtual endomorphisms")
        self.resize(1150, 720)
        self._build()
        self._populate_library()

    # ------------------------------------------------------------- widgets
    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        left = QVBoxLayout()
        left.addWidget(QLabel("Session library"))
        self.session_list = QListWidget()
        self.session_list.currentTextChanged.connect(self._on_pick)
        left.addWidget(self.session_list, 1)
        self.open_button = QPushButton("Open session…")
        self.open_button.clicked.connect(self._on_open)
        left.addWidget(self.open_button)
        layout.addLayout(left, 0)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.summary_view = QPlainTextEdit()
        self.summary_view.setReadOnly(True)
        self.summary_view.setFont(_mono_font())
        self.tabs.addTab(self.summary_view, "Summary")

        portrait_tab = QWidget()
        pv = QVBoxLayout(portrait_tab)
        self.portrait_table = self._table()
        pv.addWidget(self.portrait_table, 1)
        self.portrait_note = QLabel("")
        self.portrait_note.setWordWrap(True)
        pv.addWidget(self.portrait_note)
        self.tabs.addTab(portrait_tab, "Portrait")

        self.automaton_view = QPlainTextEdit()
        self.automaton_view.setReadOnly(True)
        self.automaton_view.setFont(_mono_font())
        self.tabs.addTab(self.automaton_view, "Automaton")

        energy_tab = QWidget()
        ev = QVBoxLayout(energy_tab)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("p"))
        self.p_combo = QComboBox()
        self.p_combo.addItems(["2", "3", "4"])
        controls.addWidget(self.p_combo)
        controls.addWidget(QLabel("n up to"))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 4)
        self.n_spin.setValue(1)
        controls.addWidget(self.n_spin)
        self.run_button = QPushButton("Compute")
        self.run_button.clicked.connect(self._run_energies)
        controls.addWidget(self.run_button)
        controls.addStretch(1)
        ev.addLayout(controls)
        self.energy_table = self._table()
        ev.addWidget(self.energy_table, 1)
        self.energy_note = QLabel("press Compute")
        self.energy_note.setWordWrap(True)
        ev.addWidget(self.energy_note)
        self.tabs.addTab(energy_tab, "Energies")

    def _table(self):
        table = QTableWidget()
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        return table

    @staticmethod
    def _fill(table, headers, rows):
        table.clear()
        table.setColumnCount(len(headers))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels(list(headers))
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(str(cell)))
        table.resizeColumnsToContents()

    def _populate_library(self):
        sessions = self._root / "sessions"
        paths = sorted(sessions.glob("*.json")) if sessions.is_dir() else []
        self.session_list.blockSignals(True)
        for path in paths:
            self._paths[path.stem] = path
            self.session_list.addItem(path.stem)
        self.session_list.blockSignals(False)

    # -------------------------------------------------------------- actions
    def _on_pick(self, name):
        path = self._paths.get(name)
        if path is not None:
            self.open_session(path)

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open session", str(self._root / "sessions"), "Sessions (*.json)")
        if path:
            self.open_session(path)

    def open_session(self, path):
        """Load and render a session.  Returns False (and shows the reason in
        the Summary panel) if the file will not load -- a bad dataset is
        reported, never a crash."""
        try:
            session = Session.load(path)
        except Exception as exc:                            # noqa: BLE001
            self.session = None
            self.summary_view.setPlainText(
                f"error opening {path}\n\n{type(exc).__name__}: {exc}")
            self._fill(self.portrait_table, (), ())
            self.portrait_note.setText("")
            self.automaton_view.setPlainText("")
            self._fill(self.energy_table, (), ())
            self.energy_note.setText("")
            self.statusBar().showMessage("open failed")
            return False
        self.session = session
        self._render()
        self.statusBar().showMessage(f"opened {path}")
        return True

    def _render(self):
        self.summary_view.setPlainText(self._safe(panels.summary))
        try:
            headers, rows, note = panels.portrait(self.session)
        except Exception as exc:                            # noqa: BLE001
            headers, rows, note = (), [], f"refused: {exc}"
        self._fill(self.portrait_table, headers, rows)
        self.portrait_note.setText(note)
        self.automaton_view.setPlainText(self._safe(panels.automaton))
        self._fill(self.energy_table, (), ())
        self.energy_note.setText("press Compute")

    def _safe(self, fn):
        try:
            return fn(self.session)
        except Exception as exc:                            # noqa: BLE001
            return f"refused: {type(exc).__name__}: {exc}"

    def _run_energies(self):
        if self.session is None:
            return
        p = self.p_combo.currentText()
        n = self.n_spin.value()
        started = time.time()
        try:
            headers, rows, note = panels.energies(self.session, p=p, n=n)
        except Exception as exc:                            # noqa: BLE001
            headers, rows, note = (), [], f"refused: {type(exc).__name__}: {exc}"
        self._fill(self.energy_table, headers, rows)
        self.energy_note.setText(note)
        elapsed = time.time() - started
        self.statusBar().showMessage(
            f"energies p={p}, n={n} in {elapsed:.2f} s"
            + ("  (slow: the bound re-tunes the metric each n)" if elapsed > 2 else ""))

    # ---------------------------------------------------------- test hooks
    def summary_text(self):
        return self.summary_view.toPlainText()

    def automaton_text(self):
        return self.automaton_view.toPlainText()

    def portrait_rows(self):
        t = self.portrait_table
        return [[t.item(i, j).text() if t.item(i, j) else ""
                 for j in range(t.columnCount())] for i in range(t.rowCount())]

    def energy_rows(self):
        t = self.energy_table
        return [[t.item(i, j).text() if t.item(i, j) else ""
                 for j in range(t.columnCount())] for i in range(t.rowCount())]


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="pullback-app",
                                 description="The pullback applet (desktop).")
    ap.add_argument("session", nargs="?", default=None,
                    help="a session JSON to open at start")
    args = ap.parse_args(argv)
    app = QApplication(sys.argv[:1])
    window = MainWindow()
    if args.session:
        window.open_session(args.session)
    window.show()
    return app.exec()
