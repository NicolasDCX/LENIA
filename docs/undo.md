# Undo and redo

`lenia_workbench` has one undo history for everything that defines the experiment's rule and
look while it runs: the **node graph**, the **display graph**, the **kernel** and the **growth**.
*Edit → Undo* (**Ctrl+Z**) and *Edit → Redo* (**Ctrl+Y** or **Ctrl+Shift+Z**) walk through it in
the order the edits were made, whichever panel they were made in. The menu names the step
("Undo Move node (node graph)") and each action is enabled only when there is a step to undo or
redo. The shortcuts work from every panel, floating panels, the Vulkan viewport and spin boxes
(which would otherwise use Ctrl+Z for their own text). The status bar says what was undone.

## Undo never touches the running world

Undoing or redoing an edit restores the graphs, kernel or growth exactly as they were (bit for
bit: whole values are stored, not differences), and nothing else. The world keeps its
**density**, its **generation** and its **play / pause** state: a running simulation continues
from where it is with the restored setup from its next generation on, exactly as if the edit
had been reverted by hand; a paused one stays paused with the same cells. The node graph keeps
its view and selection (when the selected node still exists).

## One gesture, one step

| Edit | Step |
|------|------|
| Drag a node (node or display graph) | one *Move node*, however long the drag |
| Drag from a port to a port | one *Link*; picking up a linked input and dropping it elsewhere is one *Link*, dropping it on empty space one *Unlink*, back where it was no step |
| Add a node (toolbar or right-click), delete it (Del, inspector, right-click) | one *Add …* / *Delete …* (with its links) |
| Unlink (Del on a link, right-click), *Unlink all inputs* | one step each |
| *Wire the built-in Lenia rule* / *Wire the panel display as nodes* | one step |
| Load a Lenia rule example or a display example | one *Load example* |
| Inspector choice (combo box) | one *Set …* per choice |
| Inspector number: arrows, wheel, typing + Enter (also a Kernel K's radius, ring weights and shifts) | one *Edit …* per field until the field loses focus |
| Color ramp: drag a stop / add / recolor / remove / palette / reverse | one step each (a drag once, on release) |
| Kernel paint stroke (left, right or ctrl) | one *Paint kernel* / *Erase kernel paint* per stroke, however many dabs |
| Ring profile drag (sweeping several rings) | one *Set ring weights* |
| Kernel radius, rings (spin boxes) | one step per field until it loses focus |
| Kernel core, normalization, *Clear paint* | one step each |
| Growth μ, σ, dt (spin boxes) | one step per field until it loses focus |
| Growth function | one step |

A number field stepped back to where it started leaves no step. Steps are recorded when a
gesture ends (mouse button released, menu command done, a field's value changes): pressing
Ctrl+Z in the middle of a drag undoes what the drag did so far. A new edit after an undo drops
the steps that could have been redone, as usual. At most 200 steps are kept.

## What is not in the history

- The **world**: painting cells in the viewport, Reset, Impulse, seed and world size choices,
  stepping and playing. These are the simulation itself, which undo leaves alone.
- The Simulation panel's **View** settings (field, palette, low, high), playback settings,
  convolution method and brushes: viewing and tool settings, not the experiment's rule.
- Node graph view changes (pan, zoom, selection), including which Convolve node the
  Neighborhood field previews.
- File commands.

## When the history is cleared

The steps describe one setup; they are dropped when another one replaces it:

- *File → Open experiment…* and *Open snapshot…* (only when the file opens: a file that is
  refused leaves both the experiment and its history untouched);
- *Restore Orbium*, the *Game of Life* preset and *Load pattern file…*, which load a new creature
  and reseed the world.

Saving does not change the history.

## Implementation

`apps/workbench/edit_history.hpp`: `EditHistory` keeps the last committed `SetupState` (both
graphs, kernel shape, normalization and paint layer, growth) next to a `QUndoStack`. Editors
change the setup directly and signal the end of each gesture (`NodeCanvas::edited`,
`NodeEditor::edited`, `KernelCanvas::edited`, `RingProfileWidget::edited`, panel handlers in
`MainWindow`); `EditHistory::commit` turns the difference into one step holding only the parts
that changed, before and after. Spin box steps carry a merge key (node, parameter or field, and
an editing session bumped when the field loses focus) so `QUndoCommand::mergeWith` folds them
into one step; undo and redo end merging. `MainWindow::apply_setup` applies a step's parts the
way hand edits are applied (`LeniaView::inputs_changed`, `display_changed`).

## Tests

- `tests/test_edit_history.cpp` (CTest `edit_history`; Qt, no window or GPU): no step without a
  change; a 20-move drag is one step; undo / redo restore exactly and apply only the parts a step
  changed; a two-part step undoes both; spin box merging, sealing, distinct fields, a run back to
  the start leaving no step; no merge into a step uncovered by undo; redo dropped by a new edit;
  Ctrl+Z in mid-gesture; the 200-step limit; reset.
- `tests/test_workbench_undo.cpp` (CTest `workbench_undo`, label `gui`; needs a desktop session
  and a Vulkan device, else skipped): the real window driven with mouse events: a node drag
  (12 moves), a node added, a port-to-port link drag, inspector spin box steps (merged, then split
  by focus loss), a kernel paint stroke (20 moves), a display example, a display graph node drag
  and growth μ steps; each exactly one step, named in the Edit menu. Undoing all of them (Edit
  menu and Ctrl+Z in the viewport) restores every intermediate setup exactly; redoing them (Edit
  menu and Ctrl+Y) restores every later one. Undo and redo keep the world paused at its
  generation with bit-identical GPU density, and keep a running world running without resetting
  it. Also: Undo / Redo enabled state throughout, a focused spin box leaving Ctrl+Z to the menu,
  a new edit dropping redo, a refused open keeping the history, and opening an experiment or a
  snapshot clearing it.

Not covered by automated tests: the color ramp editor's gestures, the ring profile drag, the
right-click menus, Delete key and inspector Delete button, the kernel panel's spin boxes and
combo boxes, and Ctrl+Z / Ctrl+Y delivered by the operating system as real key presses to a
docked widget (the tests trigger the actions, send key events to the viewport, and check the
spin box shortcut override). They go through the same `edited` signals and `EditHistory::commit`
as the tested paths.
