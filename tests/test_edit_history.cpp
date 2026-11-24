// The workbench's undo history (apps/workbench/edit_history.hpp, docs/undo.md) on a plain
// SetupState, without a window or a GPU: checkpoints become steps, gestures and spin box runs
// merge into one step, undo / redo restore exactly the parts a step changed, and reset drops
// everything.

#include "edit_history.hpp"

#include <QCoreApplication>
#include <QUndoStack>

#include <cstdio>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok    " : "FAILED", what);
    if (!ok) ++failures;
}

} // namespace

int main(int argc, char** argv) {
    using namespace workbench;
    QCoreApplication app(argc, argv);

    // The "window": edits change it directly, as the editors change MainWindow's members.
    SetupState state;
    state.graph = lenia::make_preset(lenia::GraphPreset::Idle);
    state.kernel.paint = lenia::Grid(27, 27, 0.0f);
    int applies = 0;
    SetupParts last_applied;
    EditHistory history([&] { return state; },
                        [&](const SetupParts& p) {
                            ++applies;
                            last_applied = p;
                            if (p.graph) state.graph = *p.graph;
                            if (p.display_graph) state.display_graph = *p.display_graph;
                            if (p.kernel) state.kernel = *p.kernel;
                            if (p.growth) state.growth = *p.growth;
                        });
    QUndoStack* stack = history.stack();
    const SetupState initial = state;

    expect(!stack->canUndo() && !stack->canRedo(), "a new history has nothing to undo or redo");
    expect(!history.commit(QStringLiteral("Nothing")) && stack->count() == 0, "an unchanged setup makes no step");

    // One gesture, many changes: a node dragged through 20 positions is one step.
    const int node = state.graph.nodes().front().id;
    for (int i = 1; i <= 20; ++i) state.graph.move_node(node, 10.0f * i, 5.0f * i);
    expect(history.has_uncommitted_changes(), "a drag in progress is not committed yet");
    expect(history.commit(QStringLiteral("Move node")), "the drag's end commits it");
    expect(stack->count() == 1 && stack->undoText() == QStringLiteral("Move node"), "20 moves are one named step");
    const SetupState moved = state;

    // A step only holds what changed.
    stack->undo();
    expect(state == initial, "undo restores the graph exactly");
    expect(last_applied.graph && !last_applied.display_graph && !last_applied.kernel && !last_applied.growth,
           "undoing a graph step applies the graph only");
    expect(!history.has_uncommitted_changes(), "after undo the restored state is the committed one");
    expect(stack->canRedo() && !stack->canUndo(), "undo enables redo");
    stack->redo();
    expect(state == moved, "redo restores the moved graph exactly");

    // Several parts in one step (an edit touching graph and kernel) undo together.
    state.kernel.paint.at(13, 13) = 0.5f;
    state.display_graph.add_node(lenia::NodeKind::Color, 1.0f, 2.0f);
    history.commit(QStringLiteral("Two parts"));
    stack->undo();
    expect(state == moved, "a two-part step undoes both parts");
    expect(last_applied.kernel && last_applied.display_graph && !last_applied.graph && !last_applied.growth,
           "and applies exactly those two parts");
    stack->redo();

    // Spin box steps with one merge key: one step; seal() ends it.
    const SetupState before_spin = state;
    const int steps_before = stack->count();
    for (int i = 1; i <= 5; ++i) {
        state.growth.mu = 0.15f + 0.001f * static_cast<float>(i);
        history.commit(QStringLiteral("Growth mu"), QStringLiteral("mu 0"));
    }
    expect(stack->count() == steps_before + 1, "five steps on one field merge into one undo step");
    const SetupState after_spin = state;
    history.seal();
    state.growth.mu = 0.2f;
    history.commit(QStringLiteral("Growth mu"), QStringLiteral("mu 0"));
    expect(stack->count() == steps_before + 2, "after seal() (focus lost) the same field starts a new step");
    stack->undo();
    expect(state == after_spin, "undo of the second run restores the end of the first");
    stack->undo();
    expect(state == before_spin, "undo of the merged run restores the value before all five steps");

    // A different field's key does not merge; a merged part only touched later is kept right.
    stack->redo();
    stack->redo();
    const int count = stack->count();
    state.growth.sigma = 0.02f;
    history.commit(QStringLiteral("Growth sigma"), QStringLiteral("sigma 0"));
    state.growth.dt = 0.2f;
    history.commit(QStringLiteral("Growth dt"), QStringLiteral("dt 0"));
    expect(stack->count() == count + 2, "different fields are different steps");

    // Stepping back to the start removes the step altogether.
    const int count2 = stack->count();
    const float dt = state.growth.dt;
    state.growth.dt = dt + 0.1f;
    history.commit(QStringLiteral("Growth dt"), QStringLiteral("dt 1"));
    state.growth.dt = dt;
    history.commit(QStringLiteral("Growth dt"), QStringLiteral("dt 1"));
    expect(stack->count() == count2, "a run that ends where it started leaves no step");

    // Undo ends merging: a later edit with the same key is a new step, not folded into an
    // older one that undo uncovered.
    state.growth.mu = 0.3f;
    history.commit(QStringLiteral("Growth mu"), QStringLiteral("mu 2"));
    state.graph.move_node(node, 1.0f, 1.0f);
    history.commit(QStringLiteral("Move node"));
    const int count3 = stack->count();
    history.undo(); // the move, as the Edit menu does; "mu 2" is on top again
    state.growth.mu = 0.31f;
    history.commit(QStringLiteral("Growth mu"), QStringLiteral("mu 2"));
    expect(stack->count() == count3 && stack->index() == count3 && !stack->canRedo(),
           "a new edit after undo drops the redo step and is a step of its own");
    history.undo();
    expect(state.growth.mu == 0.3f, "it did not merge into the step undo uncovered");

    // Ctrl+Z in the middle of a gesture: what the gesture changed so far is undone, not lost
    // into the next step.
    const SetupState before_gesture = state;
    state.graph.move_node(node, 99.0f, 99.0f); // not committed
    history.undo();
    expect(state == before_gesture, "undo in mid-gesture reverts the unfinished change");
    expect(!history.has_uncommitted_changes(), "and leaves nothing uncommitted");
    expect(stack->canRedo(), "the unfinished change can be redone");
    history.redo();
    expect(state.graph.node(node)->x == 99.0f, "redo brings it back");

    // The history is bounded.
    for (int i = 0; i < EditHistory::kUndoLimit + 20; ++i) {
        state.graph.move_node(node, static_cast<float>(i), 0.0f);
        history.commit(QStringLiteral("Move node"));
    }
    expect(stack->count() == EditHistory::kUndoLimit, "at most kUndoLimit steps are kept");

    // reset(): another experiment. Nothing to undo, and its state is the committed one.
    state.graph = lenia::make_preset(lenia::GraphPreset::GameOfLife);
    history.reset();
    expect(stack->count() == 0 && !stack->canUndo() && !stack->canRedo(), "reset drops every step");
    expect(!history.has_uncommitted_changes(), "reset takes the current setup as committed");
    const int applies_before = applies;
    history.undo();
    expect(applies == applies_before && state.graph == lenia::make_preset(lenia::GraphPreset::GameOfLife),
           "undo after reset changes nothing");

    if (failures == 0) {
        std::printf("All edit history tests passed.\n");
        return 0;
    }
    std::printf("%d failure(s).\n", failures);
    return 1;
}
