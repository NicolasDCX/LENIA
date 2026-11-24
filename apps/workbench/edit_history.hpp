#pragma once

#include "lenia/kernel_shape.hpp"
#include "lenia/node_graph.hpp"
#include "lenia/step.hpp"

#include <QObject>
#include <QString>

#include <functional>
#include <optional>

class QAction;
class QUndoStack;

namespace workbench {

// The kernel as the Kernel panel edits it (KernelModel): shape, normalization and paint layer.
struct KernelState {
    lenia::KernelShape shape;
    lenia::KernelNormalization normalization = lenia::KernelNormalization::Sum;
    lenia::Grid paint; // (2R + 1)^2, zeros for no paint
    bool operator==(const KernelState&) const = default;
};

// Everything the undo history covers (docs/undo.md): the simulation and display graphs, the
// kernel and the growth. Not the world, its generation or play / pause: undoing an edit never
// touches the running simulation.
struct SetupState {
    lenia::Graph graph;
    lenia::Graph display_graph{lenia::GraphDomain::Display};
    KernelState kernel;
    lenia::GrowthParams growth;
    bool operator==(const SetupState&) const = default;
};

// Some parts of a SetupState: what one undo step changes, before or after.
struct SetupParts {
    std::optional<lenia::Graph> graph;
    std::optional<lenia::Graph> display_graph;
    std::optional<KernelState> kernel;
    std::optional<lenia::GrowthParams> growth;
    bool empty() const { return !graph && !display_graph && !kernel && !growth; }
    bool operator==(const SetupParts&) const = default;
};

// One unified undo history over a QUndoStack, built from checkpoints. Editors change the setup
// directly, as before, and call commit() when a gesture ends (a drag released, a menu command
// done, a spin box step): the difference between the last committed state and the current one
// becomes one undo step. Changes made during a gesture (a node drag's every move, a paint
// stroke's every dab) are therefore one step, whatever their number.
//
// commit() with a merge key folds the change into the previous step if that step has the same
// key and is still open (not undone, not sealed): consecutive spin box steps on one field are
// one step until the field loses focus. Merging back to the original state drops the step.
//
// Undo and redo apply the stored parts through `apply`, which must not commit, and end merging.
// Uncommitted changes (an edit in the middle of a gesture) are committed first, so they are
// never lost or folded into an unrelated step.
class EditHistory : public QObject {
    Q_OBJECT
public:
    using Capture = std::function<SetupState()>;
    using Apply = std::function<void(const SetupParts& parts)>;

    EditHistory(Capture capture, Apply apply, QObject* parent = nullptr);

    // At most this many steps are kept; the oldest are dropped.
    static constexpr int kUndoLimit = 200;

    // The steps themselves (count, index, canUndo, undoText). Undo and redo go through undo()
    // and redo() below, not the stack's own slots, so that uncommitted changes are kept.
    QUndoStack* stack() const { return stack_; }
    // Edit menu actions: trigger undo() / redo(), enabled only when there is a step to undo /
    // redo, their text naming it ("Undo Move node"). Shortcuts are the caller's.
    QAction* create_undo_action(QObject* parent);
    QAction* create_redo_action(QObject* parent);

    // Records the change since the last commit as one undo step named `text`; false (and no
    // step) if nothing changed. See the class comment for `merge_key`.
    bool commit(const QString& text, const QString& merge_key = {});
    // Ends merging into the current top step (a field lost focus, say).
    void seal();
    // Drops the whole history and takes the current state as the committed one: for opening
    // another experiment or snapshot, whose setup the old steps do not describe.
    void reset();

    void undo();
    void redo();

    // For tests and diagnostics: whether the setup differs from the last commit.
    bool has_uncommitted_changes() const;

signals:
    // After an undo or redo was applied; `text` names the step.
    void applied(const QString& text, bool undo);

private:
    friend class SetupCommand;
    // A step's undo or redo: applies `parts` and takes the result as the committed state.
    void apply(const SetupParts& parts, const QString& text, bool undo);

    Capture capture_;
    Apply apply_;
    QUndoStack* stack_;
    SetupState committed_;
};

} // namespace workbench
