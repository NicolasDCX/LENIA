#include "edit_history.hpp"

#include <QAction>
#include <QUndoCommand>
#include <QUndoStack>

#include <utility>

namespace workbench {

// One undo step: the parts of the setup it changed, before and after. The state is already
// "after" when the step is pushed, so its first redo (QUndoStack::push) does nothing.
class SetupCommand : public QUndoCommand {
public:
    SetupCommand(EditHistory* history, SetupParts before, SetupParts after, const QString& text, QString merge_key)
        : history_(history), before_(std::move(before)), after_(std::move(after)), merge_key_(std::move(merge_key)) {
        setText(text);
    }

    void undo() override {
        open_ = false; // an undone step never absorbs later edits
        history_->apply(before_, text(), true);
    }

    void redo() override {
        if (pushed_) {
            pushed_ = false;
            return;
        }
        history_->apply(after_, text(), false);
    }

    int id() const override { return merge_key_.isEmpty() ? -1 : kMergeId; }

    bool mergeWith(const QUndoCommand* other) override {
        const auto* next = static_cast<const SetupCommand*>(other); // the only command with kMergeId
        if (!open_ || next->merge_key_ != merge_key_) return false;
        // `next` starts where this step ends. A part only `next` changed was unchanged here
        // until then, so its "before" is next's.
        const auto fold = [](auto& before, auto& after, const auto& next_before, const auto& next_after) {
            if (!next_after) return;
            if (!before) before = next_before;
            after = next_after;
        };
        fold(before_.graph, after_.graph, next->before_.graph, next->after_.graph);
        fold(before_.display_graph, after_.display_graph, next->before_.display_graph, next->after_.display_graph);
        fold(before_.kernel, after_.kernel, next->before_.kernel, next->after_.kernel);
        fold(before_.growth, after_.growth, next->before_.growth, next->after_.growth);
        // Back where it started (a value stepped up, then down again): QUndoStack drops the step.
        setObsolete(before_ == after_);
        return true;
    }

    void seal() const { open_ = false; }

private:
    static constexpr int kMergeId = 0x4c45; // "LE"

    EditHistory* history_;
    SetupParts before_;
    SetupParts after_;
    QString merge_key_;
    bool pushed_ = true;
    mutable bool open_ = true;
};

EditHistory::EditHistory(Capture capture, Apply apply, QObject* parent)
    : QObject(parent), capture_(std::move(capture)), apply_(std::move(apply)), stack_(new QUndoStack(this)) {
    stack_->setUndoLimit(kUndoLimit);
    committed_ = capture_();
}

bool EditHistory::commit(const QString& text, const QString& merge_key) {
    SetupState now = capture_();
    SetupParts before;
    SetupParts after;
    if (!(now.graph == committed_.graph)) {
        before.graph = committed_.graph;
        after.graph = now.graph;
    }
    if (!(now.display_graph == committed_.display_graph)) {
        before.display_graph = committed_.display_graph;
        after.display_graph = now.display_graph;
    }
    if (!(now.kernel == committed_.kernel)) {
        before.kernel = committed_.kernel;
        after.kernel = now.kernel;
    }
    if (!(now.growth == committed_.growth)) {
        before.growth = committed_.growth;
        after.growth = now.growth;
    }
    if (after.empty()) return false;
    committed_ = std::move(now);
    stack_->push(new SetupCommand(this, std::move(before), std::move(after), text, merge_key));
    return true;
}

void EditHistory::seal() {
    if (stack_->index() == 0) return;
    if (const auto* top = dynamic_cast<const SetupCommand*>(stack_->command(stack_->index() - 1))) top->seal();
}

void EditHistory::reset() {
    stack_->clear();
    committed_ = capture_();
}

// Ctrl+Z in the middle of a gesture (keys work while a mouse button is held) first commits what
// the gesture changed so far, then undoes that. For Ctrl+Y the committed change is the newest
// step, which drops the redo steps as any new edit does, so there is nothing left to redo.
// Either also ends merging: the step now on top was made before, by another gesture.
void EditHistory::undo() {
    commit(tr("Unfinished edit"));
    stack_->undo();
    seal();
}

void EditHistory::redo() {
    commit(tr("Unfinished edit"));
    stack_->redo();
    seal();
}

bool EditHistory::has_uncommitted_changes() const { return !(capture_() == committed_); }

void EditHistory::apply(const SetupParts& parts, const QString& text, bool undo) {
    apply_(parts);
    committed_ = capture_();
    emit applied(text, undo);
}

QAction* EditHistory::create_undo_action(QObject* parent) {
    auto* action = new QAction(parent);
    const auto update = [this, action] {
        action->setEnabled(stack_->canUndo());
        action->setText(stack_->canUndo() ? tr("&Undo %1").arg(stack_->undoText()) : tr("&Undo"));
    };
    connect(stack_, &QUndoStack::canUndoChanged, action, update);
    connect(stack_, &QUndoStack::undoTextChanged, action, update);
    connect(action, &QAction::triggered, this, &EditHistory::undo);
    update();
    return action;
}

QAction* EditHistory::create_redo_action(QObject* parent) {
    auto* action = new QAction(parent);
    const auto update = [this, action] {
        action->setEnabled(stack_->canRedo());
        action->setText(stack_->canRedo() ? tr("&Redo %1").arg(stack_->redoText()) : tr("&Redo"));
    };
    connect(stack_, &QUndoStack::canRedoChanged, action, update);
    connect(stack_, &QUndoStack::redoTextChanged, action, update);
    connect(action, &QAction::triggered, this, &EditHistory::redo);
    update();
    return action;
}

} // namespace workbench
