// Undo / redo in the real workbench window (docs/undo.md), driven like a user:
//
// - mouse gestures sent to the real widgets: a node dragged across the node graph canvas, a link
//   dragged from a port to a port, a paint stroke on the kernel canvas, a node dragged on the
//   display graph canvas; each is one undo step however many mouse moves it has;
// - inspector and panel spin box steps (one step per field until it loses focus), a node added
//   from the toolbar path and a display graph example loaded;
// - one history for all of it: undo walks back through graph, kernel, display and growth edits
//   in order and restores each setup exactly, redo walks forward again; Ctrl+Z / Ctrl+Y also
//   reach the history from the Vulkan viewport and from a focused spin box;
// - the Edit menu's actions are enabled exactly when there is something to undo / redo;
// - undo never touches the world: the generation, the GPU density (bit for bit) and play /
//   pause are the same after it, paused or running;
// - a new edit drops the redo steps; a failed open keeps the history; opening an experiment or
//   a snapshot clears it;
// - Kernel K and Convolve nodes: previewing a Convolve node leaves the simulation bit-identical,
//   and their inspector edits and field links are single undo steps restored exactly.
//
// Needs a desktop session and a Vulkan device; exits with 77 (skipped) without them.

#include "main_window.hpp"

#include "lenia/display_graph.hpp"
#include "lenia/node_graph.hpp"
#include "lenia/pattern.hpp"
#include "lenia/project.hpp"

#include <QApplication>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QFile>
#include <QKeyEvent>
#include <QMessageBox>
#include <QMouseEvent>
#include <QTemporaryDir>
#include <QThread>
#include <QTimer>
#include <QUndoStack>
#include <QVulkanInstance>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr int kSkip = 77; // SKIP_RETURN_CODE in CMakeLists.txt

int failures = 0;

void expect(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok    " : "FAILED", what);
    if (!ok) ++failures;
}

bool same(const lenia::Grid& a, const lenia::Grid& b) {
    if (a.width != b.width || a.height != b.height || a.cells.size() != b.cells.size()) return false;
    for (std::size_t i = 0; i < a.cells.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(a.cells[i]) != std::bit_cast<std::uint32_t>(b.cells[i])) return false;
    }
    return true;
}

bool wait_until(const std::function<bool()>& done, int timeout_ms = 30000) {
    QElapsedTimer timer;
    timer.start();
    while (!done()) {
        if (timer.elapsed() > timeout_ms) return false;
        QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
        QThread::msleep(1);
    }
    return true;
}

void settle(int ms) {
    wait_until([] { return false; }, ms);
}

std::optional<workbench::WorldCapture> capture(workbench::LeniaView* view, const char* what) {
    auto result = std::make_shared<std::optional<workbench::WorldCapture>>();
    view->capture_world([result](const workbench::WorldCapture& c) { *result = c; });
    if (!wait_until([&] { return result->has_value(); }) || !(*result)->error.isEmpty()) {
        std::printf("FAILED: capture (%s)\n", what);
        ++failures;
        return std::nullopt;
    }
    return *result;
}

// A left (or right) button drag as the window system delivers it: press, `moves` moves with
// the button held, release.
void drag(QWidget* w, QPointF from, QPointF to, int moves = 10, Qt::MouseButton button = Qt::LeftButton) {
    QMouseEvent press(QEvent::MouseButtonPress, from, w->mapToGlobal(from), button, button, Qt::NoModifier);
    QCoreApplication::sendEvent(w, &press);
    for (int i = 1; i <= moves; ++i) {
        const QPointF p = from + (to - from) * (static_cast<double>(i) / moves);
        QMouseEvent move(QEvent::MouseMove, p, w->mapToGlobal(p), Qt::NoButton, button, Qt::NoModifier);
        QCoreApplication::sendEvent(w, &move);
    }
    QMouseEvent release(QEvent::MouseButtonRelease, to, w->mapToGlobal(to), button, Qt::NoButton, Qt::NoModifier);
    QCoreApplication::sendEvent(w, &release);
}

int first_node(const lenia::Graph& g, lenia::NodeKind kind) {
    for (const lenia::Node& n : g.nodes()) {
        if (n.kind == kind) return n.id;
    }
    return -1;
}

bool write_bytes(const QString& path, const QByteArray& bytes) {
    QFile f(path);
    return f.open(QIODevice::WriteOnly) && f.write(bytes) == bytes.size();
}

} // namespace

int main(int argc, char** argv) {
    using namespace workbench;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    QApplication app(argc, argv);

    QVulkanInstance instance;
    if (!instance.create()) {
        std::printf("SKIPPED: no Vulkan instance (VkResult %d)\n", static_cast<int>(instance.errorCode()));
        return kSkip;
    }

    // Any message box the workbench opens is a failure, closed so the test goes on.
    QTimer dialog_watch;
    QObject::connect(&dialog_watch, &QTimer::timeout, [] {
        if (QWidget* modal = QApplication::activeModalWidget()) {
            const auto* box = qobject_cast<QMessageBox*>(modal);
            std::printf("FAILED: the workbench opened a dialog: %s\n", qPrintable(box ? box->text() : modal->windowTitle()));
            ++failures;
            modal->close();
        }
    });
    dialog_watch.start(100);

    lenia::Pattern orbium;
    try {
        orbium = lenia::load_pattern_file(std::string(LENIA_PATTERN_DIR) + "/orbium.lenia");
    } catch (const std::exception& e) {
        std::printf("FAILED: %s\n", e.what());
        return 1;
    }

    MainWindow window(&instance);
    LeniaView* view = window.view();
    if (view->availablePhysicalDevices().isEmpty()) {
        std::printf("SKIPPED: no Vulkan physical device\n");
        return kSkip;
    }
    bool device_ready = false;
    QObject::connect(view, &LeniaView::device_ready, [&](const QString&) { device_ready = true; });
    window.show();
    if (!wait_until([&] { return device_ready && view->isExposed() && view->isValid(); })) {
        std::printf("SKIPPED: the Vulkan viewport did not come up (no desktop session or device?)\n");
        return kSkip;
    }

    QTemporaryDir dir;
    if (!dir.isValid()) {
        std::printf("FAILED: no temporary directory\n");
        return 1;
    }

    EditHistory* history = window.history();
    QUndoStack* stack = history->stack();
    QAction* undo = window.undo_action();
    QAction* redo = window.redo_action();
    const auto menu_state = [&](bool can_undo, bool can_redo) {
        return undo->isEnabled() == can_undo && redo->isEnabled() == can_redo && stack->canUndo() == can_undo &&
               stack->canRedo() == can_redo;
    };
    expect(undo->shortcuts().contains(QKeySequence(Qt::CTRL | Qt::Key_Z)), "Undo is Ctrl+Z");
    expect(redo->shortcuts().contains(QKeySequence(Qt::CTRL | Qt::Key_Y)) &&
               redo->shortcuts().contains(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_Z)),
           "Redo is Ctrl+Y and Ctrl+Shift+Z");

    // --- A known experiment, paused at generation 5. ----------------------------------------
    lenia::Project known;
    known.kernel = orbium.kernel;
    known.growth = orbium.growth;
    known.world_size = 128;
    known.seed = lenia::SeedKind::Pattern;
    known.pattern = orbium;
    known.graph = lenia::make_preset(lenia::GraphPreset::BreathingKernel);
    const QString experiment = dir.filePath(QStringLiteral("known.leniaexp"));
    write_bytes(experiment, QByteArray::fromStdString(lenia::write_project(known)));
    expect(window.open_experiment_file(experiment).isEmpty(), "open the known experiment");
    expect(menu_state(false, false), "after opening: Undo and Redo disabled");

    view->step(5);
    expect(wait_until([&] { return view->generation() == 5; }), "stepped to generation 5");
    const std::optional<WorldCapture> at5 = capture(view, "generation 5");
    if (!at5) return 1;

    // states[k]: the setup after k steps.
    std::vector<SetupState> states{window.capture_setup()};
    std::vector<QString> texts;
    const auto step_made = [&](const char* what) {
        const bool one = stack->count() == static_cast<int>(states.size()) && stack->index() == stack->count();
        expect(one, what);
        states.push_back(window.capture_setup());
        texts.push_back(stack->undoText());
        std::printf("        step %zu: %s\n", texts.size(), qPrintable(texts.back()));
    };

    // --- Gestures and edits, each exactly one step. ------------------------------------------
    NodeCanvas* canvas = window.node_editor()->canvas();
    const int osc = first_node(window.graph(), lenia::NodeKind::Oscillator);
    expect(osc >= 0, "setup: the graph has an Oscillator");
    const lenia::Node osc_before = *window.graph().node(osc);
    const QPointF title = canvas->title_widget_pos(osc);
    drag(canvas, title, title + QPointF(60, 30), 12);
    expect(canvas->selected_node() == osc, "the press picked the Oscillator");
    const lenia::Node* osc_after = window.graph().node(osc);
    expect(osc_after->x != osc_before.x && osc_after->y != osc_before.y, "a mouse drag moves the node");
    step_made("a node drag of 12 mouse moves is one undo step");
    expect(texts.back().contains(QStringLiteral("Move node")), "the step is named Move node");
    expect(menu_state(true, false), "Undo enabled, Redo disabled after an edit");
    expect(undo->text().contains(QStringLiteral("Move node")), "the Undo action names the step");

    canvas->add_node(lenia::NodeKind::Constant);
    const int constant = window.graph().nodes().back().id;
    step_made("adding a node is one step");

    drag(canvas, canvas->port_widget_pos(constant, 0, true),
         canvas->port_widget_pos(osc, lenia::oscillator_node::AmplitudeIn, false), 15);
    const lenia::Link* link = window.graph().link_into(osc, lenia::oscillator_node::AmplitudeIn);
    expect(link && link->from_node == constant, "dragging from an output port to an input port links them");
    step_made("a link drag is one step");

    canvas->select_node(osc);
    auto* period = qobject_cast<QDoubleSpinBox*>(window.node_editor()->inspector_field(lenia::oscillator_node::Period));
    expect(period != nullptr, "the inspector shows the Oscillator's Period");
    if (!period) return 1;
    for (int i = 0; i < 3; ++i) period->stepBy(1);
    step_made("three inspector spin box steps are one step");
    emit period->editingFinished(); // focus leaves the field
    period->stepBy(1);
    step_made("after the field loses focus, the next step is a new one");

    KernelCanvas* kcanvas = window.kernel_canvas();
    const lenia::Grid paint_before = window.kernel_model()->paint_layer();
    const QPointF kc = QRectF(kcanvas->rect()).center();
    drag(kcanvas, kc - QPointF(30, 10), kc + QPointF(30, 20), 20);
    expect(!(window.kernel_model()->paint_layer() == paint_before), "a mouse stroke paints the kernel");
    step_made("a kernel paint stroke of 20 mouse moves is one step");

    window.load_display_preset(lenia::DisplayPreset::GrowthOverlay);
    step_made("loading a display example is one step");

    if (auto* dock = window.display_editor()->parentWidget()) dock->raise(); // its tab, like a user would
    settle(100);
    NodeCanvas* dcanvas = window.display_editor()->canvas();
    const int fields = first_node(window.display_graph(), lenia::NodeKind::Fields);
    const QPointF ftitle = dcanvas->title_widget_pos(fields);
    drag(dcanvas, ftitle, ftitle + QPointF(-40, 50), 8);
    step_made("a node drag on the display graph is one step");

    QDoubleSpinBox* mu = nullptr;
    for (QDoubleSpinBox* s : window.findChildren<QDoubleSpinBox*>()) {
        if (window.node_editor()->isAncestorOf(s) || window.display_editor()->isAncestorOf(s)) continue;
        if (s->minimum() == 0.0 && s->maximum() == 2.0 && s->decimals() == 4) mu = s; // the growth μ
    }
    expect(mu != nullptr, "setup: found the growth μ spin box");
    if (!mu) return 1;
    const float mu_before = window.capture_setup().growth.mu;
    mu->stepBy(1);
    mu->stepBy(1);
    expect(window.capture_setup().growth.mu != mu_before, "the μ spin box changes the growth");
    step_made("two μ steps are one step");

    // A focused spin box does not take Ctrl+Z for its own text undo.
    {
        QKeyEvent override_event(QEvent::ShortcutOverride, Qt::Key_Z, Qt::ControlModifier);
        QCoreApplication::sendEvent(mu, &override_event);
        expect(!override_event.isAccepted(), "a spin box leaves Ctrl+Z to the Edit menu");
    }

    expect(view->paused() && view->generation() == 5, "editing left the world paused at generation 5");
    if (const auto c = capture(view, "after the edits")) {
        expect(c->generation == 5 && same(c->density, at5->density), "editing left the density bit-identical");
    }

    // --- Undo everything, in order, then redo it. --------------------------------------------
    const int steps = static_cast<int>(texts.size());
    bool exact = true;
    bool world_kept = true;
    for (int k = steps; k > 0; --k) {
        if (k == steps - 1) {
            // Ctrl+Z pressed in the Vulkan viewport (a separate QWindow) reaches the history too.
            QKeyEvent key(QEvent::KeyPress, Qt::Key_Z, Qt::ControlModifier);
            QCoreApplication::sendEvent(view, &key);
        } else {
            undo->trigger();
        }
        exact = exact && window.capture_setup() == states[static_cast<std::size_t>(k - 1)];
        world_kept = world_kept && view->paused() && view->generation() == 5;
        if (k > 1 && !undo->isEnabled()) exact = false;
    }
    expect(exact, "each undo, from the Edit menu or Ctrl+Z in the viewport, restores the setup before its step");
    expect(world_kept, "every undo kept the world paused at generation 5");
    expect(menu_state(false, true), "everything undone: Undo disabled, Redo enabled");
    expect(window.graph().node(osc)->x == osc_before.x && window.kernel_model()->paint_layer() == paint_before &&
               window.capture_setup().growth.mu == mu_before,
           "node position, kernel paint and μ are back to their originals");
    expect(std::fabs(mu->value() - mu_before) < 1e-4, "the μ spin box shows the restored value");
    if (const auto c = capture(view, "after undoing everything")) {
        expect(c->generation == 5 && same(c->density, at5->density), "undo left the density bit-identical");
    }

    exact = true;
    for (int k = 1; k <= steps; ++k) {
        if (k == 2) {
            QKeyEvent key(QEvent::KeyPress, Qt::Key_Y, Qt::ControlModifier);
            QCoreApplication::sendEvent(view, &key);
        } else {
            redo->trigger();
        }
        exact = exact && window.capture_setup() == states[static_cast<std::size_t>(k)];
    }
    expect(exact, "each redo, from the Edit menu or Ctrl+Y in the viewport, restores the setup after its step");
    expect(menu_state(true, false), "everything redone: Undo enabled, Redo disabled");

    // --- Running: undo keeps playing and does not restart the world. -------------------------
    window.set_playing(true);
    expect(wait_until([&] { return view->generation() >= 20; }), "playback runs");
    const std::uint64_t running_at = view->generation();
    undo->trigger();
    expect(window.playing() && !view->paused(), "undo while running keeps it running");
    expect(view->generation() >= running_at, "undo while running does not reset the generation");
    expect(wait_until([&] { return view->generation() >= running_at + 10; }), "and the world keeps stepping");
    redo->trigger();
    window.set_playing(false);
    settle(200);
    const std::uint64_t paused_at = view->generation();
    const std::optional<WorldCapture> before_undo = capture(view, "paused, before undo");
    undo->trigger();
    redo->trigger();
    undo->trigger();
    settle(200);
    expect(!window.playing() && view->paused() && view->generation() == paused_at, "undo while paused stays paused");
    if (const auto c = capture(view, "paused, after undo / redo / undo"); c && before_undo) {
        expect(c->generation == paused_at && same(c->density, before_undo->density),
               "undo / redo while paused leave the density bit-identical");
    }

    // --- A new edit drops the redo steps. ----------------------------------------------------
    expect(redo->isEnabled(), "the undone step can be redone");
    const QPointF again = canvas->title_widget_pos(osc);
    drag(canvas, again, again + QPointF(-25, 0), 5);
    expect(menu_state(true, false), "a new edit after undo disables Redo");

    // --- Opening: a failed open keeps the history, a successful one clears it. -------------
    const QString broken = dir.filePath(QStringLiteral("broken.leniaexp"));
    write_bytes(broken, "not an experiment");
    expect(!window.open_experiment_file(broken).isEmpty(), "a broken experiment is refused");
    expect(menu_state(true, false), "the refused open kept the history");

    expect(window.open_experiment_file(experiment).isEmpty(), "reopen the experiment");
    expect(menu_state(false, false) && stack->count() == 0, "opening an experiment clears the history");

    drag(kcanvas, kc, kc + QPointF(15, 15), 6);
    expect(menu_state(true, false), "an edit in the reopened experiment can be undone");
    const QString snapshot = dir.filePath(QStringLiteral("undo.leniasnap"));
    std::optional<QString> saved;
    window.save_snapshot_file(snapshot, [&](const QString& error) { saved = error; });
    expect(wait_until([&] { return saved.has_value(); }) && saved->isEmpty(), "snapshot saved");
    drag(kcanvas, kc, kc - QPointF(15, 15), 6);
    expect(stack->count() == 2, "two strokes, two steps");
    expect(window.open_snapshot_file(snapshot).isEmpty(), "open the snapshot");
    expect(menu_state(false, false) && stack->count() == 0, "opening a snapshot clears the history");
    history->undo();
    expect(menu_state(false, false), "Ctrl+Z after opening does nothing");

    // --- Kernel K and Convolve nodes (node_graph.md, Several kernels). ----------------------
    if (auto* dock = window.node_editor()->parentWidget()) dock->raise();
    settle(100);
    const QString two_kernels = dir.filePath(QStringLiteral("two_kernels.leniaexp"));
    write_bytes(two_kernels, QByteArray::fromStdString(lenia::write_project(lenia::make_short_long_project())));
    // Reference: three generations with nothing selected.
    expect(window.open_experiment_file(two_kernels).isEmpty(), "open the two-kernel experiment");
    canvas->select_node(-1);
    view->step(3);
    expect(wait_until([&] { return view->generation() == 3; }), "two kernels: stepped to generation 3");
    const std::optional<WorldCapture> plain = capture(view, "two kernels, nothing selected");
    // The same three generations while a Convolve node is previewed in the Neighborhood field.
    expect(window.open_experiment_file(two_kernels).isEmpty(), "reopen the two-kernel experiment");
    const int conv = first_node(window.graph(), lenia::NodeKind::Convolve);
    const int kk = first_node(window.graph(), lenia::NodeKind::ConvKernel);
    expect(conv >= 0 && kk >= 0, "setup: the graph has Kernel K and Convolve nodes");
    if (conv < 0 || kk < 0) return 1;
    canvas->select_node(conv);
    view->step(3);
    expect(wait_until([&] { return view->generation() == 3; }), "previewing: stepped to generation 3");
    const std::optional<WorldCapture> previewed = capture(view, "two kernels, Convolve selected");
    if (plain && previewed) {
        expect(same(plain->density, previewed->density) && same(plain->nutrient, previewed->nutrient),
               "previewing a Convolve node leaves the simulation bit-identical");
        expect(!same(plain->density, lenia::Grid(plain->density.width, plain->density.height, 0.0f)),
               "the two-kernel rule keeps a pattern alive");
    }

    std::vector<SetupState> kstates{window.capture_setup()};
    const auto kstep = [&](const char* what) {
        expect(stack->count() == static_cast<int>(kstates.size()) && stack->index() == stack->count(), what);
        kstates.push_back(window.capture_setup());
    };
    canvas->select_node(kk);
    auto* radius =
        qobject_cast<QDoubleSpinBox*>(window.node_editor()->inspector_field(lenia::conv_kernel_node::Radius));
    expect(radius != nullptr, "the inspector shows the Kernel K's Radius");
    if (!radius) return 1;
    const float radius_before = window.graph().node(kk)->params[lenia::conv_kernel_node::Radius];
    radius->stepBy(1);
    radius->stepBy(1);
    expect(window.graph().node(kk)->params[lenia::conv_kernel_node::Radius] == radius_before + 2.0f,
           "Radius steps change the Kernel K");
    kstep("two Kernel K Radius steps are one undo step");
    emit radius->editingFinished();
    auto* shift = qobject_cast<QDoubleSpinBox*>(window.node_editor()->inspector_field(lenia::conv_kernel_node::ShiftX));
    expect(shift != nullptr, "the inspector shows the Kernel K's Shift X");
    if (!shift) return 1;
    shift->stepBy(3);
    expect(window.graph().node(kk)->params[lenia::conv_kernel_node::ShiftX] == 1.5f, "Shift X steps by 0.5 cells");
    kstep("Shift X steps are one undo step");

    canvas->add_node(lenia::NodeKind::Nutrient);
    const int nutrient = window.graph().nodes().back().id;
    kstep("adding a Nutrient N node is one step");
    drag(canvas, canvas->port_widget_pos(nutrient, lenia::field_node::FieldOut, true),
         canvas->port_widget_pos(conv, lenia::convolve_node::FieldIn, false), 10);
    const lenia::Link* field_link = window.graph().link_into(conv, lenia::convolve_node::FieldIn);
    expect(field_link && field_link->from_node == nutrient, "a Nutrient N's Field output links into Convolve's Field");
    kstep("relinking a Convolve's field is one step");

    bool kexact = true;
    for (int k = static_cast<int>(kstates.size()) - 1; k > 0; --k) {
        undo->trigger();
        kexact = kexact && window.capture_setup() == kstates[static_cast<std::size_t>(k - 1)];
    }
    expect(kexact, "undo restores each Kernel K / Convolve edit exactly");
    expect(window.graph().node(kk)->params[lenia::conv_kernel_node::Radius] == radius_before,
           "the Kernel K's radius is back");
    for (std::size_t k = 1; k < kstates.size(); ++k) {
        redo->trigger();
        kexact = kexact && window.capture_setup() == kstates[k];
    }
    expect(kexact, "redo restores them again");
    expect(view->paused() && view->generation() == 3, "Kernel K / Convolve undo kept the world paused at generation 3");

    window.close();
    settle(100);
    if (failures == 0) {
        std::printf("All workbench undo tests passed.\n");
        return 0;
    }
    std::printf("%d failure(s).\n", failures);
    return 1;
}
