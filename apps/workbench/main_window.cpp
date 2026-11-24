#include "main_window.hpp"

#include "palettes.hpp"

#include <QAbstractSpinBox>
#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDockWidget>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPushButton>
#include <QSaveFile>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStatusBar>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace workbench {

namespace {

// patterns/orbium.lenia, compiled in as a Qt resource (CMakeLists.txt) so the default preset
// does not depend on the working directory. The file and its license are also copied next to
// the executable.
const char* const kOrbiumResource = ":/patterns/orbium.lenia";

// Experiment files (lenia::Project, docs/projects.md).
QString project_suffix() { return QStringLiteral("leniaexp"); }
constexpr qint64 kMaxProjectBytes = 64 * 1024 * 1024;
// Simulation snapshots (lenia::Snapshot, docs/snapshots.md): an experiment plus at most a
// 1024 × 1024 float world.
QString snapshot_suffix() { return QStringLiteral("leniasnap"); }
constexpr qint64 kMaxSnapshotBytes = kMaxProjectBytes + 8 * 1024 * 1024;

// Reads a whole file of at most `limit` bytes; empty optional and `error` set on failure.
std::optional<QByteArray> read_file(const QString& path, qint64 limit, QString& error) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        error = file.errorString();
        return std::nullopt;
    }
    if (file.size() > limit) {
        error = QObject::tr("The file is larger than %1 MB.").arg(limit / (1024 * 1024));
        return std::nullopt;
    }
    QByteArray bytes = file.readAll();
    if (file.error() != QFileDevice::NoError) {
        error = file.errorString();
        return std::nullopt;
    }
    return bytes;
}

// Writes `bytes` to `path` through QSaveFile: a temporary file next to the target, renamed over
// it only once completely written, so an existing file survives any failure intact.
QString write_file_atomically(const QString& path, const std::string& bytes) {
    QSaveFile file(path);
    const bool ok = file.open(QIODevice::WriteOnly) &&
                    file.write(bytes.data(), static_cast<qint64>(bytes.size())) == static_cast<qint64>(bytes.size()) &&
                    file.commit();
    if (ok) return {};
    const QString error = file.errorString();
    return error.isEmpty() ? QObject::tr("Unknown write error") : error;
}

// The project format names fields and palettes in the workbench's order.
static_assert(PaletteCount == lenia::kDisplayPaletteCount);
static_assert(static_cast<int>(Field::Nutrient) + 1 == lenia::kDisplayFieldCount);
static_assert(static_cast<int>(Field::Nutrient) == static_cast<int>(lenia::DisplayField::Nutrient));

QDoubleSpinBox* make_double(double lo, double hi, double step, int decimals, double value) {
    auto* s = new QDoubleSpinBox;
    s->setRange(lo, hi);
    s->setSingleStep(step);
    s->setDecimals(decimals);
    s->setValue(value);
    s->setKeyboardTracking(false);
    return s;
}

QSpinBox* make_int(int lo, int hi, int value) {
    auto* s = new QSpinBox;
    s->setRange(lo, hi);
    s->setValue(value);
    s->setKeyboardTracking(false);
    return s;
}

// Palette and range each field starts with, chosen for Lenia.
ViewSettings default_view(Field field) {
    switch (field) {
    case Field::Density: return {Field::Density, Viridis, 0.0f, 1.0f};
    case Field::Neighborhood: return {Field::Neighborhood, Inferno, 0.0f, 0.3f};
    case Field::Growth: return {Field::Growth, RedBlue, -1.0f, 1.0f};
    case Field::Nutrient: return {Field::Nutrient, Inferno, 0.0f, 1.0f};
    }
    return {};
}

QString describe(const char* name, const lenia::FieldStats& s) {
    return QObject::tr("%1 min %2, mean %3, max %4, total %5")
        .arg(QLatin1String(name))
        .arg(static_cast<double>(s.min), 0, 'f', 4)
        .arg(s.mean, 0, 'f', 4)
        .arg(static_cast<double>(s.max), 0, 'f', 4)
        .arg(s.sum, 0, 'f', 1);
}

QWidget* scrollable(QWidget* content) {
    auto* scroll = new QScrollArea;
    scroll->setWidget(content);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    return scroll;
}

} // namespace

MainWindow::MainWindow(QVulkanInstance* instance) {
    setWindowTitle(tr("Lenia kernel workbench"));

    view_ = new LeniaView;
    view_->setVulkanInstance(instance);
    // Prefer a discrete GPU; QVulkanWindow otherwise takes the first device.
    const auto devices = view_->availablePhysicalDevices();
    for (int i = 0; i < devices.size(); ++i) {
        if (devices[i].deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            view_->setPhysicalDeviceIndex(i);
            break;
        }
    }
    setCentralWidget(QWidget::createWindowContainer(view_, this));

    kernel_ = new KernelModel(this);

    for (std::size_t i = 0; i < view_settings_.size(); ++i) view_settings_[i] = default_view(static_cast<Field>(i));

    auto* kernel_dock = new QDockWidget(tr("Kernel"), this);
    kernel_dock->setObjectName(QStringLiteral("kernel"));
    kernel_dock->setWidget(scrollable(build_kernel_panel()));
    kernel_dock->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
    addDockWidget(Qt::LeftDockWidgetArea, kernel_dock);

    auto* sim_dock = new QDockWidget(tr("Simulation"), this);
    sim_dock->setObjectName(QStringLiteral("simulation"));
    sim_dock->setWidget(scrollable(build_simulation_panel()));
    sim_dock->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
    addDockWidget(Qt::RightDockWidgetArea, sim_dock);

    // Default graph: the Lenia update wired from rule nodes, nothing modulated, so Orbium runs
    // exactly as with the built-in update.
    graph_ = lenia::make_preset(lenia::GraphPreset::Idle);
    node_editor_ = new NodeEditor(&graph_);
    auto* graph_dock = new QDockWidget(tr("Node graph (per-cell rule, kernel and environment over time)"), this);
    graph_dock->setObjectName(QStringLiteral("node_graph"));
    graph_dock->setWidget(node_editor_);
    addDockWidget(Qt::BottomDockWidgetArea, graph_dock);
    resizeDocks({graph_dock}, {400}, Qt::Vertical);

    // Display graph: a second editor on its own graph, tabbed with the node graph. It starts
    // with the View node unlinked, i.e. the panel display.
    display_editor_ = new NodeEditor(&display_graph_);
    auto* display_dock = new QDockWidget(tr("Display graph (viewport colors)"), this);
    display_dock->setObjectName(QStringLiteral("display_graph"));
    display_dock->setWidget(display_editor_);
    addDockWidget(Qt::BottomDockWidgetArea, display_dock);
    tabifyDockWidget(graph_dock, display_dock);
    graph_dock->raise();

    generation_ = new QLabel;
    device_ = new QLabel(tr("Starting Vulkan…"));
    statusBar()->addWidget(generation_);
    statusBar()->addPermanentWidget(device_);

    // One history for both graphs, the kernel and the growth (docs/undo.md). Reset below, once
    // the bundled Orbium is loaded.
    history_ = new EditHistory([this] { return capture_setup(); }, [this](const SetupParts& p) { apply_setup(p); },
                               this);
    connect(history_, &EditHistory::applied, this, [this](const QString& text, bool undo) {
        statusBar()->showMessage((undo ? tr("Undo: %1") : tr("Redo: %1")).arg(text), 5000);
        report_rule(); // after: a rule that does not compile is the more important note
    });
    connect(node_editor_, &NodeEditor::edited, this, [this](const QString& text, const QString& key) {
        history_->commit(tr("%1 (node graph)").arg(text), key.isEmpty() ? key : QStringLiteral("graph ") + key);
    });
    connect(display_editor_, &NodeEditor::edited, this, [this](const QString& text, const QString& key) {
        history_->commit(tr("%1 (display graph)").arg(text), key.isEmpty() ? key : QStringLiteral("display ") + key);
    });
    qApp->installEventFilter(this);

    build_actions();
    build_menus();

    static_kernel_ = std::make_shared<const lenia::Grid>(kernel_->kernel());
    connect(kernel_, &KernelModel::changed, this, [this] {
        static_kernel_ = std::make_shared<const lenia::Grid>(kernel_->kernel());
        warped_kernel_.reset();
        kernel_summary_->setText(kernel_->summary());
        update_preview_context();
        view_->inputs_changed();
    });
    connect(node_editor_, &NodeEditor::graph_changed, view_, &LeniaView::inputs_changed);
    connect(node_editor_, &NodeEditor::graph_changed, this, &MainWindow::report_rule);
    // Selecting a Convolve node previews its U in the Neighborhood field (inputs_at); the
    // re-analysis shows it at once, also while paused.
    connect(node_editor_->canvas(), &NodeCanvas::selection_changed, this, [this](int id) {
        const lenia::Node* n = graph_.node(id);
        const bool was = previewing_;
        previewing_ = n && n->kind == lenia::NodeKind::Convolve;
        if (!previewing_ && !was) return;
        view_->inputs_changed();
        if (previewing_) {
            statusBar()->showMessage(tr("The Neighborhood field shows Convolve node %1's U while it is selected")
                                         .arg(id),
                                     6000);
        }
    });
    connect(node_editor_, &NodeEditor::preset_requested, this, &MainWindow::load_graph_preset);
    connect(node_editor_, &NodeEditor::message, this, [this](const QString& m) { statusBar()->showMessage(m, 6000); });
    connect(display_editor_, &NodeEditor::graph_changed, this, &MainWindow::display_graph_changed);
    connect(display_editor_, &NodeEditor::display_preset_requested, this, &MainWindow::load_display_preset);
    connect(display_editor_, &NodeEditor::wire_view_requested, this, &MainWindow::wire_panel_display);
    connect(display_editor_, &NodeEditor::message, this,
            [this](const QString& m) { statusBar()->showMessage(m, 6000); });
    connect(view_, &LeniaView::generation_changed, this, [this](quint64 g) {
        generation_->setText(tr("Generation %1").arg(g));
        node_editor_->canvas()->set_generation(g);
        display_editor_->canvas()->set_generation(g);
    });
    connect(view_, &LeniaView::device_ready, this, [this](const QString& d) { device_->setText(d); });
    connect(
        view_, &LeniaView::vulkan_failed, this,
        [this](const QString& message) {
            device_->setText(tr("Vulkan error"));
            QMessageBox::critical(this, tr("Vulkan error"), message);
        },
        Qt::QueuedConnection);

    push_view();
    push_brushes();
    view_->set_steps_per_frame(steps_per_frame_->value());
    view_->set_inputs([this](std::uint64_t g) { return inputs_at(g); });
    view_->set_display([this](std::uint64_t g) { return display_at(g); });
    display_graph_changed();
    kernel_->set_normalization(lenia::KernelNormalization::Sum);
    push_growth();
    reset_world(); // random square, replaced below unless the bundled Orbium fails to load
    restore_orbium();
    reset_history();
    resize(1600, 1000);
}

// ---------------------------------------------------------------------------------------------
// Undo history

SetupState MainWindow::capture_setup() const {
    SetupState s;
    s.graph = graph_;
    s.display_graph = display_graph_;
    s.kernel = {kernel_->shape(), kernel_->normalization(), kernel_->paint_layer()};
    s.growth = growth_;
    return s;
}

void MainWindow::apply_setup(const SetupParts& parts) {
    // Like an edit made by hand: inputs_changed / display_changed re-analyze and redraw, the
    // world, its generation and play / pause stay as they are.
    if (parts.kernel) {
        kernel_->restore(parts.kernel->shape, parts.kernel->normalization, parts.kernel->paint);
        sync_kernel_controls();
    }
    if (parts.growth) {
        const lenia::GrowthParams& g = *parts.growth;
        {
            const QSignalBlocker b1(mu_), b2(sigma_), b3(dt_), b4(growth_function_);
            growth_function_->setCurrentIndex(static_cast<int>(g.function));
            mu_->setValue(g.mu);
            sigma_->setValue(g.sigma);
            dt_->setValue(g.dt);
        }
        // Exactly as recorded, not as read back from the spin boxes.
        growth_ = g;
        node_editor_->canvas()->set_panel_growth(growth_);
        view_->inputs_changed();
    }
    if (parts.graph) {
        graph_ = *parts.graph;
        node_editor_->canvas()->graph_restored();
        view_->inputs_changed();
    }
    if (parts.display_graph) {
        display_graph_ = *parts.display_graph;
        display_editor_->canvas()->graph_restored();
        display_graph_changed();
    }
}

void MainWindow::reset_history() {
    if (history_) history_->reset();
}

void MainWindow::commit_panel_edit(const QString& text, const QString& merge_key) {
    if (!history_) return;
    history_->commit(text, merge_key.isEmpty() ? merge_key
                                               : QStringLiteral("panel %1 %2").arg(merge_key).arg(panel_session_));
}

bool MainWindow::eventFilter(QObject* watched, QEvent* event) {
    if (event->type() == QEvent::ShortcutOverride && undo_action_ &&
        (qobject_cast<QAbstractSpinBox*>(watched) || qobject_cast<QAbstractSpinBox*>(watched->parent()))) {
        const QKeySequence keys(static_cast<QKeyEvent*>(event)->keyCombination());
        if (undo_action_->shortcuts().contains(keys) || redo_action_->shortcuts().contains(keys)) {
            event->ignore(); // not overridden: the Edit menu's shortcut runs
            return true;
        }
    }
    return QMainWindow::eventFilter(watched, event);
}

GenerationInputs MainWindow::inputs_at(std::uint64_t generation) {
    const lenia::GraphFrame f = lenia::evaluate(graph_, generation);
    GenerationInputs in;
    in.growth = growth_;
    in.kernel = static_kernel_;
    if (f.kernel_connected && !f.kernel.neutral()) {
        lenia::KernelWarp w = f.kernel;
        // Keep ceil(Re) within the view's kernel size limit.
        const double max_re = (LeniaView::kMaxKernelSize - 1) / 2;
        w.radius_delta = std::min(w.radius_delta, static_cast<float>(max_re / kernel_->shape().radius - 1.0 - 1e-4));
        if (!warped_kernel_ || !(w == warped_for_)) {
            warped_kernel_ = std::make_shared<const lenia::Grid>(kernel_->build(w));
            warped_for_ = w;
        }
        in.kernel = warped_kernel_;
    }
    if (f.environment_connected) in.environment = f.environment;
    if (f.rule_connected || f.nutrient_connected) { // Next nutrient alone: compile_rule's error
        lenia::RuleCompile rule = lenia::compile_rule(graph_, generation, growth_);
        if (!rule.error.empty()) {
            in.kernel.reset(); // LeniaView does not step; report_rule tells the user why
            return in;
        }
        // Same program as last generation (the usual case): same pointer, no upload.
        if (!rule_ || !(*rule_ == rule.program)) {
            rule_ = std::make_shared<const lenia::RuleProgram>(std::move(rule.program));
        }
        in.rule = rule_;
        // A selected Convolve node the rule uses: its U replaces U in the Neighborhood field.
        const int selected = node_editor_->canvas()->selected_node();
        for (const auto& [node, reg] : rule.convolve_registers) {
            if (node == selected) in.preview = reg;
        }
    }
    return in;
}

void MainWindow::report_rule() {
    const lenia::RuleCompile rule = lenia::compile_rule(graph_, view_->generation(), growth_);
    if (!rule.error.empty()) {
        statusBar()->showMessage(tr("Not stepping: %1").arg(QString::fromStdString(rule.error)), 15000);
    }
}

void MainWindow::load_graph_preset(lenia::GraphPreset preset) {
    if (lenia::is_life_preset(preset)) {
        load_life(preset);
        return;
    }
    if (preset == lenia::GraphPreset::NutrientTrail) {
        load_nutrient_trail();
        return;
    }
    if (preset == lenia::GraphPreset::ShortLongRange) {
        load_short_long();
        return;
    }
    // A Lenia rule on the Moore kernel and a glider (or on the two-kernel demo's noise) does
    // nothing useful: bring Orbium back.
    if (life_loaded_ || short_long_loaded_) restore_orbium();
    node_editor_->load_preset(preset);
}

void MainWindow::load_short_long() {
    lenia::Project p = lenia::make_short_long_project();
    const QString error = check_project(p);
    if (!error.isEmpty()) { // cannot happen with the workbench's limits; guards future edits
        QMessageBox::warning(this, tr("Short / long range"), error);
        return;
    }
    apply_project(std::move(p));
    short_long_loaded_ = true;
    run_->setChecked(true);
    statusBar()->showMessage(tr("Two kernels on one field: a cell grows where its short-range average (Kernel K, "
                                "radius %1) beats its long-range one (radius %2). Select a Convolve node to see its "
                                "U in the Neighborhood view; edit a Kernel K to change the spacing.")
                                 .arg(lenia::kShortRadius)
                                 .arg(lenia::kLongRadius),
                             15000);
}

void MainWindow::load_life(lenia::GraphPreset preset) {
    lenia::Project p = lenia::make_life_project(preset);
    const QString error = check_project(p);
    if (!error.isEmpty()) { // cannot happen with the workbench's limits; guards future edits
        QMessageBox::warning(this, tr("Game of Life"), error);
        return;
    }
    apply_project(std::move(p));
    life_loaded_ = true;
    run_->setChecked(true);
    statusBar()->showMessage(tr("Game of Life (B3/S23): raw Moore kernel (8 neighbors, normalization None), glider "
                                "seed. R restarts it; the glider moves one cell diagonally every 4 generations."),
                             12000);
}

void MainWindow::load_nutrient_trail() {
    lenia::Pattern orbium;
    try {
        QFile file(QString::fromLatin1(kOrbiumResource));
        if (!file.open(QIODevice::ReadOnly)) throw std::runtime_error("cannot open the bundled Orbium resource");
        std::istringstream in(file.readAll().toStdString());
        orbium = lenia::parse_pattern(in);
    } catch (const std::exception& e) {
        QMessageBox::warning(this, tr("Nutrient trail"), QString::fromUtf8(e.what()));
        return;
    }
    lenia::Project p = lenia::make_nutrient_trail_project(orbium);
    const QString error = check_project(p);
    if (!error.isEmpty()) { // cannot happen with the workbench's limits; guards future edits
        QMessageBox::warning(this, tr("Nutrient trail"), error);
        return;
    }
    apply_project(std::move(p));
    trail_loaded_ = true;
    run_->setChecked(true);
    statusBar()->showMessage(tr("Density + Nutrient: Orbium eats the nutrient (green) as it glides and leaves a "
                                "dark, slowly recovering trail; growth slows where the nutrient is depleted. View → "
                                "Field shows either field; alt + click a cell to inspect both."),
                             15000);
}

void MainWindow::inspect_fields(std::optional<QPoint> cell) {
    if (view_->capture_pending()) {
        statusBar()->showMessage(tr("An inspection or snapshot is already reading the world"), 5000);
        return;
    }
    // Deferred out of the frame that read the world back (capture_world's rule).
    view_->capture_world([this, cell](const WorldCapture& c) {
        QTimer::singleShot(0, this, [this, cell, c] {
            if (!c.error.isEmpty()) {
                statusBar()->showMessage(tr("Inspection failed: %1").arg(c.error), 8000);
                return;
            }
            QString text = tr("Generation %1\n%2\n%3")
                               .arg(c.generation)
                               .arg(describe("Density A:", lenia::field_stats(c.density)),
                                    describe("Nutrient N:", lenia::field_stats(c.nutrient)));
            if (cell && cell->x() >= 0 && cell->y() >= 0 && static_cast<std::size_t>(cell->x()) < c.density.width &&
                static_cast<std::size_t>(cell->y()) < c.density.height) {
                const auto x = static_cast<std::size_t>(cell->x()), y = static_cast<std::size_t>(cell->y());
                const QString at = tr("Cell (%1, %2): A = %3, N = %4")
                                       .arg(cell->x())
                                       .arg(cell->y())
                                       .arg(static_cast<double>(c.density.at(x, y)), 0, 'f', 4)
                                       .arg(static_cast<double>(c.nutrient.at(x, y)), 0, 'f', 4);
                text += QStringLiteral("\n") + at;
                statusBar()->showMessage(tr("%1 at generation %2").arg(at).arg(c.generation), 10000);
            }
            inspection_->setText(text);
        });
    });
}

QString MainWindow::inspection_text() const { return inspection_->text(); }

std::shared_ptr<const lenia::DisplayProgram> MainWindow::display_at(std::uint64_t generation) {
    lenia::DisplayCompile c = lenia::compile_display(display_graph_, generation);
    if (!c.linked || !c.error.empty()) return nullptr;
    // Same program as last frame (the usual case): same pointer, no upload.
    if (!display_program_ || !(*display_program_ == c.program)) {
        display_program_ = std::make_shared<const lenia::DisplayProgram>(std::move(c.program));
    }
    return display_program_;
}

lenia::DisplayRange MainWindow::panel_range() const {
    const ViewSettings& v = view_settings_[static_cast<std::size_t>(field_->currentIndex())];
    return {static_cast<lenia::DisplayPalette>(v.palette), v.lo, v.hi};
}

void MainWindow::load_display_preset(lenia::DisplayPreset preset) {
    display_graph_ = lenia::make_display_preset(preset, static_cast<lenia::DisplayField>(field_->currentIndex()),
                                                panel_range());
    display_editor_->canvas()->graph_replaced();
    display_graph_changed();
    history_->commit(tr("Load example (display graph)"));
    statusBar()->showMessage(preset == lenia::DisplayPreset::Panel
                                 ? tr("Display graph reset: the view shows the Simulation panel's field and palette")
                                 : tr("Loaded a display graph example; it colors the view, running or paused"),
                             8000);
}

void MainWindow::wire_panel_display() {
    if (!lenia::wire_panel_display(display_graph_, static_cast<lenia::DisplayField>(field_->currentIndex()),
                                   panel_range())) {
        statusBar()->showMessage(tr("The View input is already linked"), 6000);
        return;
    }
    display_editor_->canvas()->graph_replaced();
    display_graph_changed();
    history_->commit(tr("Wire the panel display (display graph)"));
    statusBar()->showMessage(tr("Wired the panel display as nodes: same picture, now editable"), 8000);
}

void MainWindow::display_graph_changed() {
    const lenia::DisplayCompile c = lenia::compile_display(display_graph_, view_->generation());
    const bool graph_shown = c.linked && c.error.empty();
    for (QWidget* w : std::initializer_list<QWidget*>{field_, palette_, lo_, hi_}) w->setEnabled(!graph_shown);
    display_note_->setVisible(graph_shown);
    if (!c.error.empty()) {
        statusBar()->showMessage(tr("Display graph not used: %1").arg(QString::fromStdString(c.error)), 15000);
    }
    view_->display_changed();
}

void MainWindow::update_preview_context() {
    const QSize world = view_->world_size();
    node_editor_->canvas()->set_preview_context(kernel_->shape(), world.width(), world.height());
}

void MainWindow::build_menus() {
    QMenu* file = menuBar()->addMenu(tr("&File"));
    QAction* open = file->addAction(tr("&Open experiment…"), this, &MainWindow::open_project);
    open->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_O));
    open->setStatusTip(tr("Replace the whole setup with a saved experiment; it restarts paused at generation 0"));
    QAction* save = file->addAction(tr("&Save experiment"), this, &MainWindow::save_project);
    save->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_S));
    QAction* save_as = file->addAction(tr("Save experiment &as…"), this, &MainWindow::save_project_as);
    save_as->setShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_S));
    for (QAction* a : {save, save_as}) {
        a->setStatusTip(tr("Save kernel, growth, node graph, world seed and display; the world restarts from "
                           "the seed (generation 0) when reopened"));
    }
    file->addSeparator();
    QAction* open_snap = file->addAction(tr("Open s&napshot…"), this, &MainWindow::open_snapshot);
    open_snap->setShortcut(QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_O));
    open_snap->setStatusTip(tr("Resume a saved running world at its generation, paused, with its experiment"));
    QAction* save_snap = file->addAction(tr("Save sna&pshot…"), this, &MainWindow::save_snapshot_as);
    save_snap->setShortcut(QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_S));
    save_snap->setStatusTip(tr("Save the running world as it is now (density and generation) with the experiment, "
                               "to continue exactly from here later"));
    // One history for the node graph, the display graph, the kernel and the growth
    // (docs/undo.md). Application-wide shortcuts, so they also work in floating panels.
    QMenu* edit = menuBar()->addMenu(tr("&Edit"));
    undo_action_ = history_->create_undo_action(this);
    undo_action_->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Z));
    undo_action_->setStatusTip(tr("Undo the last edit of a graph, the kernel or the growth; the world keeps its "
                                  "generation, density and play / pause"));
    redo_action_ = history_->create_redo_action(this);
    redo_action_->setShortcuts({QKeySequence(Qt::CTRL | Qt::Key_Y), QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_Z)});
    redo_action_->setStatusTip(tr("Redo the last undone edit"));
    for (QAction* a : {undo_action_, redo_action_}) {
        a->setShortcutContext(Qt::ApplicationShortcut);
        edit->addAction(a);
    }

    shortcut_actions_ = {open, save, save_as, open_snap, save_snap, undo_action_, redo_action_};
    // The Vulkan view is a separate QWindow: forward its Ctrl shortcuts to these actions.
    connect(view_, &LeniaView::shortcut_pressed, this, [this](const QKeySequence& keys) {
        for (QAction* a : shortcut_actions_) {
            if (a->shortcuts().contains(keys)) {
                a->trigger(); // nothing if disabled
                return;
            }
        }
    });

    QMenu* view_menu = menuBar()->addMenu(tr("&View"));
    for (QDockWidget* dock : findChildren<QDockWidget*>()) view_menu->addAction(dock->toggleViewAction());

    QMenu* help = menuBar()->addMenu(tr("&Help"));
    help->addAction(tr("Node graph quick guide"), this, [this] {
        QMessageBox::information(
            this, tr("Node graph"),
            tr("<p>The <b>Node graph</b> panel (bottom; View menu to reopen) defines the per-cell update "
               "rule and changes the kernel and the environment while the simulation runs. The Vulkan step "
               "evaluates it once per generation and runs the rule for every cell.</p>"
               "<ul><li><b>Time</b>, <b>Oscillator</b>, <b>Constant</b> produce values (orange, one per "
               "generation).</li>"
               "<li><b>Kernel</b> warps the Kernel panel's ring shape: radius, ring peak position and ring "
               "sharpness, so the profile itself changes, not just its brightness.</li>"
               "<li><b>Environment</b> is a spatial field (stripes, spot, gradient, ripples, uniform) that "
               "can drift and pulse; its <i>Field</i> output is Strength × E(x, y) per cell.</li>"
               "<li>Cell nodes (pink, one value per cell): <b>Density A</b>, <b>Neighborhood U</b> (the "
               "active kernel's weighted sum), <b>Growth G(U)</b>, <b>Add</b>, <b>Multiply</b>, "
               "<b>Clamp</b>, <b>Compare</b>, <b>Select</b> and the exact <b>Life transition</b>. Value "
               "outputs can feed cell inputs.</li>"
               "<li><b>Simulation</b> is what the GPU uses. Its <i>Next state</i> takes the rule's result; "
               "unlinked, the built-in Lenia update runs (older experiments open that way; the "
               "Simulation node's inspector can wire it visibly).</li></ul>"
               "<p>The default graph wires Lenia: clamp(A + dt × (G(U) + field), 0, 1), with Growth "
               "following the Simulation panel. <i>Example → Game of Life</i> (or the Game of Life preset "
               "button) loads Conway's rule with the raw Moore kernel and a glider.</p>"
               "<p><b>Two state fields.</b> Besides the density A every cell has a nutrient N: the "
               "<b>Nutrient N</b> node reads it, the Simulation node's <i>Next nutrient</i> writes it. Each "
               "generation computes next A and next N from the same current state, so the order of the rows "
               "does not matter. Unlinked, the nutrient stays as it is. <i>Example → Density + Nutrient "
               "trail</i> (or Presets → Nutrient trail) shows uptake and recovery that depend on both "
               "fields.</p>"
               "<p>Drag from an output port to an input port to link; drag a linked input away to unlink; "
               "right-click for add / delete; Del removes the selection. Select a node to edit it on the "
               "right. A linked Value input adds to the parameter of the same name; a linked cell input "
               "replaces it.</p>"
               "<p>The graph depends only on the generation number, so Pause, Step ×N and Reset (R) "
               "replay exactly. Try <i>Example → Breathing kernel → Load</i>, then R.</p>"));
    });
    help->addAction(tr("Display graph quick guide"), this, [this] {
        QMessageBox::information(
            this, tr("Display graph"),
            tr("<p>The <b>Display graph</b> panel (tabbed with the node graph) decides how the viewport colors "
               "each cell. The GPU runs it per pixel every frame, running or paused.</p>"
               "<ul><li><b>Fields</b> outputs the cell's density A, neighborhood U, growth and nutrient N.</li>"
               "<li><b>Color ramp</b> maps a value between Low and High onto its color stops. Select it to "
               "edit the stops: drag to move, double-click a stop to recolor it, double-click the bar to add "
               "one, right-click to remove; or set them from a palette.</li>"
               "<li><b>Color</b> is a fixed color; <b>Mix</b> blends two colors (mix, add, multiply, screen) by "
               "a factor that can come from a field or an Oscillator.</li>"
               "<li><b>View</b> shows what is linked into it. Unlinked, the Simulation panel's field and "
               "palette are shown, as before; <i>Wire the panel display as nodes</i> turns them into an "
               "editable ramp.</li></ul>"
               "<p>Try <i>Example → Density + pulsing growth overlay → Load</i>, or <i>Density over "
               "nutrient</i> for the two state fields together. The display graph is saved "
               "with the experiment and never changes the simulation.</p>"));
    });
    help->addAction(tr("Keyboard shortcuts"), this, [this] {
        QMessageBox::information(this, tr("Keyboard shortcuts"),
                                 tr("Space  run / pause\nN  pause and step (Step ×N count)\nR  reset to the seed "
                                    "(generation 0)\nI  impulse seed\nO  restore Orbium\nF  fit the world view\n"
                                    "Alt + click  inspect a cell's density and nutrient\n\n"
                                    "Ctrl+O  open experiment\nCtrl+S  save experiment\n"
                                    "Ctrl+Shift+S  save experiment as\n"
                                    "Ctrl+Alt+O  open snapshot\nCtrl+Alt+S  save snapshot\n\n"
                                    "Ctrl+Z  undo a graph, kernel or growth edit\n"
                                    "Ctrl+Y / Ctrl+Shift+Z  redo\n\n"
                                    "Node canvas: Del / Backspace delete selection, Home frame all"));
    });
}

QWidget* MainWindow::build_kernel_panel() {
    auto* panel = new QWidget;
    auto* layout = new QVBoxLayout(panel);

    auto* shape_box = new QGroupBox(tr("Ring shape"));
    auto* form = new QFormLayout(shape_box);
    radius_ = make_int(1, (LeniaView::kMaxKernelSize - 1) / 2, 13);
    core_ = new QComboBox;
    core_->addItems({tr("Exponential bump"), tr("Polynomial"), tr("Step"), tr("Gaussian bell")});
    ring_count_ = make_int(1, 8, 1);
    normalization_ = new QComboBox;
    normalization_->addItems({tr("None (raw weights)"), tr("Sum = 1"), tr("Positive part = 1"), tr("Σ|w| = 1")});
    form->addRow(tr("Radius R (cells)"), radius_);
    form->addRow(tr("Core"), core_);
    form->addRow(tr("Rings"), ring_count_);
    form->addRow(tr("Normalization"), normalization_);
    layout->addWidget(shape_box);

    auto* profile_box = new QGroupBox(tr("Radial profile (drag to set signed ring weights)"));
    auto* profile_layout = new QVBoxLayout(profile_box);
    auto* profile = new RingProfileWidget(kernel_);
    profile_layout->addWidget(profile);
    layout->addWidget(profile_box);

    auto* paint_box = new QGroupBox(tr("Paint (left +, right −, ctrl erases)"));
    auto* paint_layout = new QVBoxLayout(paint_box);
    canvas_ = new KernelCanvas(kernel_);
    paint_layout->addWidget(canvas_);
    auto* brush_form = new QFormLayout;
    kernel_brush_radius_ = make_double(0.0, 32.0, 0.5, 1, 1.5);
    kernel_brush_strength_ = make_double(0.01, 1.0, 0.05, 2, 0.1);
    ring_brush_ = new QCheckBox(tr("Ring brush (paint whole radius)"));
    brush_form->addRow(tr("Brush radius"), kernel_brush_radius_);
    brush_form->addRow(tr("Strength"), kernel_brush_strength_);
    brush_form->addRow(ring_brush_);
    paint_layout->addLayout(brush_form);
    auto* clear_paint = new QPushButton(tr("Clear paint"));
    paint_layout->addWidget(clear_paint);
    layout->addWidget(paint_box);

    kernel_summary_ = new QLabel;
    kernel_summary_->setWordWrap(true);
    kernel_summary_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(kernel_summary_);
    layout->addStretch();

    // Each edit is an undo step; a spin box's steps are one until it loses focus.
    connect(radius_, &QSpinBox::valueChanged, this, [this](int r) {
        kernel_->set_radius(r);
        commit_panel_edit(tr("Kernel radius"), QStringLiteral("radius"));
    });
    connect(core_, &QComboBox::currentIndexChanged, this, [this](int i) {
        kernel_->set_core(static_cast<lenia::KernelCore>(i));
        commit_panel_edit(tr("Kernel core"));
    });
    connect(ring_count_, &QSpinBox::valueChanged, this, [this](int n) {
        kernel_->set_ring_count(n);
        commit_panel_edit(tr("Kernel rings"), QStringLiteral("rings"));
    });
    connect(normalization_, &QComboBox::currentIndexChanged, this, [this](int i) {
        kernel_->set_normalization(static_cast<lenia::KernelNormalization>(i));
        commit_panel_edit(tr("Kernel normalization"));
    });
    for (QAbstractSpinBox* s : std::initializer_list<QAbstractSpinBox*>{radius_, ring_count_}) {
        connect(s, &QAbstractSpinBox::editingFinished, this, [this] { ++panel_session_; });
    }
    // A drag on the profile or a paint stroke is one step, committed on release.
    connect(profile, &RingProfileWidget::edited, this, [this](const QString& text) { commit_panel_edit(text); });
    connect(canvas_, &KernelCanvas::edited, this, [this](const QString& text) { commit_panel_edit(text); });
    const auto brush_changed = [this] {
        canvas_->set_brush(kernel_brush_radius_->value(), static_cast<float>(kernel_brush_strength_->value()),
                           ring_brush_->isChecked());
    };
    connect(kernel_brush_radius_, &QDoubleSpinBox::valueChanged, this, brush_changed);
    connect(kernel_brush_strength_, &QDoubleSpinBox::valueChanged, this, brush_changed);
    connect(ring_brush_, &QCheckBox::toggled, this, brush_changed);
    connect(clear_paint, &QPushButton::clicked, this, [this] {
        kernel_->clear_paint();
        commit_panel_edit(tr("Clear kernel paint"));
    });
    brush_changed();
    return panel;
}

QWidget* MainWindow::build_simulation_panel() {
    auto* panel = new QWidget;
    auto* layout = new QVBoxLayout(panel);

    auto* growth_box = new QGroupBox(tr("Growth G(U), z = (U − μ) / σ"));
    auto* growth_form = new QFormLayout(growth_box);
    // Item index == lenia::GrowthFunction value.
    growth_function_ = new QComboBox;
    growth_function_->addItems({tr("Gaussian: 2·exp(−z²/2) − 1"), tr("Polynomial: 2·max(0, 1 − z²/9)⁴ − 1")});
    mu_ = make_double(0.0, 2.0, 0.001, 4, 0.15);
    sigma_ = make_double(0.0001, 1.0, 0.001, 4, 0.015);
    dt_ = make_double(0.001, 1.0, 0.01, 3, 0.1);
    growth_form->addRow(tr("Function"), growth_function_);
    growth_form->addRow(tr("μ"), mu_);
    growth_form->addRow(tr("σ"), sigma_);
    growth_form->addRow(tr("dt"), dt_);
    layout->addWidget(growth_box);

    auto* play_box = new QGroupBox(tr("Playback"));
    auto* play_layout = new QVBoxLayout(play_box);
    auto* buttons = new QHBoxLayout;
    run_ = new QPushButton(tr("Pause"));
    run_->setCheckable(true);
    run_->setChecked(true);
    run_->setToolTip(tr("Space"));
    auto* step = new QPushButton(tr("Step"));
    step->setToolTip(tr("N: pause and advance by \"Step ×N\" generations"));
    buttons->addWidget(run_);
    buttons->addWidget(step);
    play_layout->addLayout(buttons);
    auto* play_form = new QFormLayout;
    step_count_ = make_int(1, 10000, 1);
    step_count_->setToolTip(tr("Generations per Step click. The node graph is evaluated per generation, so "
                               "one Step ×10 equals ten Step ×1."));
    steps_per_frame_ = make_int(1, 64, 1);
    play_form->addRow(tr("Step ×N"), step_count_);
    play_form->addRow(tr("Generations per frame"), steps_per_frame_);
    auto* convolution = new QComboBox;
    convolution->addItem(tr("Direct"), static_cast<int>(lenia::ConvolutionMethod::Direct));
    convolution->addItem(tr("FFT (large kernels)"), static_cast<int>(lenia::ConvolutionMethod::Fft));
    convolution->setToolTip(tr("How U = K ∗ A is computed. Direct sums the kernel's weights per cell. FFT multiplies "
                               "spectra instead: its cost does not grow with the kernel, so it is faster for large, "
                               "dense kernels. It applies to power-of-two worlds and kernels with at least %1 nonzero "
                               "weights; other generations use Direct. Same results up to float rounding.")
                                .arg(LeniaView::kFftMinTaps));
    play_form->addRow(tr("Convolution"), convolution);
    play_layout->addLayout(play_form);
    layout->addWidget(play_box);

    auto* world_box = new QGroupBox(tr("World"));
    auto* world_form = new QFormLayout(world_box);
    world_size_ = new QComboBox;
    for (int s : {128, 256, 512, 1024}) world_size_->addItem(QStringLiteral("%1 × %1").arg(s), s);
    world_size_->setCurrentIndex(1);
    seed_ = new QComboBox;
    seed_->addItems({tr("Random square"), tr("Loaded pattern"), tr("Impulse"), tr("Empty")});
    random_seed_ = make_int(0, 1 << 30, 1);
    auto* reset = new QPushButton(tr("Reset"));
    reset->setToolTip(tr("R: restore the seed"));
    auto* impulse = new QPushButton(tr("Impulse"));
    impulse->setToolTip(tr("I: single 1.0 cell in the middle; its neighborhood view shows the kernel"));
    auto* seed_buttons = new QHBoxLayout;
    seed_buttons->addWidget(reset);
    seed_buttons->addWidget(impulse);
    nutrient_start_ = new QComboBox;
    nutrient_start_->addItems({tr("Uniform"), tr("Random"), tr("Gradient (left to right)")});
    nutrient_start_->setToolTip(tr("How the nutrient N, the second state field, starts on Reset: the level "
                                   "everywhere, random values up to it (from the random seed), or rising from 0 "
                                   "at the left edge to it at the right. Only rules with Nutrient nodes read it."));
    nutrient_level_ = make_double(0.0, 1.0, 0.05, 4, 1.0);
    nutrient_level_->setToolTip(nutrient_start_->toolTip());
    world_brush_radius_ = make_int(0, 64, 4);
    world_brush_value_ = make_double(0.0, 1.0, 0.1, 2, 1.0);
    world_brush_field_ = new QComboBox;
    world_brush_field_->addItems({tr("Density A"), tr("Nutrient N")});
    world_brush_field_->setToolTip(tr("The state field left drag paints into (shift + drag sets it to 0)"));
    world_form->addRow(tr("Size"), world_size_);
    world_form->addRow(tr("Seed"), seed_);
    world_form->addRow(tr("Random seed"), random_seed_);
    world_form->addRow(tr("Nutrient start"), nutrient_start_);
    world_form->addRow(tr("Nutrient level"), nutrient_level_);
    world_form->addRow(seed_buttons);
    world_form->addRow(tr("Paint field"), world_brush_field_);
    world_form->addRow(tr("Paint radius"), world_brush_radius_);
    world_form->addRow(tr("Paint value"), world_brush_value_);
    auto* hint = new QLabel(tr("In the view: left drag paints, shift + drag erases, alt + click inspects a cell, "
                               "right drag pans, wheel zooms, F refits. Seed and nutrient settings apply on "
                               "Reset."));
    hint->setWordWrap(true);
    world_form->addRow(hint);
    layout->addWidget(world_box);

    auto* preset_box = new QGroupBox(tr("Presets"));
    auto* preset_layout = new QVBoxLayout(preset_box);
    auto* orbium = new QPushButton(tr("Restore Orbium"));
    orbium->setToolTip(tr("O: reload the bundled Orbium (kernel, growth and cells) with the default wired Lenia "
                          "rule graph, and restart from it"));
    auto* life = new QPushButton(tr("Game of Life (glider)"));
    life->setToolTip(tr("Conway's B3/S23 as a node-graph rule: raw 8-neighbor Moore kernel, a glider in a "
                        "128 × 128 world. Restore Orbium returns to Lenia."));
    auto* trail = new QPushButton(tr("Nutrient trail (Density + Nutrient)"));
    trail->setToolTip(tr("Two interacting state fields: Orbium consumes a nutrient field as it glides, leaving a "
                         "depleted trail that slowly recovers, and grows less where the nutrient is low. Both "
                         "fields update together each generation. Restore Orbium returns to plain Lenia."));
    auto* short_long = new QPushButton(tr("Short / long range (two kernels)"));
    short_long->setToolTip(tr("Two Kernel K nodes convolve the density: a cell grows where its short-range "
                              "average beats its long-range one. Random noise settles into spots and labyrinths. "
                              "Select a Convolve node to see its U in the Neighborhood view. Restore Orbium "
                              "returns to Lenia."));
    auto* load = new QPushButton(tr("Load pattern file…"));
    attribution_ = new QLabel;
    attribution_->setWordWrap(true);
    attribution_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    preset_layout->addWidget(orbium);
    preset_layout->addWidget(life);
    preset_layout->addWidget(trail);
    preset_layout->addWidget(short_long);
    preset_layout->addWidget(load);
    preset_layout->addWidget(attribution_);
    layout->addWidget(preset_box);

    auto* view_box = new QGroupBox(tr("View"));
    auto* view_form = new QFormLayout(view_box);
    field_ = new QComboBox;
    field_->addItems({tr("Density A"), tr("Neighborhood U = K ∗ A"), tr("Growth G(U)"), tr("Nutrient N")});
    palette_ = new QComboBox;
    palette_->addItems(palette_names());
    lo_ = make_double(-100.0, 100.0, 0.01, 4, 0.0);
    hi_ = make_double(-100.0, 100.0, 0.01, 4, 1.0);
    auto* fit = new QPushButton(tr("Fit view"));
    view_form->addRow(tr("Field"), field_);
    view_form->addRow(tr("Palette"), palette_);
    view_form->addRow(tr("Low"), lo_);
    view_form->addRow(tr("High"), hi_);
    view_form->addRow(fit);
    auto* field_note = new QLabel(tr("While running, U and G(U) are those of the state each step started from. "
                                     "With a wired rule, G(U) is its (first) Growth node's output, or next − current "
                                     "when the rule has none (the Game of Life); without one it is the built-in "
                                     "update's G(U) including the environment term."));
    field_note->setWordWrap(true);
    view_form->addRow(field_note);
    display_note_ = new QLabel(tr("The display graph's View node is linked, so it colors the view; the field and "
                                  "palette above are not used. Unlink View (or load its \"Panel display\" example) "
                                  "to use them again."));
    display_note_->setWordWrap(true);
    display_note_->setVisible(false);
    view_form->addRow(display_note_);
    auto* inspect = new QPushButton(tr("Inspect fields"));
    inspect->setToolTip(tr("Read both state fields back from the GPU once and show their range, mean and total "
                           "(alt + click a cell in the view for its values). Playback never reads them back."));
    inspection_ = new QLabel(tr("Not inspected yet"));
    inspection_->setWordWrap(true);
    inspection_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    view_form->addRow(inspect);
    view_form->addRow(inspection_);
    layout->addWidget(view_box);
    layout->addStretch();

    // push_growth first, then the undo step (connections run in order). Programmatic changes
    // (patterns, experiments, undo itself) block these signals and push_growth directly.
    for (QDoubleSpinBox* s : {mu_, sigma_, dt_}) {
        connect(s, &QDoubleSpinBox::valueChanged, this, &MainWindow::push_growth);
        connect(s, &QDoubleSpinBox::editingFinished, this, [this] { ++panel_session_; });
    }
    connect(mu_, &QDoubleSpinBox::valueChanged, this, [this] { commit_panel_edit(tr("Growth μ"), QStringLiteral("mu")); });
    connect(sigma_, &QDoubleSpinBox::valueChanged, this,
            [this] { commit_panel_edit(tr("Growth σ"), QStringLiteral("sigma")); });
    connect(dt_, &QDoubleSpinBox::valueChanged, this, [this] { commit_panel_edit(tr("Growth dt"), QStringLiteral("dt")); });
    connect(growth_function_, &QComboBox::currentIndexChanged, this, &MainWindow::push_growth);
    connect(growth_function_, &QComboBox::currentIndexChanged, this,
            [this] { commit_panel_edit(tr("Growth function")); });
    connect(run_, &QPushButton::toggled, this, &MainWindow::set_running);
    connect(step, &QPushButton::clicked, this, &MainWindow::step_once);
    connect(steps_per_frame_, &QSpinBox::valueChanged, view_, &LeniaView::set_steps_per_frame);
    connect(convolution, &QComboBox::currentIndexChanged, this, [this, convolution] {
        const auto method = static_cast<lenia::ConvolutionMethod>(convolution->currentData().toInt());
        view_->set_convolution(method);
        if (method == lenia::ConvolutionMethod::Fft) {
            statusBar()->showMessage(tr("FFT convolution selected: used for power-of-two worlds and kernels with at "
                                        "least %1 nonzero weights")
                                         .arg(LeniaView::kFftMinTaps),
                                     8000);
        }
    });
    connect(view_, &LeniaView::convolution_used, this, [this](bool fft) {
        statusBar()->showMessage(fft ? tr("Convolution: FFT")
                                     : tr("Convolution: direct (world or kernel not suited to FFT, or Direct chosen)"),
                                 6000);
    });
    connect(world_size_, &QComboBox::currentIndexChanged, this, &MainWindow::reset_world);
    connect(reset, &QPushButton::clicked, this, &MainWindow::reset_world);
    connect(impulse, &QPushButton::clicked, this, &MainWindow::seed_impulse);
    connect(world_brush_radius_, &QSpinBox::valueChanged, this, &MainWindow::push_brushes);
    connect(world_brush_value_, &QDoubleSpinBox::valueChanged, this, &MainWindow::push_brushes);
    connect(world_brush_field_, &QComboBox::currentIndexChanged, this, &MainWindow::push_brushes);
    connect(orbium, &QPushButton::clicked, this, &MainWindow::restore_orbium);
    connect(life, &QPushButton::clicked, this, [this] { load_life(lenia::GraphPreset::GameOfLife); });
    connect(trail, &QPushButton::clicked, this, &MainWindow::load_nutrient_trail);
    connect(short_long, &QPushButton::clicked, this, &MainWindow::load_short_long);
    connect(inspect, &QPushButton::clicked, this, [this] { inspect_fields(); });
    connect(view_, &LeniaView::cell_probed, this, [this](int x, int y) { inspect_fields(QPoint(x, y)); });
    connect(load, &QPushButton::clicked, this, &MainWindow::load_pattern);
    connect(field_, &QComboBox::currentIndexChanged, this, [this] {
        sync_view_controls();
        push_view();
    });
    const auto view_edited = [this] {
        ViewSettings& v = view_settings_[static_cast<std::size_t>(field_->currentIndex())];
        v.palette = palette_->currentIndex();
        v.lo = static_cast<float>(lo_->value());
        v.hi = static_cast<float>(hi_->value());
        push_view();
    };
    connect(palette_, &QComboBox::currentIndexChanged, this, view_edited);
    connect(lo_, &QDoubleSpinBox::valueChanged, this, view_edited);
    connect(hi_, &QDoubleSpinBox::valueChanged, this, view_edited);
    connect(fit, &QPushButton::clicked, view_, &LeniaView::fit_view);

    sync_view_controls();
    return panel;
}

void MainWindow::build_actions() {
    const auto toggle_run = [this] { run_->toggle(); };
    const auto step = [this] { step_once(); };
    const auto reset = [this] { reset_world(); };
    const auto impulse = [this] { seed_impulse(); };
    const auto fit = [this] { view_->fit_view(); };
    const auto orbium = [this] { restore_orbium(); };

    const auto add = [this](Qt::Key key, auto slot) {
        auto* action = new QAction(this);
        action->setShortcut(QKeySequence(key));
        connect(action, &QAction::triggered, this, slot);
        addAction(action);
    };
    add(Qt::Key_Space, toggle_run);
    add(Qt::Key_N, step);
    add(Qt::Key_R, reset);
    add(Qt::Key_I, impulse);
    add(Qt::Key_F, fit);
    add(Qt::Key_O, orbium);

    // The Vulkan view is a separate QWindow, so widget shortcuts do not see its key presses.
    connect(view_, &LeniaView::key_pressed, this, [=](int key) {
        switch (key) {
        case Qt::Key_Space: toggle_run(); break;
        case Qt::Key_N: step(); break;
        case Qt::Key_R: reset(); break;
        case Qt::Key_I: impulse(); break;
        case Qt::Key_F: fit(); break;
        case Qt::Key_O: orbium(); break;
        default: break;
        }
    });
}

void MainWindow::sync_kernel_controls() {
    const QSignalBlocker b1(radius_), b2(core_), b3(ring_count_), b4(normalization_);
    const lenia::KernelShape& s = kernel_->shape();
    radius_->setValue(s.radius);
    core_->setCurrentIndex(static_cast<int>(s.core));
    ring_count_->setValue(static_cast<int>(s.rings.size()));
    normalization_->setCurrentIndex(static_cast<int>(kernel_->normalization()));
}

void MainWindow::sync_view_controls() {
    const QSignalBlocker b1(palette_), b2(lo_), b3(hi_);
    const ViewSettings& v = view_settings_[static_cast<std::size_t>(field_->currentIndex())];
    palette_->setCurrentIndex(v.palette);
    lo_->setValue(v.lo);
    hi_->setValue(v.hi);
}

void MainWindow::push_growth() {
    growth_ = {static_cast<float>(mu_->value()), static_cast<float>(sigma_->value()), static_cast<float>(dt_->value()),
               static_cast<lenia::GrowthFunction>(growth_function_->currentIndex())};
    // Growth nodes with Source = panel follow these values.
    if (node_editor_) node_editor_->canvas()->set_panel_growth(growth_);
    view_->inputs_changed();
}

void MainWindow::push_view() { view_->set_view(view_settings_[static_cast<std::size_t>(field_->currentIndex())]); }

void MainWindow::push_brushes() {
    view_->set_brush(world_brush_radius_->value(), static_cast<float>(world_brush_value_->value()));
    view_->set_brush_field(static_cast<BrushField>(world_brush_field_->currentIndex()));
}

void MainWindow::reset_world() {
    const int size = world_size_->currentData().toInt();
    const int seed = seed_->currentIndex();
    const lenia::NutrientSeed nutrient{static_cast<lenia::NutrientStart>(nutrient_start_->currentIndex()),
                                       static_cast<float>(nutrient_level_->value())};

    lenia::Grid world(static_cast<std::size_t>(size), static_cast<std::size_t>(size), 0.0f);
    if (seed == SeedImpulse) {
        world.at(world.width / 2, world.height / 2) = 1.0f; // as LeniaView::seed_impulse
    } else if (seed == SeedPattern && pattern_) {
        lenia::stamp_pattern(world, pattern_->cells, world.width / 2, world.height / 2);
    } else if (seed == SeedPattern) {
        statusBar()->showMessage(tr("No pattern loaded; seeded a random square instead"), 5000);
    }
    if (seed == SeedRandom || (seed == SeedPattern && !pattern_)) {
        std::mt19937 rng(static_cast<std::uint32_t>(random_seed_->value()));
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        const std::size_t side = world.width / 2;
        const std::size_t start = (world.width - side) / 2;
        for (std::size_t y = start; y < start + side; ++y) {
            for (std::size_t x = start; x < start + side; ++x) world.at(x, y) = uniform(rng);
        }
    }
    view_->upload_world(world, lenia::seed_nutrient(nutrient, size, random_seed_->value()), 0);
    update_preview_context();
}

void MainWindow::seed_impulse() {
    seed_->setCurrentIndex(SeedImpulse);
    reset_world();
}

void MainWindow::set_running(bool running) {
    view_->set_paused(!running);
    run_->setText(running ? tr("Pause") : tr("Run"));
}

void MainWindow::step_once() {
    run_->setChecked(false);
    view_->step(step_count_->value());
}

void MainWindow::restore_orbium() {
    lenia::Pattern p;
    try {
        QFile file(QString::fromLatin1(kOrbiumResource));
        if (!file.open(QIODevice::ReadOnly)) {
            throw std::runtime_error("cannot open the bundled Orbium resource");
        }
        std::istringstream in(file.readAll().toStdString());
        p = lenia::parse_pattern(in);
    } catch (const std::exception& e) {
        QMessageBox::warning(this, tr("Restore Orbium"), QString::fromUtf8(e.what()));
        return;
    }
    // The default graph too: after the Game of Life, Orbium needs the Lenia rule back.
    graph_ = lenia::make_preset(lenia::GraphPreset::Idle);
    node_editor_->canvas()->graph_replaced();
    view_->inputs_changed();
    if (life_loaded_ || short_long_loaded_) {
        // Undo the Life experiment's 128 world and binary display, or the two-kernel demo's.
        life_loaded_ = false;
        short_long_loaded_ = false;
        const QSignalBlocker b(world_size_);
        world_size_->setCurrentIndex(world_size_->findData(256));
        for (std::size_t i = 0; i < view_settings_.size(); ++i) view_settings_[i] = default_view(static_cast<Field>(i));
        sync_view_controls();
        push_view();
    }
    if (trail_loaded_) {
        // Undo the nutrient trail demo's display: the panel's density view again.
        trail_loaded_ = false;
        display_graph_ = lenia::make_display_preset(lenia::DisplayPreset::Panel);
        display_editor_->canvas()->graph_replaced();
        display_graph_changed();
        const QSignalBlocker b(field_);
        field_->setCurrentIndex(static_cast<int>(Field::Density));
        sync_view_controls();
        push_view();
    }
    const bool applied = apply_pattern(std::move(p), tr("Restore Orbium"));
    reset_history(); // the graph was replaced even if the pattern was not
    if (applied) run_->setChecked(true);
}

void MainWindow::load_pattern() {
    const QString path = QFileDialog::getOpenFileName(this, tr("Load Lenia pattern"), QString(),
                                                      tr("Lenia patterns (*.txt *.lenia);;All files (*)"));
    if (path.isEmpty()) return;
    lenia::Pattern p;
    try {
        p = lenia::load_pattern_file(path.toStdString());
    } catch (const std::exception& e) {
        QMessageBox::warning(this, tr("Load pattern"), QString::fromUtf8(e.what()));
        return;
    }
    apply_pattern(std::move(p), tr("Load pattern"));
}

bool MainWindow::apply_pattern(lenia::Pattern p, const QString& title) {
    if (p.kernel.radius > (LeniaView::kMaxKernelSize - 1) / 2) {
        QMessageBox::warning(this, title,
                             tr("Kernel radius %1 exceeds the workbench limit of %2.")
                                 .arg(p.kernel.radius)
                                 .arg((LeniaView::kMaxKernelSize - 1) / 2));
        return false;
    }

    {
        const QSignalBlocker b(kernel_);
        kernel_->set_normalization(lenia::KernelNormalization::Sum);
        kernel_->clear_paint();
    }
    kernel_->set_shape(p.kernel);
    sync_kernel_controls();
    {
        const QSignalBlocker b1(mu_), b2(sigma_), b3(dt_), b4(growth_function_);
        growth_function_->setCurrentIndex(static_cast<int>(p.growth.function));
        mu_->setValue(p.growth.mu);
        sigma_->setValue(p.growth.sigma);
        dt_->setValue(p.growth.dt);
    }
    push_growth();

    pattern_ = std::move(p);
    set_attribution();
    {
        const QSignalBlocker b(seed_);
        seed_->setCurrentIndex(SeedPattern);
    }
    reset_world();
    // A new creature and world: the old steps describe another setup.
    reset_history();
    return true;
}

void MainWindow::set_attribution() {
    if (!pattern_) {
        attribution_->setText(tr("No pattern loaded"));
        return;
    }
    const lenia::Pattern& p = *pattern_;
    const QString name = p.name.empty() ? tr("Unnamed pattern") : QString::fromStdString(p.name);
    const QString source = p.source.empty() ? tr("no source recorded") : QString::fromStdString(p.source);
    const QString license = p.license.empty() ? tr("no license recorded") : QString::fromStdString(p.license);
    attribution_->setText(tr("%1 (%2×%3 cells)\nSource: %4\nLicense: %5")
                              .arg(name)
                              .arg(p.cells.width)
                              .arg(p.cells.height)
                              .arg(source, license));
}

// ---------------------------------------------------------------------------------------------
// Experiment files

lenia::Project MainWindow::capture_project() const {
    lenia::Project p;
    p.kernel = kernel_->shape();
    p.normalization = kernel_->normalization();
    if (kernel_->has_paint()) p.paint = kernel_->paint_layer();
    p.growth = growth_;
    p.world_size = world_size_->currentData().toInt();
    p.seed = static_cast<lenia::SeedKind>(seed_->currentIndex());
    p.random_seed = random_seed_->value();
    p.nutrient = {static_cast<lenia::NutrientStart>(nutrient_start_->currentIndex()),
                  static_cast<float>(nutrient_level_->value())};
    p.pattern = pattern_;
    p.field = static_cast<lenia::DisplayField>(field_->currentIndex());
    for (std::size_t i = 0; i < view_settings_.size(); ++i) {
        const ViewSettings& v = view_settings_[i];
        p.display[i] = {static_cast<lenia::DisplayPalette>(v.palette), v.lo, v.hi};
    }
    p.graph = graph_;
    p.display_graph = display_graph_;
    return p;
}

QString MainWindow::check_project(const lenia::Project& p) const {
    const int max_radius = (LeniaView::kMaxKernelSize - 1) / 2;
    if (p.kernel.radius > max_radius) {
        return tr("Kernel radius %1 exceeds the workbench limit of %2.").arg(p.kernel.radius).arg(max_radius);
    }
    // Compared as the spin box will hold it: a float saved at a bound (σ = 0.0001f is just below
    // 0.0001) must still read back.
    const auto in_range = [](const QDoubleSpinBox* s, double v) {
        const double scale = std::pow(10.0, s->decimals());
        const double shown = std::round(v * scale) / scale;
        return shown >= s->minimum() && shown <= s->maximum();
    };
    const auto range_error = [](const QString& name, const QDoubleSpinBox* s, double v) {
        return tr("%1 = %2 is outside the workbench range [%3, %4].").arg(name).arg(v).arg(s->minimum()).arg(s->maximum());
    };
    if (!in_range(mu_, p.growth.mu)) return range_error(tr("Growth μ"), mu_, p.growth.mu);
    if (!in_range(sigma_, p.growth.sigma)) return range_error(tr("Growth σ"), sigma_, p.growth.sigma);
    if (!in_range(dt_, p.growth.dt)) return range_error(tr("Growth dt"), dt_, p.growth.dt);
    if (world_size_->findData(p.world_size) < 0) {
        return tr("World size %1 is not one the workbench offers (128, 256, 512, 1024).").arg(p.world_size);
    }
    if (p.random_seed > random_seed_->maximum()) {
        return tr("Random seed %1 exceeds the workbench limit of %2.").arg(p.random_seed).arg(random_seed_->maximum());
    }
    for (const lenia::DisplayRange& r : p.display) {
        if (!in_range(lo_, r.lo)) return range_error(tr("Display low"), lo_, r.lo);
        if (!in_range(hi_, r.hi)) return range_error(tr("Display high"), hi_, r.hi);
    }
    // The level reseeds the world, so it must survive the spin box exactly.
    const double scale = std::pow(10.0, nutrient_level_->decimals());
    if (static_cast<float>(std::round(p.nutrient.level * scale) / scale) != p.nutrient.level) {
        return tr("Nutrient level %1 has more than %2 decimals, which the workbench cannot hold exactly.")
            .arg(static_cast<double>(p.nutrient.level))
            .arg(nutrient_level_->decimals());
    }
    return {};
}

void MainWindow::apply_project(lenia::Project p, const lenia::WorldState* world, std::uint64_t generation) {
    // Pause first, so nothing steps between the reset below and the user seeing generation 0
    // (or the snapshot's generation).
    run_->setChecked(false);
    life_loaded_ = false;
    trail_loaded_ = false;
    short_long_loaded_ = false;

    kernel_->restore(p.kernel, p.normalization, p.paint); // validated by parse_project
    sync_kernel_controls();
    {
        const QSignalBlocker b1(mu_), b2(sigma_), b3(dt_), b4(growth_function_);
        growth_function_->setCurrentIndex(static_cast<int>(p.growth.function));
        mu_->setValue(p.growth.mu);
        sigma_->setValue(p.growth.sigma);
        dt_->setValue(p.growth.dt);
    }
    push_growth();

    pattern_ = std::move(p.pattern);
    set_attribution();

    for (std::size_t i = 0; i < view_settings_.size(); ++i) {
        const lenia::DisplayRange& r = p.display[i];
        view_settings_[i] = {static_cast<Field>(i), static_cast<int>(r.palette), r.lo, r.hi};
    }
    {
        const QSignalBlocker b(field_);
        field_->setCurrentIndex(static_cast<int>(p.field));
    }
    sync_view_controls();
    push_view();

    graph_ = std::move(p.graph);
    node_editor_->canvas()->graph_replaced();
    view_->inputs_changed();

    display_graph_ = std::move(p.display_graph);
    display_editor_->canvas()->graph_replaced();
    display_graph_changed();

    {
        const QSignalBlocker b1(world_size_), b2(seed_), b3(random_seed_), b4(nutrient_start_), b5(nutrient_level_);
        world_size_->setCurrentIndex(world_size_->findData(p.world_size));
        seed_->setCurrentIndex(static_cast<int>(p.seed));
        random_seed_->setValue(p.random_seed);
        nutrient_start_->setCurrentIndex(static_cast<int>(p.nutrient.start));
        nutrient_level_->setValue(p.nutrient.level);
    }
    if (world) {
        // The snapshot's world as it was: generation, kernel warp, rule and display all follow
        // from the generation, so the next step computes what the saved session would have.
        view_->upload_world(world->density, world->nutrient, generation);
        update_preview_context();
    } else {
        reset_world();
    }
    // Another experiment or snapshot: its undo history starts here, empty.
    reset_history();
}

QString MainWindow::open_experiment_file(const QString& path) {
    // Everything is read and checked before the current experiment is touched.
    QString error;
    const std::optional<QByteArray> bytes = read_file(path, kMaxProjectBytes, error);
    if (!bytes) return error;
    std::optional<lenia::Project> project;
    std::istringstream in(bytes->toStdString());
    try {
        project = lenia::parse_project(in);
    } catch (const std::exception& e) {
        return QString::fromUtf8(e.what());
    }
    error = check_project(*project);
    if (!error.isEmpty()) return error;

    apply_project(std::move(*project));
    project_path_ = path;
    document_name_ = QFileInfo(path).fileName();
    update_title();
    return {};
}

void MainWindow::open_project() {
    const QString dir = project_path_.isEmpty() ? QString() : QFileInfo(project_path_).absolutePath();
    const QString path = QFileDialog::getOpenFileName(this, tr("Open experiment"), dir,
                                                      tr("Lenia experiments (*.%1);;All files (*)").arg(project_suffix()));
    if (path.isEmpty()) return;
    const QString file_name = QFileInfo(path).fileName();
    const QString error = open_experiment_file(path);
    if (!error.isEmpty()) {
        QMessageBox::warning(this, tr("Open experiment"),
                             tr("%1 was not opened; the current experiment is unchanged.\n\n%2").arg(file_name, error));
        return;
    }
    const QString rule = lenia::evaluate(graph_, 0).rule_connected
                             ? QString()
                             : tr(" Its Next state is unlinked, so the built-in Lenia update runs; select the "
                                  "Simulation node to wire it as editable nodes.");
    statusBar()->showMessage(tr("Opened %1 at generation 0, paused (experiments store the setup, not a running "
                                "world). Space runs it.%2")
                                 .arg(file_name, rule),
                             15000);
    report_rule();
}

bool MainWindow::save_project() {
    if (project_path_.isEmpty()) return save_project_as();
    return write_project_file(project_path_);
}

bool MainWindow::save_project_as() {
    const QString suggested = project_path_.isEmpty() ? tr("experiment.%1").arg(project_suffix()) : project_path_;
    QString path = QFileDialog::getSaveFileName(this, tr("Save experiment"), suggested,
                                                tr("Lenia experiments (*.%1);;All files (*)").arg(project_suffix()));
    if (path.isEmpty()) return false;
    if (QFileInfo(path).suffix().isEmpty()) path += QStringLiteral(".") + project_suffix();
    if (!write_project_file(path)) return false;
    project_path_ = path;
    update_title();
    return true;
}

bool MainWindow::write_project_file(const QString& path) {
    const std::string text = lenia::write_project(capture_project());
    const QString file_name = QFileInfo(path).fileName();
    // Never write a file this build could not open again.
    try {
        std::istringstream check(text);
        const QString error = check_project(lenia::parse_project(check));
        if (!error.isEmpty()) throw std::runtime_error(error.toStdString());
    } catch (const std::exception& e) {
        QMessageBox::warning(this, tr("Save experiment"),
                             tr("%1 was not saved: the experiment does not read back.\n\n%2")
                                 .arg(file_name, QString::fromUtf8(e.what())));
        return false;
    }

    const QString error = write_file_atomically(path, text);
    if (!error.isEmpty()) {
        QMessageBox::warning(this, tr("Save experiment"),
                             tr("%1 was not saved (any existing file is unchanged).\n\n%2").arg(file_name, error));
        return false;
    }
    document_name_ = file_name;
    update_title();
    statusBar()->showMessage(tr("Saved %1. Reopening restarts it at generation 0 from the seed; cells painted "
                                "into the running world are not stored (Save snapshot stores them).")
                                 .arg(file_name),
                             10000);
    return true;
}

// ---------------------------------------------------------------------------------------------
// Simulation snapshots

QString MainWindow::open_snapshot_file(const QString& path) {
    // Read, parse and check everything (format, checksum, experiment, world size) before the
    // current setup or world is touched.
    QString error;
    const std::optional<QByteArray> bytes = read_file(path, kMaxSnapshotBytes, error);
    if (!bytes) return error;
    std::optional<lenia::Snapshot> snapshot;
    try {
        snapshot = lenia::parse_snapshot(std::string_view(bytes->constData(), static_cast<std::size_t>(bytes->size())));
    } catch (const std::exception& e) {
        return QString::fromUtf8(e.what());
    }
    error = check_project(snapshot->project);
    if (!error.isEmpty()) return error;

    const lenia::WorldState world{std::move(snapshot->density), std::move(snapshot->nutrient)};
    apply_project(std::move(snapshot->project), &world, snapshot->generation);
    // The experiment came from a snapshot: Save experiment asks where to put it.
    project_path_.clear();
    document_name_ = QFileInfo(path).fileName();
    snapshot_dir_ = QFileInfo(path).absolutePath();
    update_title();
    report_rule();
    return {};
}

void MainWindow::save_snapshot_file(const QString& path, std::function<void(const QString& error)> done) {
    // The callback runs inside the frame that read the world back: capture_project() there is
    // exactly the setup the world's next generation will be computed with. Writing the file and
    // reporting wait until the frame is over.
    const auto report = [this, done](QString error) {
        QTimer::singleShot(0, this, [done, error] {
            if (done) done(error);
        });
    };
    view_->capture_world([this, path, report](const WorldCapture& c) {
        if (!c.error.isEmpty()) {
            report(c.error);
            return;
        }
        auto snapshot = std::make_shared<lenia::Snapshot>();
        snapshot->project = capture_project();
        snapshot->generation = c.generation;
        snapshot->density = c.density;
        snapshot->nutrient = c.nutrient;
        QTimer::singleShot(0, this, [this, path, report, snapshot] { report(write_snapshot_file(path, *snapshot)); });
    });
}

QString MainWindow::write_snapshot_file(const QString& path, const lenia::Snapshot& snapshot) {
    std::string bytes;
    try {
        bytes = lenia::write_snapshot(snapshot);
        // Never write a file this build could not open again.
        const lenia::Snapshot check = lenia::parse_snapshot(bytes);
        const QString error = check_project(check.project);
        if (!error.isEmpty()) throw std::runtime_error(error.toStdString());
    } catch (const std::exception& e) {
        return tr("The snapshot does not read back: %1").arg(QString::fromUtf8(e.what()));
    }
    return write_file_atomically(path, bytes);
}

void MainWindow::open_snapshot() {
    const QString path = QFileDialog::getOpenFileName(this, tr("Open snapshot"), snapshot_dir_,
                                                      tr("Lenia snapshots (*.%1);;All files (*)").arg(snapshot_suffix()));
    if (path.isEmpty()) return;
    const QString file_name = QFileInfo(path).fileName();
    const QString error = open_snapshot_file(path);
    if (!error.isEmpty()) {
        QMessageBox::warning(this, tr("Open snapshot"),
                             tr("%1 was not opened; the current experiment and world are unchanged.\n\n%2")
                                 .arg(file_name, error));
        return;
    }
    statusBar()->showMessage(tr("Opened %1 at generation %2, paused. Space continues it exactly; R restarts its "
                                "experiment from the seed (generation 0).")
                                 .arg(file_name)
                                 .arg(view_->generation()),
                             15000);
}

void MainWindow::save_snapshot_as() {
    if (view_->capture_pending()) {
        statusBar()->showMessage(tr("A snapshot is already being saved"), 5000);
        return;
    }
    const QString name = QStringLiteral("generation-%1.%2").arg(view_->generation()).arg(snapshot_suffix());
    const QString suggested = snapshot_dir_.isEmpty() ? name : snapshot_dir_ + QStringLiteral("/") + name;
    QString path = QFileDialog::getSaveFileName(this, tr("Save snapshot"), suggested,
                                                tr("Lenia snapshots (*.%1);;All files (*)").arg(snapshot_suffix()));
    if (path.isEmpty()) return;
    if (QFileInfo(path).suffix().isEmpty()) path += QStringLiteral(".") + snapshot_suffix();
    snapshot_dir_ = QFileInfo(path).absolutePath();
    const QString file_name = QFileInfo(path).fileName();
    save_snapshot_file(path, [this, file_name](const QString& error) {
        if (!error.isEmpty()) {
            QMessageBox::warning(this, tr("Save snapshot"),
                                 tr("%1 was not saved (any existing file is unchanged).\n\n%2").arg(file_name, error));
            return;
        }
        document_name_ = file_name;
        update_title();
        statusBar()->showMessage(tr("Saved %1: the world, its generation and the experiment. Opening it continues "
                                    "exactly from there.")
                                     .arg(file_name),
                                 10000);
    });
}

void MainWindow::set_playing(bool playing) { run_->setChecked(playing); }

bool MainWindow::playing() const { return run_->isChecked(); }

void MainWindow::update_title() {
    const QString base = tr("Lenia kernel workbench");
    setWindowTitle(document_name_.isEmpty() ? base : QStringLiteral("%1 — %2").arg(document_name_, base));
}

} // namespace workbench
