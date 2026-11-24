#pragma once

#include "edit_history.hpp"
#include "kernel_editor.hpp"
#include "lenia_view.hpp"
#include "node_editor.hpp"

#include "lenia/node_graph.hpp"
#include "lenia/pattern.hpp"
#include "lenia/project.hpp"
#include "lenia/rule.hpp"
#include "lenia/snapshot.hpp"
#include "lenia/world.hpp"

#include <QList>
#include <QMainWindow>
#include <QPoint>
#include <QString>

#include <array>
#include <functional>
#include <memory>
#include <optional>

class QAction;
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSpinBox;
class QVulkanInstance;

namespace workbench {

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QVulkanInstance* instance);

    LeniaView* view() const { return view_; }

    // The File menu's commands without their dialogs (the menu adds file choosers and message
    // boxes; tests drive these directly). Each returns an empty string on success, otherwise
    // why nothing changed.
    //
    // Experiment (docs/projects.md): replaces the setup and restarts paused at generation 0.
    QString open_experiment_file(const QString& path);
    // Simulation snapshot (docs/snapshots.md): reads and checks the whole file first, then
    // replaces the setup and resumes the stored world, paused at the stored generation.
    QString open_snapshot_file(const QString& path);
    // Captures the GPU world at the next frame together with the setup of that frame, and writes
    // it atomically (QSaveFile). Asynchronous: `done` gets the outcome, after the capture.
    void save_snapshot_file(const QString& path, std::function<void(const QString& error)> done);

    void set_playing(bool playing);
    bool playing() const;
    // Replaces the display graph with an example (the Display graph toolbar's Example → Load).
    void load_display_preset(lenia::DisplayPreset preset);
    // The Density + Nutrient demo (lenia::make_nutrient_trail_project with the bundled Orbium),
    // reseeded and running: Presets → Nutrient trail, or the node graph's Example list.
    void load_nutrient_trail();
    // The two-kernel demo (lenia::make_short_long_project), reseeded and running: Presets →
    // Short / long range, or the node graph's Example list.
    void load_short_long();
    // Reads both fields back and shows their statistics, and the values at (x, y) if given, in
    // the View panel's inspection line (and the status bar for a cell). Asynchronous.
    void inspect_fields(std::optional<QPoint> cell = std::nullopt);
    QString inspection_text() const;

    // Undo history (docs/undo.md) and the parts of the window it covers, for tests.
    EditHistory* history() const { return history_; }
    QAction* undo_action() const { return undo_action_; }
    QAction* redo_action() const { return redo_action_; }
    NodeEditor* node_editor() const { return node_editor_; }
    NodeEditor* display_editor() const { return display_editor_; }
    KernelModel* kernel_model() const { return kernel_; }
    KernelCanvas* kernel_canvas() const { return canvas_; }
    const lenia::Graph& graph() const { return graph_; }
    const lenia::Graph& display_graph() const { return display_graph_; }
    SetupState capture_setup() const;

protected:
    // Keeps Ctrl+Z / Ctrl+Y for the undo history while a spin box has focus (its line edit
    // would take them for its own text undo).
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    enum Seed { SeedRandom, SeedPattern, SeedImpulse, SeedEmpty };

    QWidget* build_kernel_panel();
    QWidget* build_simulation_panel();
    void build_actions();
    void build_menus();

    // Undo history: an undo step's parts applied to the window (graphs, kernel, growth), never
    // touching the world, its generation or play / pause; and the history dropped when another
    // experiment, snapshot or pattern replaces the setup.
    void apply_setup(const SetupParts& parts);
    void reset_history();
    // Records a panel edit; spin box steps share `merge_key` (with panel_session_) so that a
    // field's steps are one undo step until it loses focus.
    void commit_panel_edit(const QString& text, const QString& merge_key = QString());

    // LeniaView's InputsProvider: the graph evaluated at `generation` applied to the Kernel
    // panel's kernel and the Simulation panel's growth, plus the graph's compiled cell rule if
    // Next state is linked. A rule that does not compile stops stepping (no kernel). The
    // selected node, if a Convolve node the rule uses, is the Neighborhood field's preview.
    GenerationInputs inputs_at(std::uint64_t generation);
    void update_preview_context();
    // Status bar note when the graph's rule does not compile.
    void report_rule();

    // Node graph examples: Lenia ones replace the graph only; Game of Life ones load the whole
    // Life experiment (Moore kernel, glider seed, display) and run it.
    void load_graph_preset(lenia::GraphPreset preset);
    void load_life(lenia::GraphPreset preset);

    // Display graph (docs/display_graph.md). LeniaView's DisplayProvider: the compiled program,
    // or null while the View node is unlinked or the graph is too large (the panel display).
    std::shared_ptr<const lenia::DisplayProgram> display_at(std::uint64_t generation);
    void wire_panel_display();
    // The panel's display of the current field, as a lenia::DisplayRange.
    lenia::DisplayRange panel_range() const;
    // After a display graph edit or replacement: redraw, enable the panel's View controls only
    // while they are what is shown, and report a graph that is too large.
    void display_graph_changed();

    void sync_kernel_controls();
    void sync_view_controls();
    void push_growth();
    void push_view();
    void push_brushes();

    void reset_world();
    void seed_impulse();
    void set_running(bool running);
    void step_once();
    // Reloads the bundled Orbium (kernel, growth, cells) and the default wired Lenia graph,
    // reseeds the world from it and runs.
    void restore_orbium();
    void load_pattern();
    // Takes over the pattern's kernel and growth and reseeds the world from its cells. Warns
    // under `title` and returns false if the workbench cannot run it.
    bool apply_pattern(lenia::Pattern p, const QString& title);
    void set_attribution();

    // Experiment files (docs/projects.md). Opening replaces the whole setup and restarts it
    // paused at generation 0; a file that fails to parse or to fit the workbench's controls is
    // rejected before anything changes. Saving goes through QSaveFile, so an existing file is
    // only replaced once the new one is completely written.
    lenia::Project capture_project() const;
    // Why the workbench cannot represent `p` exactly (empty if it can).
    QString check_project(const lenia::Project& p) const;
    // Replaces the setup with `p` and, paused, either reseeds the world (generation 0) or
    // resumes `world` at `generation` (a snapshot's, both fields already checked to be
    // p.world_size square).
    void apply_project(lenia::Project p, const lenia::WorldState* world = nullptr, std::uint64_t generation = 0);
    void open_project();
    bool save_project();
    bool save_project_as();
    bool write_project_file(const QString& path);
    // Snapshots (docs/snapshots.md).
    void open_snapshot();
    void save_snapshot_as();
    QString write_snapshot_file(const QString& path, const lenia::Snapshot& snapshot);
    void update_title();

    LeniaView* view_;
    QString project_path_;  // experiment file for Save; empty until opened or saved
    QString document_name_; // title bar: the experiment or snapshot last opened or saved
    QString snapshot_dir_;  // where the snapshot dialogs start
    QList<QAction*> shortcut_actions_; // forwarded from the Vulkan view's Ctrl key presses
    EditHistory* history_ = nullptr;
    QAction* undo_action_ = nullptr;
    QAction* redo_action_ = nullptr;
    int panel_session_ = 0;
    KernelModel* kernel_;
    std::optional<lenia::Pattern> pattern_;
    std::array<ViewSettings, lenia::kDisplayFieldCount> view_settings_;

    lenia::Graph graph_;
    NodeEditor* node_editor_ = nullptr;
    lenia::GrowthParams growth_;
    std::shared_ptr<const lenia::Grid> static_kernel_; // kernel_->kernel()
    std::shared_ptr<const lenia::Grid> warped_kernel_; // kernel_->build(warped_for_)
    lenia::KernelWarp warped_for_;
    std::shared_ptr<const lenia::RuleProgram> rule_; // last compiled rule, kept while unchanged
    bool life_loaded_ = false; // the Game of Life preset set the world size and display
    bool trail_loaded_ = false; // the nutrient trail demo set the display graph and field
    bool short_long_loaded_ = false; // the two-kernel demo set the kernel, noise seed and display
    bool previewing_ = false; // a Convolve node is selected: its U is the Neighborhood field

    lenia::Graph display_graph_ = lenia::make_display_preset(lenia::DisplayPreset::Panel);
    NodeEditor* display_editor_ = nullptr;
    std::shared_ptr<const lenia::DisplayProgram> display_program_; // last compiled, kept while unchanged

    // Kernel panel.
    QSpinBox* radius_ = nullptr;
    QComboBox* core_ = nullptr;
    QSpinBox* ring_count_ = nullptr;
    QComboBox* normalization_ = nullptr;
    KernelCanvas* canvas_ = nullptr;
    QDoubleSpinBox* kernel_brush_radius_ = nullptr;
    QDoubleSpinBox* kernel_brush_strength_ = nullptr;
    QCheckBox* ring_brush_ = nullptr;
    QLabel* kernel_summary_ = nullptr;

    // Simulation panel.
    QComboBox* growth_function_ = nullptr;
    QDoubleSpinBox* mu_ = nullptr;
    QDoubleSpinBox* sigma_ = nullptr;
    QDoubleSpinBox* dt_ = nullptr;
    QPushButton* run_ = nullptr;
    QSpinBox* step_count_ = nullptr;
    QSpinBox* steps_per_frame_ = nullptr;
    QComboBox* world_size_ = nullptr;
    QComboBox* seed_ = nullptr;
    QSpinBox* random_seed_ = nullptr;
    QComboBox* nutrient_start_ = nullptr; // item index == lenia::NutrientStart
    QDoubleSpinBox* nutrient_level_ = nullptr;
    QSpinBox* world_brush_radius_ = nullptr;
    QDoubleSpinBox* world_brush_value_ = nullptr;
    QComboBox* world_brush_field_ = nullptr; // item index == BrushField
    QLabel* attribution_ = nullptr;
    QComboBox* field_ = nullptr;
    QComboBox* palette_ = nullptr;
    QDoubleSpinBox* lo_ = nullptr;
    QDoubleSpinBox* hi_ = nullptr;
    QLabel* display_note_ = nullptr; // shown while the display graph colors the view
    QLabel* inspection_ = nullptr;   // the last inspect_fields result

    QLabel* generation_ = nullptr;
    QLabel* device_ = nullptr;
};

} // namespace workbench
