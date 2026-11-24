#pragma once

#include "lenia/display_graph.hpp"
#include "lenia/node_graph.hpp"

#include <QPointF>
#include <QWidget>

#include <cstdint>
#include <optional>
#include <vector>

class QComboBox;
class QFormLayout;
class QLabel;
class QPainter;
class QPushButton;

namespace workbench {

// Interactive canvas for a lenia::Graph (owned elsewhere), simulation or display. Nodes show
// their live values at the current generation, and the Oscillator, Kernel, Environment,
// Growth, Life and Simulation nodes draw previews (wave, warped radial profile, environment
// field, growth curve, rule), as do Kernel K (its weights) and Convolve (whether the rule uses
// it) and the display's Color ramp, Color and View nodes.
//
// Mouse: drag from a port to a port of the same type to link (a Value output also links into
// a cell input; an input takes one link, a new one replaces it); drag a linked input away to
// unlink it; drag a node to move it; left drag on empty space or middle drag pans; the wheel
// zooms; right-click adds, deletes, unlinks.
// Keys: Delete / Backspace remove the selected node or link, Home frames all nodes.
class NodeCanvas : public QWidget {
    Q_OBJECT
public:
    explicit NodeCanvas(lenia::Graph* graph, QWidget* parent = nullptr);
    QSize sizeHint() const override { return {900, 330}; }

    void set_generation(std::uint64_t generation);
    // Kernel shape and world size used by the Kernel and Environment previews.
    void set_preview_context(const lenia::KernelShape& shape, int world_width, int world_height);
    // The Simulation panel's growth, which Growth nodes with Source = panel show and use.
    void set_panel_growth(const lenia::GrowthParams& growth);
    // Wires the built-in Lenia update as nodes (lenia::wire_lenia_rule) if Next state is unlinked.
    void wire_lenia_rule();

    // Widget coordinates of a port and of a node's title bar, for driving the canvas with mouse
    // events (tests); a null point if the node does not exist.
    QPointF port_widget_pos(int node, int port, bool output) const;
    QPointF title_widget_pos(int node) const;

    int selected_node() const { return selected_node_; }
    void select_node(int id);
    void add_node(lenia::NodeKind kind); // near the middle of the visible area
    void frame_all();
    // The graph was replaced wholesale (a preset): drops the selection and frames it.
    void graph_replaced();
    // The graph was set back to an earlier state (undo / redo): keeps the view and the
    // selection where they still exist, cancels a drag in progress.
    void graph_restored();

signals:
    void graph_changed(); // nodes or links changed (positions do not count)
    // A gesture that may have changed the graph ended (link drag released, node moved, menu
    // command done): one undo step named `text` (EditHistory::commit, docs/undo.md).
    void edited(const QString& text, const QString& merge_key = QString());
    void selection_changed(int node);
    void message(const QString& text);
    // Display graphs: "Wire the panel display as nodes" on an unlinked View node. The owner
    // knows the panel's field and palette (lenia::wire_panel_display).
    void wire_view_requested();

protected:
    void paintEvent(QPaintEvent*) override;
    void resizeEvent(QResizeEvent*) override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;
    void keyPressEvent(QKeyEvent* e) override;
    void contextMenuEvent(QContextMenuEvent* e) override;

private:
    struct PortRef {
        int node = -1;
        int port = -1;
        bool output = false;
        bool valid() const { return node >= 0; }
        bool operator==(const PortRef&) const = default;
    };

    QPointF to_scene(QPointF widget) const { return (widget - pan_) / zoom_; }
    QRectF node_rect(const lenia::Node& n) const;
    QPointF port_pos(const lenia::Node& n, int port, bool output) const;
    lenia::PortType port_type(const PortRef& p) const;
    bool compatible(const PortRef& a, const PortRef& b) const;
    PortRef port_at(QPointF scene) const;
    int node_at(QPointF scene) const;
    std::optional<lenia::Link> link_at(QPointF scene) const;

    void draw_node(QPainter& p, const lenia::Node& n);
    void draw_oscillator_preview(QPainter& p, const lenia::Node& n, QRectF area);
    void draw_kernel_preview(QPainter& p, const lenia::Node& n, QRectF area);
    void draw_conv_kernel_preview(QPainter& p, const lenia::Node& n, QRectF area);
    void draw_convolve_preview(QPainter& p, const lenia::Node& n, QRectF area);
    void draw_environment_preview(QPainter& p, const lenia::Node& n, QRectF area);
    void draw_growth_preview(QPainter& p, const lenia::Node& n, QRectF area);
    void draw_life_preview(QPainter& p, const lenia::Node& n, QRectF area);
    void draw_output_preview(QPainter& p, QRectF area);
    void draw_ramp_preview(QPainter& p, const lenia::Node& n, QRectF area);
    void draw_color_preview(QPainter& p, const lenia::Node& n, QRectF area);
    void draw_view_preview(QPainter& p, QRectF area);
    QString node_title(const lenia::Node& n) const;
    QString input_text(const lenia::Node& n, int port, bool linked) const;

    void add_node_at(lenia::NodeKind kind, QPointF scene);
    void remove_node(int id);
    void delete_selection();
    void set_selected_link(std::optional<lenia::Link> link);

    lenia::Graph* graph_;
    std::uint64_t generation_ = 0;
    lenia::GrowthParams panel_growth_;
    lenia::KernelShape shape_;
    int world_width_ = 256;
    int world_height_ = 256;

    QPointF pan_{20, 20}; // widget = scene * zoom + pan
    double zoom_ = 1.0;
    bool framed_ = false; // frame_all once the widget has a real size
    int selected_node_ = -1;
    std::optional<lenia::Link> selected_link_;

    enum class Drag { None, Move, Pan, Link } drag_ = Drag::None;
    PortRef drag_from_; // fixed end of the link being dragged
    QPointF drag_pos_;  // scene
    QPointF move_offset_;
    QPointF last_mouse_; // widget
    PortRef hover_port_;
    // Link drags that picked up a linked input: the graph before, and the link picked up, so
    // dropping it back where it was restores the graph exactly (no undo step).
    std::optional<lenia::Graph> link_drag_before_;
    std::optional<lenia::Link> detached_;
};

// NodeCanvas with a toolbar (add nodes, example graphs, frame) and an inspector for the
// selected node's parameters; Color ramp nodes get a stop editor. The toolbar follows the
// graph's domain: simulation graphs offer value and cell rule nodes and the rule examples,
// display graphs value and display nodes and the display examples.
class NodeEditor : public QWidget {
    Q_OBJECT
public:
    explicit NodeEditor(lenia::Graph* graph, QWidget* parent = nullptr);

    NodeCanvas* canvas() const { return canvas_; }
    // Replaces the graph with the preset (graph only; simulation graphs).
    void load_preset(lenia::GraphPreset preset);
    // The selected node's parameter editor for parameter `index`, if shown (tests).
    QWidget* inspector_field(int index) const;

signals:
    void graph_changed();
    // Canvas gestures and inspector edits, as NodeCanvas::edited. Spin box steps on one field
    // share a merge key until the field loses focus, so they are one undo step.
    void edited(const QString& text, const QString& merge_key = QString());
    void message(const QString& text);
    // The Example "Load" button: the owner decides what a preset needs besides the graph (the
    // Game of Life its kernel and seed) and calls load_preset for graph-only ones.
    void preset_requested(lenia::GraphPreset preset);
    // The same for display graphs: the owner supplies the panel display the presets may copy.
    void display_preset_requested(lenia::DisplayPreset preset);
    void wire_view_requested();

private:
    void rebuild_inspector();

    lenia::Graph* graph_;
    NodeCanvas* canvas_;
    QComboBox* presets_;
    QLabel* inspector_title_;
    QLabel* inspector_help_;
    QWidget* inspector_fields_;
    QFormLayout* inspector_form_;
    QPushButton* delete_node_;
    std::vector<QWidget*> inspector_fields_by_param_; // index: parameter; null if not shown
    int inspector_session_ = 0; // part of the spin boxes' merge keys; bumped when one loses focus
};

} // namespace workbench
