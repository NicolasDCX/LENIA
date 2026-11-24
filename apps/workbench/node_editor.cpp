#include "node_editor.hpp"

#include "palettes.hpp"

#include "lenia/rule.hpp"
#include "lenia/step.hpp"

#include <QAction>
#include <QColorDialog>
#include <QComboBox>
#include <QContextMenuEvent>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QImage>
#include <QKeyEvent>
#include <QLabel>
#include <QMenu>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPushButton>
#include <QScrollArea>
#include <QSplitter>
#include <QVBoxLayout>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace workbench {

namespace {

using lenia::NodeKind;
using lenia::PortType;

constexpr double kNodeWidth = 200.0;
constexpr double kHeader = 24.0;
constexpr double kRow = 21.0;
constexpr double kPortRadius = 5.0;
constexpr double kPickRadius = 10.0; // widget pixels
constexpr double kGrid = 24.0;

// Canvas colors, matched to the dark Fusion palette set in main.cpp.
const QColor kBackground(28, 29, 33);
const QColor kGridMinor(38, 40, 45);
const QColor kGridMajor(48, 51, 57);
const QColor kBody(48, 50, 56);
const QColor kBorder(70, 73, 80);
const QColor kSelected(240, 240, 240);
const QColor kText(225, 225, 228);
const QColor kDim(140, 143, 150);

QColor type_color(PortType t) {
    switch (t) {
    case PortType::Value: return QColor(232, 168, 64);
    case PortType::Kernel: return QColor(72, 196, 200);
    case PortType::Environment: return QColor(120, 200, 100);
    case PortType::Cell: return QColor(228, 108, 170);
    case PortType::Color: return QColor(235, 214, 90);
    }
    return kText;
}

// Add menus put these in the first ("value / modulation") menu, the rest in the second (cell
// rule nodes, or display nodes).
bool is_value_node(NodeKind k) {
    return k == NodeKind::Time || k == NodeKind::Oscillator || k == NodeKind::Constant || k == NodeKind::Kernel ||
           k == NodeKind::Environment;
}

QColor kind_color(NodeKind k) {
    switch (k) {
    case NodeKind::Time:
    case NodeKind::Oscillator:
    case NodeKind::Constant: return type_color(PortType::Value);
    case NodeKind::Kernel: return type_color(PortType::Kernel);
    case NodeKind::Environment: return type_color(PortType::Environment);
    case NodeKind::Output:
    case NodeKind::View: return QColor(170, 130, 230);
    case NodeKind::ColorRamp:
    case NodeKind::Color:
    case NodeKind::Mix: return type_color(PortType::Color);
    default: return type_color(PortType::Cell);
    }
}

double preview_height(NodeKind k) {
    switch (k) {
    case NodeKind::Oscillator: return 44.0;
    case NodeKind::Kernel: return 64.0;
    case NodeKind::Environment: return 72.0;
    case NodeKind::Growth: return 48.0;
    case NodeKind::Life: return 22.0;
    case NodeKind::Output: return 62.0;
    case NodeKind::ColorRamp: return 20.0;
    case NodeKind::Color: return 16.0;
    case NodeKind::View: return 36.0;
    default: return 0.0;
    }
}

QColor to_qcolor(const std::array<float, 3>& rgb) {
    return QColor::fromRgbF(std::clamp(rgb[0], 0.0f, 1.0f), std::clamp(rgb[1], 0.0f, 1.0f),
                            std::clamp(rgb[2], 0.0f, 1.0f));
}

std::vector<lenia::RampStop> sorted(std::vector<lenia::RampStop> stops) {
    std::stable_sort(stops.begin(), stops.end(), [](const lenia::RampStop& a, const lenia::RampStop& b) {
        return a.position < b.position;
    });
    return stops;
}

// Horizontal gradient of a ramp's stops filling `area`.
void paint_ramp(QPainter& p, const std::vector<lenia::RampStop>& stops, QRectF area) {
    const std::vector<lenia::RampStop> s = sorted(stops);
    const int w = std::max(2, static_cast<int>(area.width()));
    QImage image(w, 1, QImage::Format_RGB32);
    for (int x = 0; x < w; ++x) {
        image.setPixel(x, 0, to_qcolor(lenia::sample_ramp(s, (x + 0.5f) / static_cast<float>(w))).rgb());
    }
    p.drawImage(area, image);
}

// Stop editor for a Color ramp node: the gradient with a handle per stop. Drag a handle to move
// the stop, double-click it to pick its color, right-click it to remove it (at least two stay);
// double-click the gradient to add a stop with the color already there.
class RampEditor : public QWidget {
public:
    RampEditor(lenia::Graph* graph, int node, std::function<void()> changed)
        : graph_(graph), node_(node), changed_(std::move(changed)) {
        setMinimumHeight(56);
        setMouseTracking(true);
        setToolTip(QObject::tr("Drag a stop to move it · double-click a stop to recolor it · double-click the bar "
                               "to add a stop · right-click a stop to remove it"));
    }

    QSize sizeHint() const override { return {240, 56}; }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        const std::vector<lenia::RampStop> s = stops();
        paint_ramp(p, s, bar());
        p.setPen(QPen(kBorder, 1.0));
        p.setBrush(Qt::NoBrush);
        p.drawRect(bar());
        for (int j = 0; j < static_cast<int>(s.size()); ++j) {
            const QPointF tip(x_of(s[j].position), bar().bottom() + 1);
            QPolygonF handle;
            handle << tip << tip + QPointF(-kHandle, kHandle * 1.6) << tip + QPointF(kHandle, kHandle * 1.6);
            p.setPen(QPen(j == active_ ? kSelected : kDim, j == active_ ? 2.0 : 1.0));
            p.setBrush(to_qcolor(s[j].rgb));
            p.drawPolygon(handle);
        }
    }

    void mousePressEvent(QMouseEvent* e) override {
        const int j = stop_at(e->position());
        active_ = j;
        dragging_ = j >= 0 && e->button() == Qt::LeftButton;
        if (j >= 0 && e->button() == Qt::RightButton) {
            std::vector<lenia::RampStop> s = stops();
            if (s.size() > 2) {
                s.erase(s.begin() + j);
                active_ = -1;
                write(s);
            }
        }
        update();
    }

    void mouseMoveEvent(QMouseEvent* e) override {
        if (!dragging_ || active_ < 0) return;
        std::vector<lenia::RampStop> s = stops();
        if (active_ >= static_cast<int>(s.size())) return;
        s[static_cast<std::size_t>(active_)].position = t_of(e->position().x());
        write(s);
    }

    void mouseReleaseEvent(QMouseEvent*) override { dragging_ = false; }

    void mouseDoubleClickEvent(QMouseEvent* e) override {
        if (e->button() != Qt::LeftButton) return;
        std::vector<lenia::RampStop> s = stops();
        const int j = stop_at(e->position());
        if (j >= 0) {
            const QColor c = QColorDialog::getColor(to_qcolor(s[static_cast<std::size_t>(j)].rgb), this,
                                                    QObject::tr("Stop color"));
            if (!c.isValid()) return;
            s[static_cast<std::size_t>(j)].rgb = {static_cast<float>(c.redF()), static_cast<float>(c.greenF()),
                                                  static_cast<float>(c.blueF())};
            write(s);
        } else if (s.size() < static_cast<std::size_t>(lenia::ramp_node::kMaxStops)) {
            const float t = t_of(e->position().x());
            s.push_back({t, lenia::sample_ramp(sorted(s), t)});
            active_ = static_cast<int>(s.size()) - 1;
            write(s);
        }
    }

private:
    static constexpr double kHandle = 6.0;

    QRectF bar() const { return QRectF(kHandle + 2, 4, width() - 2 * (kHandle + 2), height() - 4 - kHandle * 1.6 - 6); }
    double x_of(float t) const { return bar().left() + t * bar().width(); }
    float t_of(double x) const {
        return static_cast<float>(std::clamp((x - bar().left()) / std::max(bar().width(), 1.0), 0.0, 1.0));
    }

    std::vector<lenia::RampStop> stops() const {
        const lenia::Node* n = graph_->node(node_);
        return n ? lenia::ramp_stops(*n) : std::vector<lenia::RampStop>{};
    }

    // Nearest handle within reach, the active one first so overlapping stops stay draggable.
    int stop_at(QPointF pos) const {
        const std::vector<lenia::RampStop> s = stops();
        if (pos.y() < bar().top()) return -1;
        int best = -1;
        double best_d = kHandle + 2;
        for (int j = 0; j < static_cast<int>(s.size()); ++j) {
            const double d = std::fabs(x_of(s[static_cast<std::size_t>(j)].position) - pos.x());
            if (d < best_d || (d <= best_d && j == active_)) {
                best = j;
                best_d = d;
            }
        }
        // On the bar itself only the handles' own column counts, so double-clicks can add.
        if (best >= 0 && pos.y() < bar().bottom() && best_d > 2.0) return -1;
        return best;
    }

    void write(const std::vector<lenia::RampStop>& s) {
        lenia::set_ramp_stops(*graph_, node_, s);
        update();
        changed_();
    }

    lenia::Graph* graph_;
    int node_;
    std::function<void()> changed_;
    int active_ = -1;
    bool dragging_ = false;
};

QString number(double v) {
    return std::fabs(v) >= 100.0 ? QString::number(v, 'f', 1) : QString::number(v, 'f', 3);
}

QPainterPath link_path(QPointF a, QPointF b) {
    const double dx = std::max(40.0, std::fabs(b.x() - a.x()) / 2.0);
    QPainterPath path(a);
    path.cubicTo(a + QPointF(dx, 0), b - QPointF(dx, 0), b);
    return path;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// NodeCanvas

NodeCanvas::NodeCanvas(lenia::Graph* graph, QWidget* parent) : QWidget(parent), graph_(graph) {
    setMinimumHeight(180);
    setMouseTracking(true);
    setFocusPolicy(Qt::ClickFocus);
    setAttribute(Qt::WA_OpaquePaintEvent);
    setToolTip(tr("Drag between ports to link · drag a linked input away to unlink · right-click for "
                  "more · Del removes the selection · Home frames all"));
}

void NodeCanvas::set_generation(std::uint64_t generation) {
    if (generation_ == generation) return;
    generation_ = generation;
    if (isVisible()) update();
}

void NodeCanvas::set_preview_context(const lenia::KernelShape& shape, int world_width, int world_height) {
    shape_ = shape;
    world_width_ = std::max(world_width, 1);
    world_height_ = std::max(world_height, 1);
    update();
}

void NodeCanvas::set_panel_growth(const lenia::GrowthParams& growth) {
    panel_growth_ = growth;
    update();
}

void NodeCanvas::wire_lenia_rule() {
    if (!lenia::wire_lenia_rule(*graph_)) {
        emit message(tr("Next state is already linked"));
        return;
    }
    emit graph_changed();
    frame_all();
    emit message(tr("Wired the built-in Lenia update as nodes: same result, now editable"));
}

void NodeCanvas::select_node(int id) {
    if (id == selected_node_) return;
    selected_node_ = id;
    if (id >= 0) selected_link_.reset();
    emit selection_changed(id);
    update();
}

void NodeCanvas::set_selected_link(std::optional<lenia::Link> link) {
    selected_link_ = link;
    if (link) select_node(-1);
    update();
}

QRectF NodeCanvas::node_rect(const lenia::Node& n) const {
    const lenia::NodeSpec& s = lenia::node_spec(n.kind);
    const double rows = static_cast<double>(s.inputs.size() + s.outputs.size());
    const double preview = preview_height(n.kind);
    return QRectF(n.x, n.y, kNodeWidth, kHeader + rows * kRow + (preview > 0 ? preview + 6.0 : 0.0) + 6.0);
}

QPointF NodeCanvas::port_pos(const lenia::Node& n, int port, bool output) const {
    const lenia::NodeSpec& s = lenia::node_spec(n.kind);
    const int row = output ? static_cast<int>(s.inputs.size()) + port : port;
    return QPointF(output ? n.x + kNodeWidth : n.x, n.y + kHeader + (row + 0.5) * kRow + 3.0);
}

PortType NodeCanvas::port_type(const PortRef& p) const {
    const lenia::Node* n = graph_->node(p.node);
    if (!n) return PortType::Value;
    const lenia::NodeSpec& s = lenia::node_spec(n->kind);
    return (p.output ? s.outputs : s.inputs)[static_cast<std::size_t>(p.port)].type;
}

bool NodeCanvas::compatible(const PortRef& a, const PortRef& b) const {
    if (!a.valid() || !b.valid() || a.output == b.output || a.node == b.node) return false;
    const PortRef& out = a.output ? a : b;
    const PortRef& in = a.output ? b : a;
    return lenia::can_link(port_type(out), port_type(in));
}

NodeCanvas::PortRef NodeCanvas::port_at(QPointF scene) const {
    const double pick = kPickRadius / zoom_;
    PortRef best;
    double best_d = pick;
    for (const lenia::Node& n : graph_->nodes()) {
        const lenia::NodeSpec& s = lenia::node_spec(n.kind);
        for (int out = 0; out < 2; ++out) {
            const int count = static_cast<int>(out ? s.outputs.size() : s.inputs.size());
            for (int i = 0; i < count; ++i) {
                const QPointF d = port_pos(n, i, out != 0) - scene;
                const double dist = std::hypot(d.x(), d.y());
                if (dist <= best_d) {
                    best_d = dist;
                    best = {n.id, i, out != 0};
                }
            }
        }
    }
    return best;
}

int NodeCanvas::node_at(QPointF scene) const {
    const std::vector<lenia::Node>& nodes = graph_->nodes();
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) { // topmost (drawn last) first
        if (node_rect(*it).contains(scene)) return it->id;
    }
    return -1;
}

std::optional<lenia::Link> NodeCanvas::link_at(QPointF scene) const {
    QPainterPathStroker stroker;
    stroker.setWidth(8.0 / zoom_);
    for (const lenia::Link& l : graph_->links()) {
        const lenia::Node* a = graph_->node(l.from_node);
        const lenia::Node* b = graph_->node(l.to_node);
        if (!a || !b) continue;
        if (stroker.createStroke(link_path(port_pos(*a, l.from_port, true), port_pos(*b, l.to_port, false)))
                .contains(scene)) {
            return l;
        }
    }
    return std::nullopt;
}

void NodeCanvas::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.fillRect(rect(), kBackground);
    p.setRenderHint(QPainter::Antialiasing);

    // Grid, in scene units.
    const QPointF tl = to_scene(QPointF(0, 0));
    const QPointF br = to_scene(QPointF(width(), height()));
    p.save();
    p.translate(pan_);
    p.scale(zoom_, zoom_);
    for (auto i = static_cast<long long>(std::floor(tl.x() / kGrid)); i * kGrid < br.x(); ++i) {
        p.setPen(QPen(i % 5 == 0 ? kGridMajor : kGridMinor, 0));
        p.drawLine(QPointF(i * kGrid, tl.y()), QPointF(i * kGrid, br.y()));
    }
    for (auto i = static_cast<long long>(std::floor(tl.y() / kGrid)); i * kGrid < br.y(); ++i) {
        p.setPen(QPen(i % 5 == 0 ? kGridMajor : kGridMinor, 0));
        p.drawLine(QPointF(tl.x(), i * kGrid), QPointF(br.x(), i * kGrid));
    }

    // Links under the nodes.
    p.setBrush(Qt::NoBrush);
    for (const lenia::Link& l : graph_->links()) {
        const lenia::Node* a = graph_->node(l.from_node);
        const lenia::Node* b = graph_->node(l.to_node);
        if (!a || !b) continue;
        const bool selected = selected_link_ && *selected_link_ == l;
        const QColor c = type_color(lenia::node_spec(a->kind).outputs[static_cast<std::size_t>(l.from_port)].type);
        p.setPen(QPen(selected ? kSelected : c, selected ? 3.5 : 2.2));
        p.drawPath(link_path(port_pos(*a, l.from_port, true), port_pos(*b, l.to_port, false)));
    }

    // The link being dragged, snapped to a compatible port under the cursor.
    if (drag_ == Drag::Link) {
        const lenia::Node* n = graph_->node(drag_from_.node);
        if (n) {
            const QPointF fixed = port_pos(*n, drag_from_.port, drag_from_.output);
            QPointF free = drag_pos_;
            if (compatible(drag_from_, hover_port_)) {
                free = port_pos(*graph_->node(hover_port_.node), hover_port_.port, hover_port_.output);
            }
            p.setPen(QPen(type_color(port_type(drag_from_)), 2.0, Qt::DashLine));
            p.drawPath(drag_from_.output ? link_path(fixed, free) : link_path(free, fixed));
        }
    }

    for (const lenia::Node& n : graph_->nodes()) draw_node(p, n);
    p.restore();

    // Usage hint, in widget pixels.
    p.setPen(kDim);
    p.drawText(QRectF(8, height() - 22, width() - 16, 18), Qt::AlignLeft | Qt::AlignVCenter,
               tr("Drag output ● → input ● to link · drag a linked input away to unlink · right-click: "
                  "add / delete · Del · wheel zoom · drag empty space to pan · Home frames all"));
}

void NodeCanvas::draw_node(QPainter& p, const lenia::Node& n) {
    const lenia::NodeSpec& s = lenia::node_spec(n.kind);
    const QRectF r = node_rect(n);
    const bool selected = n.id == selected_node_;

    p.setPen(Qt::NoPen);
    p.setBrush(QColor(0, 0, 0, 90));
    p.drawRoundedRect(r.translated(3, 4), 6, 6);
    p.setBrush(kBody);
    p.drawRoundedRect(r, 6, 6);
    const QRectF header(r.left(), r.top(), r.width(), kHeader);
    p.setBrush(kind_color(n.kind).darker(230));
    p.drawRoundedRect(header, 6, 6);
    p.drawRect(header.adjusted(0, kHeader / 2, 0, 0));
    p.setBrush(Qt::NoBrush);
    p.setPen(QPen(selected ? kSelected : kBorder, selected ? 2.0 : 1.0));
    p.drawRoundedRect(r, 6, 6);

    QFont bold = font();
    bold.setBold(true);
    p.setFont(bold);
    p.setPen(kind_color(n.kind).lighter(130));
    p.drawText(header.adjusted(10, 0, -8, 0), Qt::AlignLeft | Qt::AlignVCenter, node_title(n));
    p.setFont(font());

    const auto row_rect = [&](int row) {
        return QRectF(r.left() + 12, r.top() + kHeader + 3 + row * kRow, r.width() - 24, kRow);
    };

    // Inputs: name, then the value the node sees (accent when it comes from a link).
    for (int i = 0; i < static_cast<int>(s.inputs.size()); ++i) {
        const lenia::PortSpec& port = s.inputs[static_cast<std::size_t>(i)];
        const bool linked = graph_->link_into(n.id, i) != nullptr;
        const QRectF rr = row_rect(i);
        p.setPen(kText);
        p.drawText(rr, Qt::AlignLeft | Qt::AlignVCenter, QString::fromStdString(port.name));
        p.setPen(linked ? type_color(port.type) : kDim);
        p.drawText(rr, Qt::AlignRight | Qt::AlignVCenter, input_text(n, i, linked));

        const QPointF pos = port_pos(n, i, false);
        const PortRef ref{n.id, i, false};
        const bool hot = hover_port_ == ref || (drag_ == Drag::Link && compatible(drag_from_, ref));
        p.setPen(QPen(type_color(port.type), hot ? 2.5 : 1.5));
        p.setBrush(linked ? type_color(port.type) : kBody);
        p.drawEllipse(pos, hot ? kPortRadius + 1.5 : kPortRadius, hot ? kPortRadius + 1.5 : kPortRadius);
        p.setBrush(Qt::NoBrush);
    }

    // Outputs: value (Value outputs) and name, right-aligned.
    for (int j = 0; j < static_cast<int>(s.outputs.size()); ++j) {
        const lenia::PortSpec& port = s.outputs[static_cast<std::size_t>(j)];
        const QRectF rr = row_rect(static_cast<int>(s.inputs.size()) + j);
        p.setPen(kText);
        p.drawText(rr, Qt::AlignRight | Qt::AlignVCenter, QString::fromStdString(port.name));
        if (port.type == PortType::Value) {
            p.setPen(type_color(port.type));
            p.drawText(rr, Qt::AlignLeft | Qt::AlignVCenter,
                       number(lenia::output_value(*graph_, n.id, j, generation_, panel_growth_)));
        } else if (port.type == PortType::Cell) {
            p.setPen(kDim);
            p.drawText(rr, Qt::AlignLeft | Qt::AlignVCenter, tr("per cell"));
        }
        bool linked = false;
        for (const lenia::Link& l : graph_->links()) linked = linked || (l.from_node == n.id && l.from_port == j);
        const QPointF pos = port_pos(n, j, true);
        const PortRef ref{n.id, j, true};
        const bool hot = hover_port_ == ref || (drag_ == Drag::Link && compatible(drag_from_, ref));
        p.setPen(QPen(type_color(port.type), hot ? 2.5 : 1.5));
        p.setBrush(linked ? type_color(port.type) : kBody);
        p.drawEllipse(pos, hot ? kPortRadius + 1.5 : kPortRadius, hot ? kPortRadius + 1.5 : kPortRadius);
        p.setBrush(Qt::NoBrush);
    }

    const double ph = preview_height(n.kind);
    if (ph > 0) {
        const double top = r.top() + kHeader + 3 + static_cast<double>(s.inputs.size() + s.outputs.size()) * kRow + 3;
        const QRectF area(r.left() + 10, top, r.width() - 20, ph);
        p.setPen(Qt::NoPen);
        p.setBrush(kBackground);
        p.drawRoundedRect(area, 3, 3);
        p.setBrush(Qt::NoBrush);
        p.save();
        p.setClipRect(area);
        switch (n.kind) {
        case NodeKind::Oscillator: draw_oscillator_preview(p, n, area); break;
        case NodeKind::Kernel: draw_kernel_preview(p, n, area); break;
        case NodeKind::Environment: draw_environment_preview(p, n, area); break;
        case NodeKind::Growth: draw_growth_preview(p, n, area); break;
        case NodeKind::Life: draw_life_preview(p, n, area); break;
        case NodeKind::Output: draw_output_preview(p, area); break;
        case NodeKind::ColorRamp: draw_ramp_preview(p, n, area); break;
        case NodeKind::Color: draw_color_preview(p, n, area); break;
        case NodeKind::View: draw_view_preview(p, area); break;
        default: break;
        }
        p.restore();
    }
}

void NodeCanvas::draw_oscillator_preview(QPainter& p, const lenia::Node& n, QRectF area) {
    using namespace lenia::oscillator_node;
    const auto wave = static_cast<lenia::Waveform>(static_cast<int>(n.params[Wave]));
    const QRectF a = area.adjusted(4, 4, -4, -4);
    QPainterPath path;
    const int samples = 96;
    for (int i = 0; i <= samples; ++i) {
        const double x = static_cast<double>(i) / samples;
        const QPointF pt(a.left() + x * a.width(), a.center().y() - lenia::waveform(wave, x) * a.height() / 2);
        if (i == 0) path.moveTo(pt);
        else path.lineTo(pt);
    }
    p.setPen(QPen(kGridMajor, 1.0));
    p.drawLine(QPointF(a.left(), a.center().y()), QPointF(a.right(), a.center().y()));
    p.setPen(QPen(type_color(PortType::Value), 1.5));
    p.drawPath(path);
    // Where in its cycle the oscillator is now.
    const double t = lenia::input_value(*graph_, n.id, TimeIn, generation_);
    double phase = t / std::max(static_cast<double>(n.params[Period]), 1e-6) + n.params[Phase];
    phase -= std::floor(phase);
    const QPointF now(a.left() + phase * a.width(), a.center().y() - lenia::waveform(wave, phase) * a.height() / 2);
    p.setPen(QPen(kSelected, 1.0, Qt::DotLine));
    p.drawLine(QPointF(now.x(), area.top()), QPointF(now.x(), area.bottom()));
    p.setBrush(kSelected);
    p.drawEllipse(now, 3, 3);
}

void NodeCanvas::draw_kernel_preview(QPainter& p, const lenia::Node& n, QRectF area) {
    if (shape_.rings.empty() || shape_.radius < 1) return;
    const lenia::KernelWarp warp = lenia::node_kernel(*graph_, n.id, generation_);
    const double re = lenia::warped_radius(shape_, warp);
    // Fixed distance axis (cells) so radius changes show as the curve moving.
    const double d_max = std::max(shape_.radius * 1.6, re * 1.05);
    const bool negative = std::any_of(shape_.rings.begin(), shape_.rings.end(), [](float w) { return w < 0; });
    const double lo = negative ? -1.0 : -0.08;
    const double hi = 1.08;
    const QRectF a = area.adjusted(4, 4, -4, -4);
    const auto to_y = [&](double v) { return a.bottom() - (v - lo) / (hi - lo) * a.height(); };

    const auto curve = [&](const lenia::KernelWarp& w, double radius) {
        QPainterPath path;
        const int samples = 120;
        for (int i = 0; i <= samples; ++i) {
            const double d = d_max * i / samples;
            const float k = lenia::kernel_shell(shape_, w, static_cast<float>(d / radius));
            const QPointF pt(a.left() + d / d_max * a.width(), to_y(k));
            if (i == 0) path.moveTo(pt);
            else path.lineTo(pt);
        }
        return path;
    };
    p.setPen(QPen(kGridMajor, 1.0));
    p.drawLine(QPointF(a.left(), to_y(0)), QPointF(a.right(), to_y(0)));
    p.setPen(QPen(kDim, 1.0, Qt::DashLine));
    p.drawPath(curve({}, shape_.radius)); // static kernel, for comparison
    p.setPen(QPen(type_color(PortType::Kernel), 1.8));
    p.drawPath(curve(warp, re));
    p.setPen(kDim);
    p.drawText(area.adjusted(4, 2, -4, -2), Qt::AlignRight | Qt::AlignTop,
               tr("Re %1 (R %2)").arg(re, 0, 'f', 1).arg(shape_.radius));
}

void NodeCanvas::draw_environment_preview(QPainter& p, const lenia::Node& n, QRectF area) {
    const lenia::Environment env = lenia::node_environment(*graph_, n.id, generation_);
    // Thumbnail of strength-signed E over the whole world: red boosts growth, blue suppresses.
    const double aspect = static_cast<double>(world_width_) / world_height_;
    QRectF img_rect(0, 0, area.height() * aspect, area.height());
    if (img_rect.width() > area.width() * 0.6) img_rect = QRectF(0, 0, area.width() * 0.6, area.width() * 0.6 / aspect);
    img_rect.moveCenter(QPointF(area.left() + 4 + img_rect.width() / 2, area.center().y()));
    const int iw = std::max(8, static_cast<int>(img_rect.width() / 2));
    const int ih = std::max(8, static_cast<int>(img_rect.height() / 2));
    QImage image(iw, ih, QImage::Format_RGB32);
    const float sign = env.strength < 0 ? -1.0f : 1.0f;
    for (int y = 0; y < ih; ++y) {
        for (int x = 0; x < iw; ++x) {
            const auto wx = static_cast<std::size_t>((x + 0.5) * world_width_ / iw);
            const auto wy = static_cast<std::size_t>((y + 0.5) * world_height_ / ih);
            const float e = sign * lenia::environment_field(env, wx, wy, static_cast<std::size_t>(world_width_),
                                                            static_cast<std::size_t>(world_height_));
            image.setPixel(x, y, palette_color(RedBlue, (e + 1.0f) / 2.0f).rgb());
        }
    }
    p.setOpacity(env.strength == 0.0f ? 0.3 : 1.0);
    p.drawImage(img_rect, image);
    p.setOpacity(1.0);
    p.setPen(kText);
    const QRectF text(img_rect.right() + 8, area.top() + 2, area.right() - img_rect.right() - 10, area.height() - 4);
    const lenia::ParamSpec& pattern = lenia::node_spec(NodeKind::Environment).params[lenia::environment_node::Pattern];
    p.drawText(text, Qt::AlignLeft | Qt::AlignTop | Qt::TextWordWrap,
               tr("%1\n×%2\n%3")
                   .arg(QString::fromStdString(pattern.choices[static_cast<std::size_t>(env.pattern)]))
                   .arg(env.strength, 0, 'f', 4)
                   .arg(env.strength == 0.0f ? tr("off") : tr("active")));
}

void NodeCanvas::draw_growth_preview(QPainter& p, const lenia::Node& n, QRectF area) {
    using namespace lenia::growth_node;
    const lenia::GrowthParams g = lenia::node_growth(*graph_, n.id, panel_growth_);
    const bool per_cell = graph_->link_into(n.id, MuIn) || graph_->link_into(n.id, SigmaIn);
    const QRectF a = area.adjusted(4, 4, -4, -4);
    // U over mu ± 5 sigma (the polynomial is nonzero within ± 3 sigma), G in [-1, 1].
    const double lo = g.mu - 5.0 * g.sigma;
    const double hi = g.mu + 5.0 * g.sigma;
    QPainterPath path;
    const int samples = 96;
    for (int i = 0; i <= samples; ++i) {
        const double u = lo + (hi - lo) * i / samples;
        const double v = lenia::growth(g, static_cast<float>(u));
        const QPointF pt(a.left() + a.width() * i / samples, a.center().y() - v * a.height() / 2);
        if (i == 0) path.moveTo(pt);
        else path.lineTo(pt);
    }
    p.setPen(QPen(kGridMajor, 1.0));
    p.drawLine(QPointF(a.left(), a.center().y()), QPointF(a.right(), a.center().y()));
    p.setPen(QPen(type_color(PortType::Cell), 1.5));
    p.drawPath(path);
    p.setPen(kDim);
    p.drawText(area.adjusted(4, 1, -4, -1), Qt::AlignRight | Qt::AlignTop,
               per_cell ? tr("μ/σ per cell") : tr("μ %1 σ %2").arg(g.mu, 0, 'f', 3).arg(g.sigma, 0, 'f', 4));
    p.drawText(area.adjusted(4, 1, -4, -1), Qt::AlignLeft | Qt::AlignBottom,
               g.function == lenia::GrowthFunction::Polynomial ? tr("polynomial") : tr("gaussian"));
}

void NodeCanvas::draw_life_preview(QPainter& p, const lenia::Node& n, QRectF area) {
    using namespace lenia::life_node;
    const int birth = static_cast<int>(n.params[Birth]);
    const int survive = static_cast<int>(n.params[Survive]);
    QString name = QString::fromStdString(lenia::life_rule_name(birth, survive));
    if (birth == 8 && survive == 12) name += tr("  (Conway)");
    if (birth == 72 && survive == 12) name += tr("  (HighLife)");
    p.setPen(kText);
    p.drawText(area.adjusted(6, 0, -6, 0), Qt::AlignLeft | Qt::AlignVCenter, name);
}

void NodeCanvas::draw_ramp_preview(QPainter& p, const lenia::Node& n, QRectF area) {
    paint_ramp(p, lenia::ramp_stops(n), area.adjusted(3, 3, -3, -3));
}

void NodeCanvas::draw_color_preview(QPainter& p, const lenia::Node& n, QRectF area) {
    using namespace lenia::color_node;
    p.fillRect(area.adjusted(3, 3, -3, -3), to_qcolor({n.params[Red], n.params[Green], n.params[Blue]}));
}

void NodeCanvas::draw_view_preview(QPainter& p, QRectF area) {
    const lenia::DisplayCompile c = lenia::compile_display(*graph_, generation_);
    QString text = tr("Unlinked: the Simulation panel's\nfield and palette");
    bool error = false;
    if (c.linked) {
        error = !c.error.empty();
        text = error ? tr("Too large: showing the panel\ndisplay instead")
                     : tr("Display graph: %1 ops, %2 stops\nper pixel, running or paused")
                           .arg(c.program.ops.size())
                           .arg(c.program.stops.size());
    }
    p.setPen(error ? QColor(255, 110, 110) : kText);
    p.drawText(area.adjusted(6, 2, -6, -2), Qt::AlignLeft | Qt::AlignVCenter, text);
}

QString NodeCanvas::node_title(const lenia::Node& n) const {
    const lenia::NodeSpec& s = lenia::node_spec(n.kind);
    const QString name = QString::fromStdString(s.name);
    switch (n.kind) {
    case NodeKind::Compare: {
        const auto& ops = s.params[lenia::compare_node::Op].choices;
        return tr("%1   %2").arg(name, QString::fromStdString(ops[static_cast<std::size_t>(n.params[lenia::compare_node::Op])]));
    }
    case NodeKind::Growth:
        return static_cast<int>(n.params[lenia::growth_node::Source]) == lenia::growth_node::Panel
                   ? tr("%1 · panel").arg(name)
                   : name;
    default:
        return name;
    }
}

QString NodeCanvas::input_text(const lenia::Node& n, int port, bool linked) const {
    const lenia::PortSpec& spec = lenia::node_spec(n.kind).inputs[static_cast<std::size_t>(port)];
    switch (spec.type) {
    case PortType::Value:
        if (!linked && spec.param < 0) return tr("gen %1").arg(generation_);
        return number(lenia::input_value(*graph_, n.id, port, generation_, panel_growth_));
    case PortType::Cell: {
        if (n.kind == NodeKind::Output) return linked ? tr("rule") : tr("built-in Lenia");
        if (const lenia::Link* l = graph_->link_into(n.id, port)) {
            const lenia::Node* from = graph_->node(l->from_node);
            const PortType from_type = lenia::node_spec(from->kind).outputs[static_cast<std::size_t>(l->from_port)].type;
            // A Value link is the same in every cell, so it has one value to show.
            if (from_type == PortType::Value) {
                return number(lenia::output_value(*graph_, l->from_node, l->from_port, generation_, panel_growth_));
            }
            return tr("linked");
        }
        if (spec.param < 0) {
            const bool density = (n.kind == NodeKind::Life && port == lenia::life_node::StateIn) ||
                                 n.kind == NodeKind::ColorRamp;
            return density ? tr("= A") : tr("= U");
        }
        if (n.kind == NodeKind::Growth) {
            const lenia::GrowthParams g = lenia::node_growth(*graph_, n.id, panel_growth_);
            return number(port == lenia::growth_node::MuIn ? g.mu : g.sigma);
        }
        return number(n.params[static_cast<std::size_t>(spec.param)]);
    }
    case PortType::Kernel:
        return linked ? tr("linked") : tr("static (panel)");
    case PortType::Environment:
        if (n.kind == NodeKind::Output && linked && graph_->link_into(n.id, lenia::output_node::NextIn)) {
            return tr("unused by rule");
        }
        return linked ? tr("linked") : tr("none");
    case PortType::Color:
        if (linked) return tr("linked");
        return n.kind == NodeKind::View ? tr("panel display") : tr("black");
    }
    return {};
}

void NodeCanvas::draw_output_preview(QPainter& p, QRectF area) {
    const lenia::GraphFrame f = lenia::evaluate(*graph_, generation_);
    QString kernel = tr("Kernel: static (Kernel panel)");
    if (f.kernel_connected && !f.kernel.neutral()) {
        kernel = tr("Kernel: Re ×%1, peak %2, sharp %3")
                     .arg(1.0 + f.kernel.radius_delta, 0, 'f', 3)
                     .arg(f.kernel.peak_shift, 0, 'f', 2)
                     .arg(f.kernel.sharpness, 0, 'f', 2);
    } else if (f.kernel_connected) {
        kernel = tr("Kernel: linked, neutral = static");
    }
    QString rule = tr("Rule: built-in Lenia (panel growth)");
    QString env = f.environment_connected && f.environment.strength != 0.0f
                      ? tr("Environment: ×%1").arg(f.environment.strength, 0, 'f', 4)
                      : tr("Environment: off");
    bool error = false;
    if (f.rule_connected) {
        const lenia::RuleCompile c = lenia::compile_rule(*graph_, generation_, panel_growth_);
        error = !c.error.empty();
        rule = error ? tr("Rule: too large, not stepping") : tr("Rule: wired, %1 ops per cell").arg(c.program.ops.size());
        env = tr("Environment: via Field outputs");
    }
    const QRectF text = area.adjusted(6, 3, -6, -3);
    const double line = text.height() / 4.0;
    const QString lines[4] = {tr("gen %1").arg(generation_), kernel, rule, env};
    for (int i = 0; i < 4; ++i) {
        p.setPen(i == 2 && error ? QColor(255, 110, 110) : kText);
        p.drawText(QRectF(text.left(), text.top() + i * line, text.width(), line), Qt::AlignLeft | Qt::AlignVCenter,
                   lines[i]);
    }
}

void NodeCanvas::resizeEvent(QResizeEvent*) {
    if (!framed_ && width() > 100 && height() > 100) {
        framed_ = true;
        frame_all();
    }
}

void NodeCanvas::mousePressEvent(QMouseEvent* e) {
    const QPointF s = to_scene(e->position());
    last_mouse_ = e->position();
    if (e->button() == Qt::MiddleButton) {
        drag_ = Drag::Pan;
        return;
    }
    if (e->button() != Qt::LeftButton) return;

    const PortRef port = port_at(s);
    if (port.valid()) {
        drag_ = Drag::Link;
        drag_pos_ = s;
        drag_from_ = port;
        if (!port.output) {
            // Picking up a linked input detaches the link and keeps dragging it from its source.
            if (const lenia::Link* l = graph_->link_into(port.node, port.port)) {
                drag_from_ = {l->from_node, l->from_port, true};
                graph_->disconnect(port.node, port.port);
                set_selected_link(std::nullopt);
                emit graph_changed();
            }
        }
        update();
        return;
    }
    if (const int id = node_at(s); id >= 0) {
        select_node(id);
        const lenia::Node* n = graph_->node(id);
        move_offset_ = s - QPointF(n->x, n->y);
        drag_ = Drag::Move;
        return;
    }
    if (const auto link = link_at(s)) {
        set_selected_link(link);
        return;
    }
    select_node(-1);
    set_selected_link(std::nullopt);
    drag_ = Drag::Pan;
}

void NodeCanvas::mouseMoveEvent(QMouseEvent* e) {
    const QPointF s = to_scene(e->position());
    switch (drag_) {
    case Drag::Move:
        graph_->move_node(selected_node_, static_cast<float>(s.x() - move_offset_.x()),
                          static_cast<float>(s.y() - move_offset_.y()));
        break;
    case Drag::Pan:
        pan_ += e->position() - last_mouse_;
        break;
    case Drag::Link:
        drag_pos_ = s;
        hover_port_ = port_at(s);
        break;
    case Drag::None: {
        const PortRef hover = port_at(s);
        if (hover == hover_port_) {
            last_mouse_ = e->position();
            return;
        }
        hover_port_ = hover;
        setCursor(hover.valid() ? Qt::CrossCursor : Qt::ArrowCursor);
        break;
    }
    }
    last_mouse_ = e->position();
    update();
}

void NodeCanvas::mouseReleaseEvent(QMouseEvent* e) {
    if (drag_ == Drag::Link) {
        const PortRef target = port_at(to_scene(e->position()));
        if (target.valid() && target != drag_from_) {
            const PortRef out = drag_from_.output ? drag_from_ : target;
            const PortRef in = drag_from_.output ? target : drag_from_;
            const lenia::ConnectResult r =
                out.output && !in.output ? graph_->connect(out.node, out.port, in.node, in.port)
                                         : lenia::ConnectResult::BadPort;
            if (r == lenia::ConnectResult::Ok) {
                emit graph_changed();
            } else {
                emit message(tr("Cannot link: %1").arg(QString::fromLatin1(lenia::describe(r))));
            }
        }
    }
    drag_ = Drag::None;
    hover_port_ = {};
    update();
}

void NodeCanvas::wheelEvent(QWheelEvent* e) {
    const double notches = e->angleDelta().y() / 120.0;
    if (notches == 0.0) return;
    const QPointF before = to_scene(e->position());
    zoom_ = std::clamp(zoom_ * std::pow(1.15, notches), 0.3, 2.5);
    pan_ = e->position() - before * zoom_;
    update();
}

void NodeCanvas::keyPressEvent(QKeyEvent* e) {
    if (e->key() == Qt::Key_Delete || e->key() == Qt::Key_Backspace) {
        delete_selection();
    } else if (e->key() == Qt::Key_Home) {
        frame_all();
    } else {
        QWidget::keyPressEvent(e);
    }
}

void NodeCanvas::contextMenuEvent(QContextMenuEvent* e) {
    const QPointF s = to_scene(e->pos());
    QMenu menu(this);
    const PortRef port = port_at(s);
    const int id = node_at(s);
    const auto link = link_at(s);

    if (port.valid() && !port.output && graph_->link_into(port.node, port.port)) {
        menu.addAction(tr("Unlink this input"), this, [this, port] {
            graph_->disconnect(port.node, port.port);
            emit graph_changed();
            update();
        });
    } else if (id >= 0) {
        select_node(id);
        const lenia::Node* n = graph_->node(id);
        QAction* del = menu.addAction(tr("Delete node"), this, [this, id] { remove_node(id); });
        del->setEnabled(!lenia::is_sink(n->kind));
        if (n->kind == NodeKind::Output && !graph_->link_into(id, lenia::output_node::NextIn)) {
            menu.addAction(tr("Wire the built-in Lenia rule as nodes"), this, &NodeCanvas::wire_lenia_rule);
        }
        if (n->kind == NodeKind::View && !graph_->link_into(id, lenia::view_node::ColorIn)) {
            menu.addAction(tr("Wire the panel display as nodes"), this, &NodeCanvas::wire_view_requested);
        }
        menu.addAction(tr("Unlink all inputs"), this, [this, id] {
            const lenia::Node* node = graph_->node(id);
            if (!node) return;
            const auto count = static_cast<int>(lenia::node_spec(node->kind).inputs.size());
            for (int i = 0; i < count; ++i) graph_->disconnect(id, i);
            emit graph_changed();
            update();
        });
    } else if (link) {
        set_selected_link(link);
        menu.addAction(tr("Unlink"), this, [this, l = *link] {
            graph_->disconnect(l.to_node, l.to_port);
            set_selected_link(std::nullopt);
            emit graph_changed();
        });
    } else {
        const bool display = graph_->domain() == lenia::GraphDomain::Display;
        QMenu* add = menu.addMenu(display ? tr("Add value node") : tr("Add node"));
        QMenu* add_cell = menu.addMenu(display ? tr("Add display node") : tr("Add cell rule node"));
        for (NodeKind k : lenia::node_kinds(graph_->domain())) {
            if (lenia::is_sink(k)) continue;
            (is_value_node(k) ? add : add_cell)
                ->addAction(QString::fromStdString(lenia::node_spec(k).name), this, [this, k, s] { add_node_at(k, s); });
        }
        menu.addSeparator();
        menu.addAction(tr("Frame all (Home)"), this, &NodeCanvas::frame_all);
    }
    menu.exec(e->globalPos());
}

void NodeCanvas::add_node_at(NodeKind kind, QPointF scene) {
    const int id = graph_->add_node(kind, static_cast<float>(scene.x()), static_cast<float>(scene.y()));
    select_node(id);
    emit graph_changed();
    emit message(tr("Added %1: drag from its ports to link it").arg(QString::fromStdString(lenia::node_spec(kind).name)));
}

void NodeCanvas::add_node(NodeKind kind) {
    // Staggered so repeated adds do not stack exactly.
    const QPointF center = to_scene(QPointF(width() / 2.0, height() / 2.0));
    const double jitter = static_cast<double>(graph_->nodes().size() % 5) * 18.0;
    add_node_at(kind, center - QPointF(kNodeWidth / 2 - jitter, 60 - jitter));
}

void NodeCanvas::remove_node(int id) {
    const lenia::Node* n = graph_->node(id);
    if (!n) return;
    if (lenia::is_sink(n->kind)) {
        emit message(tr("The %1 node cannot be deleted; unlink its inputs instead")
                         .arg(QString::fromStdString(lenia::node_spec(n->kind).name)));
        return;
    }
    graph_->remove_node(id);
    if (selected_node_ == id) select_node(-1);
    if (selected_link_ && (selected_link_->from_node == id || selected_link_->to_node == id)) selected_link_.reset();
    emit graph_changed();
    update();
}

void NodeCanvas::delete_selection() {
    if (selected_link_) {
        graph_->disconnect(selected_link_->to_node, selected_link_->to_port);
        set_selected_link(std::nullopt);
        emit graph_changed();
    } else if (selected_node_ >= 0) {
        remove_node(selected_node_);
    }
}

void NodeCanvas::frame_all() {
    if (graph_->nodes().empty()) return;
    QRectF bounds;
    for (const lenia::Node& n : graph_->nodes()) bounds = bounds.united(node_rect(n));
    bounds.adjust(-30, -30, 30, 40); // room for the hint line
    zoom_ = std::clamp(std::min(width() / bounds.width(), height() / bounds.height()), 0.3, 1.2);
    pan_ = QPointF(width() / 2.0, height() / 2.0) - bounds.center() * zoom_;
    update();
}

void NodeCanvas::graph_replaced() {
    selected_link_.reset();
    selected_node_ = -2; // force the signal
    select_node(-1);
    drag_ = Drag::None;
    hover_port_ = {};
    frame_all();
}

// ---------------------------------------------------------------------------------------------
// NodeEditor

NodeEditor::NodeEditor(lenia::Graph* graph, QWidget* parent) : QWidget(parent), graph_(graph) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);

    const bool display = graph_->domain() == lenia::GraphDomain::Display;
    auto* toolbar = new QHBoxLayout;
    canvas_ = new NodeCanvas(graph_);
    // Two menus: values over time (modulation) and per-cell rule nodes, or display nodes.
    auto* add_value = new QPushButton(display ? tr("Add value") : tr("Add value / modulation"));
    auto* add_cell = new QPushButton(display ? tr("Add display node") : tr("Add cell rule node"));
    if (display) {
        add_value->setToolTip(tr("Time, Oscillator, Constant: evaluated once per generation, e.g. to animate a "
                                 "Mix factor"));
        add_cell->setToolTip(tr("Fields, Color ramp, Color, Mix: evaluated for every cell on the GPU; wire the "
                                "result into the View node"));
    } else {
        add_value->setToolTip(
            tr("Time, Oscillator, Constant, Kernel warp, Environment: evaluated once per generation"));
        add_cell->setToolTip(tr("Density, Neighborhood, Growth, Add, Multiply, Clamp, Compare, Select, Life: "
                                "evaluated for every cell; wire the result into Simulation's Next state"));
    }
    auto* value_menu = new QMenu(add_value);
    auto* cell_menu = new QMenu(add_cell);
    value_menu->setToolTipsVisible(true);
    cell_menu->setToolTipsVisible(true);
    for (NodeKind k : lenia::node_kinds(graph_->domain())) {
        if (lenia::is_sink(k)) continue;
        const lenia::NodeSpec& spec = lenia::node_spec(k);
        QAction* a = (is_value_node(k) ? value_menu : cell_menu)
                         ->addAction(QString::fromStdString(spec.name), canvas_, [this, k] { canvas_->add_node(k); });
        a->setToolTip(QString::fromStdString(spec.help));
    }
    add_value->setMenu(value_menu);
    add_cell->setMenu(cell_menu);
    toolbar->addWidget(add_value);
    toolbar->addWidget(add_cell);
    toolbar->addStretch();
    toolbar->addWidget(new QLabel(tr("Example:")));
    presets_ = new QComboBox;
    if (display) {
        presets_->addItem(tr("Panel display (View unlinked, default)"), static_cast<int>(lenia::DisplayPreset::Panel));
        presets_->addItem(tr("Panel display as nodes"), static_cast<int>(lenia::DisplayPreset::WiredPanel));
        presets_->addItem(tr("Density + pulsing growth overlay"), static_cast<int>(lenia::DisplayPreset::GrowthOverlay));
        presets_->setToolTip(tr("Display examples replace the display graph only; the simulation is untouched."));
    } else {
        presets_->addItem(tr("Lenia rule, idle (default Orbium)"), static_cast<int>(lenia::GraphPreset::Idle));
        presets_->addItem(tr("Lenia rule, breathing kernel"), static_cast<int>(lenia::GraphPreset::BreathingKernel));
        presets_->addItem(tr("Lenia rule, drifting environment"),
                          static_cast<int>(lenia::GraphPreset::DriftingEnvironment));
        presets_->addItem(tr("Game of Life (Life node)"), static_cast<int>(lenia::GraphPreset::GameOfLife));
        presets_->addItem(tr("Game of Life (Compare / Select)"),
                          static_cast<int>(lenia::GraphPreset::GameOfLifeCompare));
        presets_->setToolTip(tr("Lenia examples replace the graph; the world keeps running, press R to replay from "
                                "generation 0. Game of Life examples load the whole Life setup (Moore kernel, "
                                "glider) and run it; Restore Orbium returns to Lenia."));
    }
    auto* load = new QPushButton(tr("Load"));
    load->setToolTip(presets_->toolTip());
    auto* frame = new QPushButton(tr("Frame all"));
    frame->setToolTip(tr("Home"));
    toolbar->addWidget(presets_);
    toolbar->addWidget(load);
    toolbar->addWidget(frame);
    layout->addLayout(toolbar);

    auto* inspector = new QWidget;
    auto* inspector_layout = new QVBoxLayout(inspector);
    inspector_title_ = new QLabel;
    QFont title_font = inspector_title_->font();
    title_font.setBold(true);
    inspector_title_->setFont(title_font);
    inspector_help_ = new QLabel;
    inspector_help_->setWordWrap(true);
    inspector_help_->setForegroundRole(QPalette::PlaceholderText);
    inspector_fields_ = new QWidget;
    inspector_form_ = new QFormLayout(inspector_fields_);
    inspector_form_->setContentsMargins(0, 0, 0, 0);
    delete_node_ = new QPushButton(tr("Delete node"));
    inspector_layout->addWidget(inspector_title_);
    inspector_layout->addWidget(inspector_help_);
    inspector_layout->addWidget(inspector_fields_);
    inspector_layout->addWidget(delete_node_);
    inspector_layout->addStretch();
    auto* inspector_scroll = new QScrollArea;
    inspector_scroll->setWidget(inspector);
    inspector_scroll->setWidgetResizable(true);
    inspector_scroll->setFrameShape(QFrame::NoFrame);
    inspector_scroll->setMinimumWidth(250);

    auto* split = new QSplitter(Qt::Horizontal);
    split->addWidget(canvas_);
    split->addWidget(inspector_scroll);
    split->setStretchFactor(0, 1);
    split->setStretchFactor(1, 0);
    split->setChildrenCollapsible(false);
    layout->addWidget(split, 1);

    connect(canvas_, &NodeCanvas::graph_changed, this, &NodeEditor::graph_changed);
    connect(canvas_, &NodeCanvas::message, this, &NodeEditor::message);
    connect(canvas_, &NodeCanvas::selection_changed, this, &NodeEditor::rebuild_inspector);
    // Canvas edits (links) change which parameters show "(+ link)". Inspector edits do not
    // rebuild: that would delete the spin box emitting the change.
    connect(canvas_, &NodeCanvas::graph_changed, this, &NodeEditor::rebuild_inspector);
    connect(canvas_, &NodeCanvas::wire_view_requested, this, &NodeEditor::wire_view_requested);
    connect(load, &QPushButton::clicked, this, [this, display] {
        const int preset = presets_->currentData().toInt();
        if (display) emit display_preset_requested(static_cast<lenia::DisplayPreset>(preset));
        else emit preset_requested(static_cast<lenia::GraphPreset>(preset));
    });
    connect(frame, &QPushButton::clicked, canvas_, &NodeCanvas::frame_all);
    connect(delete_node_, &QPushButton::clicked, this, [this] {
        const int id = canvas_->selected_node();
        const lenia::Node* n = graph_->node(id);
        if (!n || lenia::is_sink(n->kind)) return;
        graph_->remove_node(id);
        canvas_->select_node(-1);
        emit graph_changed();
        canvas_->update();
    });
    rebuild_inspector();
}

void NodeEditor::load_preset(lenia::GraphPreset preset) {
    *graph_ = lenia::make_preset(preset);
    canvas_->graph_replaced();
    emit graph_changed();
    emit message(tr("Loaded example \"%1\"; R replays it from generation 0").arg(presets_->itemText(
        std::max(0, presets_->findData(static_cast<int>(preset))))));
}

void NodeEditor::rebuild_inspector() {
    while (inspector_form_->rowCount() > 0) inspector_form_->removeRow(0);

    const int id = canvas_->selected_node();
    const lenia::Node* n = graph_->node(id);
    if (!n && graph_->domain() == lenia::GraphDomain::Display) {
        inspector_title_->setText(tr("No node selected"));
        inspector_help_->setText(
            tr("Click a node to edit its parameters here.\n\n"
               "The display graph colors the viewport, running or paused. Fields gives each cell's "
               "density, neighborhood and growth; Color ramps turn them into colors (select one to edit "
               "its stops); Mix blends two colors; the View node shows the result.\n\n"
               "With View unlinked the Simulation panel's field and palette are shown, as before. Value "
               "nodes (Time, Oscillator, Constant) can animate a Mix factor; it replays exactly."));
        delete_node_->setVisible(false);
        return;
    }
    if (!n) {
        inspector_title_->setText(tr("No node selected"));
        inspector_help_->setText(
            tr("Click a node to edit its parameters here.\n\n"
               "Value nodes (Time, Oscillator, Constant) drive the inputs of Kernel and Environment; "
               "those feed the Simulation node, which the Vulkan simulation reads every generation. "
               "A linked Value input adds its value to the parameter of the same name.\n\n"
               "Cell rule nodes (pink) compute one value per cell; the one wired into Simulation's Next "
               "state is the cell's next state. A linked cell input replaces its parameter.\n\n"
               "Everything is a function of the generation: pause, step and reset replay exactly."));
        delete_node_->setVisible(false);
        return;
    }
    const lenia::NodeSpec& spec = lenia::node_spec(n->kind);
    inspector_title_->setText(QString::fromStdString(spec.name));
    inspector_help_->setText(QString::fromStdString(spec.help));
    delete_node_->setVisible(!lenia::is_sink(n->kind));

    if (n->kind == NodeKind::View && !graph_->link_into(id, lenia::view_node::ColorIn)) {
        auto* wire = new QPushButton(tr("Wire the panel display as nodes"));
        wire->setToolTip(tr("Adds Fields and a Color ramp reproducing the Simulation panel's field, palette, low "
                            "and high, linked into View: the same picture, now editable."));
        // Queued: wiring rebuilds this inspector, which deletes the button.
        connect(wire, &QPushButton::clicked, this, &NodeEditor::wire_view_requested, Qt::QueuedConnection);
        inspector_form_->addRow(wire);
    }
    if (n->kind == NodeKind::ColorRamp) {
        const auto changed = [this] {
            canvas_->update();
            emit graph_changed();
        };
        auto* ramp = new RampEditor(graph_, id, changed);
        inspector_form_->addRow(ramp);
        auto* palette = new QComboBox;
        palette->addItem(tr("Set stops from palette…"));
        palette->addItems(palette_names());
        auto* reverse = new QPushButton(tr("Reverse"));
        reverse->setToolTip(tr("Mirror the stops: position p becomes 1 − p"));
        auto* row = new QHBoxLayout;
        row->addWidget(palette, 1);
        row->addWidget(reverse);
        inspector_form_->addRow(row);
        connect(palette, &QComboBox::activated, this, [this, id, ramp, palette, changed](int i) {
            if (i <= 0) return;
            lenia::set_ramp_stops(*graph_, id, lenia::palette_ramp(static_cast<lenia::DisplayPalette>(i - 1)));
            palette->setCurrentIndex(0);
            ramp->update();
            changed();
        });
        connect(reverse, &QPushButton::clicked, this, [this, id, ramp, changed] {
            const lenia::Node* node = graph_->node(id);
            if (!node) return;
            std::vector<lenia::RampStop> s = lenia::ramp_stops(*node);
            std::reverse(s.begin(), s.end());
            for (lenia::RampStop& stop : s) stop.position = 1.0f - stop.position;
            lenia::set_ramp_stops(*graph_, id, s);
            ramp->update();
            changed();
        });
    }

    if (n->kind == NodeKind::Output && !graph_->link_into(id, lenia::output_node::NextIn)) {
        auto* wire = new QPushButton(tr("Wire the built-in Lenia rule as nodes"));
        wire->setToolTip(tr("Adds Density, Neighborhood, Growth, Add, Multiply and Clamp nodes computing exactly "
                            "the built-in update (an Environment linked here moves into the rule), so it can be "
                            "edited. Experiments saved before rule nodes open unwired."));
        // Queued: wiring rebuilds this inspector, which deletes the button.
        connect(wire, &QPushButton::clicked, canvas_, &NodeCanvas::wire_lenia_rule, Qt::QueuedConnection);
        inspector_form_->addRow(wire);
    }
    if (n->kind == NodeKind::Growth && static_cast<int>(n->params[lenia::growth_node::Source]) == lenia::growth_node::Panel) {
        auto* note = new QLabel(tr("Source = Simulation panel: Function, μ, σ and dt below are not used; the panel's "
                                   "growth controls are."));
        note->setWordWrap(true);
        note->setForegroundRole(QPalette::PlaceholderText);
        inspector_form_->addRow(note);
    }

    for (int i = 0; i < static_cast<int>(spec.params.size()); ++i) {
        // A Color ramp's stops are edited in its RampEditor above.
        if (n->kind == NodeKind::ColorRamp && i >= lenia::ramp_node::Stops) break;
        const lenia::ParamSpec& ps = spec.params[static_cast<std::size_t>(i)];
        // Does a link add to (Value input) or replace (cell input) this parameter?
        bool linked = false;
        bool replaced = false;
        for (int in = 0; in < static_cast<int>(spec.inputs.size()); ++in) {
            const lenia::PortSpec& port = spec.inputs[static_cast<std::size_t>(in)];
            if (port.param != i || !graph_->link_into(id, in)) continue;
            (port.type == PortType::Cell ? replaced : linked) = true;
        }
        QString label = QString::fromStdString(ps.name);
        if (linked) label += tr(" (+ link)");
        if (replaced) label += tr(" (linked: unused)");
        const float value = n->params[static_cast<std::size_t>(i)];

        if (!ps.choices.empty()) {
            auto* combo = new QComboBox;
            for (const std::string& c : ps.choices) combo->addItem(QString::fromStdString(c));
            combo->setCurrentIndex(static_cast<int>(value));
            const bool source = n->kind == NodeKind::Growth && i == lenia::growth_node::Source;
            connect(combo, &QComboBox::currentIndexChanged, this, [this, id, i, source](int v) {
                graph_->set_param(id, i, static_cast<float>(v));
                canvas_->update();
                emit graph_changed();
                // The panel note depends on Source; queued, since rebuilding deletes this combo.
                if (source) QMetaObject::invokeMethod(this, [this] { rebuild_inspector(); }, Qt::QueuedConnection);
            });
            inspector_form_->addRow(label, combo);
        } else {
            auto* spin = new QDoubleSpinBox;
            spin->setRange(ps.min, ps.max);
            spin->setSingleStep(ps.step);
            spin->setDecimals(ps.decimals);
            spin->setValue(value);
            spin->setKeyboardTracking(false);
            connect(spin, &QDoubleSpinBox::valueChanged, this, [this, id, i](double v) {
                graph_->set_param(id, i, static_cast<float>(v));
                canvas_->update();
                emit graph_changed();
            });
            inspector_form_->addRow(label, spin);
        }
    }
    if (spec.params.empty()) {
        inspector_form_->addRow(new QLabel(tr("No parameters: link nodes into its inputs.")));
    }
}

} // namespace workbench
