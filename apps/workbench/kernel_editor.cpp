#include "kernel_editor.hpp"

#include "palettes.hpp"

#include <QImage>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace workbench {

// ---------------------------------------------------------------------------------------------
// KernelModel

KernelModel::KernelModel(QObject* parent) : QObject(parent) { rebuild(); }

void KernelModel::set_shape(const lenia::KernelShape& shape) {
    if (shape.radius < 1 || shape.rings.empty()) return;
    const int old_r = shape_.radius;
    const int new_r = shape.radius;
    if (old_r != new_r && !paint_.cells.empty()) {
        // Nearest-neighbor resample, scaling offsets from the center.
        const std::size_t old_size = paint_.width;
        const std::size_t new_size = static_cast<std::size_t>(2 * new_r + 1);
        lenia::Grid resampled(new_size, new_size, 0.0f);
        const double scale = static_cast<double>(old_r) / new_r;
        for (std::size_t y = 0; y < new_size; ++y) {
            for (std::size_t x = 0; x < new_size; ++x) {
                const long long sx = std::llround((static_cast<double>(x) - new_r) * scale) + old_r;
                const long long sy = std::llround((static_cast<double>(y) - new_r) * scale) + old_r;
                if (sx < 0 || sy < 0 || sx >= static_cast<long long>(old_size) ||
                    sy >= static_cast<long long>(old_size)) {
                    continue;
                }
                resampled.at(x, y) = paint_.at(static_cast<std::size_t>(sx), static_cast<std::size_t>(sy));
            }
        }
        paint_ = std::move(resampled);
    }
    shape_ = shape;
    rebuild();
}

void KernelModel::set_radius(int radius) {
    lenia::KernelShape s = shape_;
    s.radius = radius;
    set_shape(s);
}

void KernelModel::set_core(lenia::KernelCore core) {
    shape_.core = core;
    rebuild();
}

void KernelModel::set_ring_count(int count) {
    if (count < 1) return;
    shape_.rings.resize(static_cast<std::size_t>(count), 0.0f);
    rebuild();
}

void KernelModel::set_ring(int index, float weight) {
    if (index < 0 || index >= static_cast<int>(shape_.rings.size())) return;
    weight = std::clamp(weight, -1.0f, 1.0f);
    if (shape_.rings[static_cast<std::size_t>(index)] == weight) return;
    shape_.rings[static_cast<std::size_t>(index)] = weight;
    rebuild();
}

void KernelModel::set_normalization(lenia::KernelNormalization mode) {
    normalization_ = mode;
    rebuild();
}

void KernelModel::clear_paint() {
    std::fill(paint_.cells.begin(), paint_.cells.end(), 0.0f);
    rebuild();
}

bool KernelModel::has_paint() const {
    return std::any_of(paint_.cells.begin(), paint_.cells.end(), [](float v) { return v != 0.0f; });
}

bool KernelModel::restore(const lenia::KernelShape& shape, lenia::KernelNormalization mode, const lenia::Grid& paint) {
    const std::size_t side = static_cast<std::size_t>(2 * shape.radius + 1);
    if (shape.radius < 1 || shape.rings.empty()) return false;
    if (!paint.cells.empty() &&
        (paint.width != side || paint.height != side || paint.cells.size() != side * side)) {
        return false;
    }
    shape_ = shape;
    normalization_ = mode;
    paint_ = paint.cells.empty() ? lenia::Grid(side, side, 0.0f) : paint;
    rebuild();
    return true;
}

void KernelModel::paint(QPointF cell, double radius, float amount, Brush brush, bool rings) {
    const std::size_t n = raw_.width;
    if (n == 0) return;
    radius = std::max(radius, 0.5);
    const double c = static_cast<double>(n) / 2.0; // kernel center in cell-corner coordinates
    const double cursor_r = std::hypot(cell.x() - c, cell.y() - c);

    for (std::size_t y = 0; y < n; ++y) {
        for (std::size_t x = 0; x < n; ++x) {
            const double px = static_cast<double>(x) + 0.5;
            const double py = static_cast<double>(y) + 0.5;
            const double dist = rings ? std::fabs(std::hypot(px - c, py - c) - cursor_r)
                                      : std::hypot(px - cell.x(), py - cell.y());
            if (dist > radius) continue;
            const float falloff = static_cast<float>(1.0 - dist / (radius + 0.5));
            float& p = paint_.at(x, y);
            if (brush == Brush::Erase) {
                p *= 1.0f - falloff;
            } else {
                const float base = shape_grid_.at(x, y);
                p = std::clamp(base + p + amount * falloff, -1.0f, 1.0f) - base;
            }
        }
    }
    rebuild();
}

void KernelModel::rebuild() {
    shape_grid_ = lenia::make_kernel(shape_);
    if (paint_.width != shape_grid_.width || paint_.height != shape_grid_.height) {
        paint_ = lenia::Grid(shape_grid_.width, shape_grid_.height, 0.0f);
    }
    raw_ = shape_grid_;
    for (std::size_t i = 0; i < raw_.cells.size(); ++i) {
        raw_.cells[i] = std::clamp(raw_.cells[i] + paint_.cells[i], -1.0f, 1.0f);
    }
    try {
        kernel_ = lenia::normalize_kernel(raw_, normalization_);
        normalization_error_.clear();
    } catch (const std::invalid_argument& e) {
        kernel_ = raw_;
        normalization_error_ = QString::fromUtf8(e.what());
    }
    emit changed();
}

lenia::Grid KernelModel::build(const lenia::KernelWarp& warp) const {
    if (warp.neutral()) return kernel_;
    lenia::Grid raw = lenia::make_kernel(shape_, warp);
    if (has_paint()) {
        // Nearest paint cell at the same relative distance from the center.
        const long long old_half = static_cast<long long>(paint_.width / 2);
        const long long new_half = static_cast<long long>(raw.width / 2);
        const double scale = lenia::warped_radius(shape_, warp) / shape_.radius;
        for (std::size_t y = 0; y < raw.height; ++y) {
            for (std::size_t x = 0; x < raw.width; ++x) {
                const long long sx = std::llround((static_cast<double>(x) - new_half) / scale) + old_half;
                const long long sy = std::llround((static_cast<double>(y) - new_half) / scale) + old_half;
                if (sx < 0 || sy < 0 || sx >= static_cast<long long>(paint_.width) ||
                    sy >= static_cast<long long>(paint_.height)) {
                    continue;
                }
                float& w = raw.at(x, y);
                w = std::clamp(w + paint_.at(static_cast<std::size_t>(sx), static_cast<std::size_t>(sy)), -1.0f, 1.0f);
            }
        }
    }
    try {
        return lenia::normalize_kernel(raw, normalization_);
    } catch (const std::invalid_argument&) {
        return raw; // same fallback as rebuild()
    }
}

QString KernelModel::summary() const {
    const lenia::KernelStats raw = lenia::kernel_stats(raw_);
    const lenia::KernelStats out = lenia::kernel_stats(kernel_);
    QString s = QStringLiteral("%1×%1 kernel, %2 nonzero cells\n"
                               "raw: Σ = %3 (positive %4, negative %5)\n"
                               "normalized: Σ = %6 (positive %7, negative %8)")
                    .arg(raw_.width)
                    .arg(raw.nonzero)
                    .arg(raw.sum, 0, 'g', 4)
                    .arg(raw.positive, 0, 'g', 4)
                    .arg(raw.negative, 0, 'g', 4)
                    .arg(out.sum, 0, 'g', 4)
                    .arg(out.positive, 0, 'g', 4)
                    .arg(out.negative, 0, 'g', 4);
    if (!normalization_error_.isEmpty()) {
        s += QStringLiteral("\nNormalization skipped, using raw weights: %1").arg(normalization_error_);
    }
    return s;
}

// ---------------------------------------------------------------------------------------------
// RingProfileWidget

RingProfileWidget::RingProfileWidget(KernelModel* model, QWidget* parent) : QWidget(parent), model_(model) {
    setMinimumHeight(120);
    connect(model_, &KernelModel::changed, this, qOverload<>(&QWidget::update));
    setToolTip(tr("Drag to set ring weights (shift snaps to 0.25), right-click zeroes a ring"));
}

QRectF RingProfileWidget::plot_rect() const { return QRectF(rect()).adjusted(30, 8, -8, -18); }

void RingProfileWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    p.fillRect(rect(), palette().base());

    const QRectF pr = plot_rect();
    const auto to_y = [&](double v) { return pr.top() + (1.0 - v) / 2.0 * pr.height(); };
    const lenia::KernelShape& shape = model_->shape();
    const int rings = static_cast<int>(shape.rings.size());
    const double ring_w = pr.width() / rings;

    // Ring bands and weight bars.
    for (int i = 0; i < rings; ++i) {
        const QRectF band(pr.left() + i * ring_w, pr.top(), ring_w, pr.height());
        if (i % 2) p.fillRect(band, palette().alternateBase());
        const float w = shape.rings[static_cast<std::size_t>(i)];
        const QRectF bar = QRectF(QPointF(band.left() + 2, to_y(0.0)), QPointF(band.right() - 2, to_y(w))).normalized();
        QColor color = palette_color(RedBlue, (w + 1.0f) / 2.0f);
        color.setAlpha(110);
        p.fillRect(bar, color);
    }

    // Zero line and axes.
    p.setPen(QPen(palette().text().color(), 0.5, Qt::DashLine));
    p.drawLine(QPointF(pr.left(), to_y(0.0)), QPointF(pr.right(), to_y(0.0)));
    p.setPen(QPen(palette().mid().color(), 1.0));
    p.drawRect(pr);
    p.setPen(palette().text().color());
    const QFontMetrics fm(font());
    p.drawText(QRectF(0, to_y(1.0) - 8, pr.left() - 4, 16), Qt::AlignRight | Qt::AlignVCenter, QStringLiteral("+1"));
    p.drawText(QRectF(0, to_y(0.0) - 8, pr.left() - 4, 16), Qt::AlignRight | Qt::AlignVCenter, QStringLiteral("0"));
    p.drawText(QRectF(0, to_y(-1.0) - 8, pr.left() - 4, 16), Qt::AlignRight | Qt::AlignVCenter, QStringLiteral("-1"));
    p.drawText(QRectF(pr.left(), pr.bottom() + 2, pr.width(), fm.height()), Qt::AlignLeft, QStringLiteral("r = 0"));
    p.drawText(QRectF(pr.left(), pr.bottom() + 2, pr.width(), fm.height()), Qt::AlignRight, QStringLiteral("r = 1"));

    p.save();
    p.setClipRect(pr);

    // Every raw cell (shape + paint) at its normalized radius.
    const lenia::Grid& raw = model_->raw();
    const double half = static_cast<double>(raw.width / 2);
    QPolygonF dots;
    dots.reserve(static_cast<int>(raw.cells.size()));
    for (std::size_t y = 0; y < raw.height; ++y) {
        for (std::size_t x = 0; x < raw.width; ++x) {
            const double r = std::hypot(x - half, y - half) / shape.radius;
            if (r > 1.0) continue;
            dots.append(QPointF(pr.left() + r * pr.width(), to_y(raw.at(x, y))));
        }
    }
    p.setPen(QPen(palette().text().color(), 2.0, Qt::SolidLine, Qt::RoundCap));
    p.setOpacity(0.35);
    p.drawPoints(dots);
    p.setOpacity(1.0);

    // K(r) of the shape alone.
    QPainterPath curve;
    const int samples = std::max(64, static_cast<int>(pr.width()));
    for (int i = 0; i <= samples; ++i) {
        const double r = static_cast<double>(i) / samples;
        const float k = r < 1.0 ? lenia::kernel_shell(shape, static_cast<float>(r)) : 0.0f;
        const QPointF pt(pr.left() + r * pr.width(), to_y(k));
        if (i == 0) curve.moveTo(pt);
        else curve.lineTo(pt);
    }
    p.setPen(QPen(palette().highlight().color(), 1.5));
    p.drawPath(curve);
    p.restore();
}

void RingProfileWidget::edit(QPointF pos, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers) {
    const QRectF pr = plot_rect();
    const int rings = static_cast<int>(model_->shape().rings.size());
    const int ring = std::clamp(static_cast<int>((pos.x() - pr.left()) / pr.width() * rings), 0, rings - 1);
    if (buttons & Qt::RightButton) {
        editing_ = true;
        model_->set_ring(ring, 0.0f);
        return;
    }
    if (!(buttons & Qt::LeftButton)) return;
    editing_ = true;
    float w = static_cast<float>(1.0 - 2.0 * (pos.y() - pr.top()) / pr.height());
    if (modifiers & Qt::ShiftModifier) w = std::round(w * 4.0f) / 4.0f;
    model_->set_ring(ring, w);
}

void RingProfileWidget::mousePressEvent(QMouseEvent* e) { edit(e->position(), e->buttons(), e->modifiers()); }
void RingProfileWidget::mouseMoveEvent(QMouseEvent* e) { edit(e->position(), e->buttons(), e->modifiers()); }

void RingProfileWidget::mouseReleaseEvent(QMouseEvent* e) {
    // Once every button is up: a drag that sweeps several rings is one step.
    if (!editing_ || e->buttons() & (Qt::LeftButton | Qt::RightButton)) return;
    editing_ = false;
    emit edited(tr("Set ring weights"));
}

// ---------------------------------------------------------------------------------------------
// KernelCanvas

KernelCanvas::KernelCanvas(KernelModel* model, QWidget* parent) : QWidget(parent), model_(model) {
    setMinimumSize(160, 160);
    setMouseTracking(true);
    setCursor(Qt::CrossCursor);
    connect(model_, &KernelModel::changed, this, qOverload<>(&QWidget::update));
    setToolTip(tr("Left drag adds weight, right drag subtracts, ctrl + drag erases paint"));
}

void KernelCanvas::set_brush(double radius, float strength, bool rings) {
    brush_radius_ = radius;
    strength_ = strength;
    rings_ = rings;
    update();
}

QRectF KernelCanvas::grid_rect() const {
    const double side = std::min(width(), height()) - 2.0;
    return QRectF((width() - side) / 2.0, (height() - side) / 2.0, side, side);
}

void KernelCanvas::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.fillRect(rect(), palette().window());

    const lenia::Grid& raw = model_->raw();
    const int n = static_cast<int>(raw.width);
    if (n == 0) return;
    QImage image(n, n, QImage::Format_RGB32);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const float v = raw.at(static_cast<std::size_t>(x), static_cast<std::size_t>(y));
            image.setPixel(x, y, palette_color(RedBlue, (v + 1.0f) / 2.0f).rgb());
        }
    }
    const QRectF gr = grid_rect();
    p.drawImage(gr, image); // no smoothing: cells stay crisp

    const double cell = gr.width() / n;
    const QPointF center = gr.center();
    const lenia::KernelShape& shape = model_->shape();
    p.setRenderHint(QPainter::Antialiasing);

    // Kernel disk (r = 1) and ring boundaries.
    p.setPen(QPen(QColor(0, 0, 0, 140), 1.0));
    p.setBrush(Qt::NoBrush);
    p.drawEllipse(center, shape.radius * cell, shape.radius * cell);
    p.setPen(QPen(QColor(0, 0, 0, 90), 1.0, Qt::DashLine));
    const int rings = static_cast<int>(shape.rings.size());
    for (int i = 1; i < rings; ++i) {
        const double r = shape.radius * cell * i / rings;
        p.drawEllipse(center, r, r);
    }

    // Brush outline under the cursor.
    if (gr.contains(hover_)) {
        p.setPen(QPen(QColor(20, 20, 20, 200), 1.0));
        const double br = std::max(brush_radius_, 0.5) * cell;
        if (rings_) {
            const double r = std::hypot(hover_.x() - center.x(), hover_.y() - center.y());
            p.drawEllipse(center, r + br, r + br);
            if (r > br) p.drawEllipse(center, r - br, r - br);
        } else {
            p.drawEllipse(hover_, br, br);
        }
    }
}

void KernelCanvas::stroke(QPointF pos, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers) {
    if (!(buttons & (Qt::LeftButton | Qt::RightButton))) return;
    const QRectF gr = grid_rect();
    const double cell = gr.width() / model_->raw().width;
    const QPointF c = (pos - gr.topLeft()) / cell;
    const bool erase = modifiers.testFlag(Qt::ControlModifier);
    const float amount = buttons.testFlag(Qt::LeftButton) ? strength_ : -strength_;
    erasing_ = (stroking_ ? erasing_ : true) && erase;
    stroking_ = true;
    model_->paint(c, brush_radius_, amount, erase ? KernelModel::Brush::Erase : KernelModel::Brush::Add, rings_);
}

void KernelCanvas::mousePressEvent(QMouseEvent* e) { stroke(e->position(), e->buttons(), e->modifiers()); }

void KernelCanvas::mouseReleaseEvent(QMouseEvent* e) {
    if (!stroking_ || e->buttons() & (Qt::LeftButton | Qt::RightButton)) return;
    stroking_ = false;
    emit edited(erasing_ ? tr("Erase kernel paint") : tr("Paint kernel"));
}

void KernelCanvas::mouseMoveEvent(QMouseEvent* e) {
    hover_ = e->position();
    stroke(e->position(), e->buttons(), e->modifiers());
    update();
}

} // namespace workbench
