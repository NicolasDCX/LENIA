#pragma once

#include "lenia/kernel_shape.hpp"

#include <QObject>
#include <QPointF>
#include <QString>
#include <QWidget>

namespace workbench {

// The kernel being edited: a parametric ring shape (lenia::make_kernel) plus a freehand
// paint layer, summed and clamped to [-1, 1] as raw weights, then normalized.
class KernelModel : public QObject {
    Q_OBJECT
public:
    explicit KernelModel(QObject* parent = nullptr);

    const lenia::KernelShape& shape() const { return shape_; }
    lenia::KernelNormalization normalization() const { return normalization_; }

    // Replaces the shape. The paint layer is resampled to the new radius, so strokes stay at
    // the same relative distance from the center.
    void set_shape(const lenia::KernelShape& shape);
    void set_radius(int radius);
    void set_core(lenia::KernelCore core);
    // Grows with zero-weight rings or drops outer rings.
    void set_ring_count(int count);
    void set_ring(int index, float weight);
    void set_normalization(lenia::KernelNormalization mode);
    void clear_paint();
    bool has_paint() const;
    // Replaces shape, normalization and paint layer at once, as saved (no resampling). `paint`
    // is empty for none or (2R + 1)^2 cells; returns false and changes nothing otherwise.
    bool restore(const lenia::KernelShape& shape, lenia::KernelNormalization mode, const lenia::Grid& paint);

    enum class Brush { Add, Erase };
    // Paints around `cell` (kernel cell coordinates, (0, 0) = top-left cell's corner) with a
    // linear falloff over `radius` cells. With `rings`, every cell whose distance from the
    // kernel center is within `radius` of the cursor's distance is painted instead: the
    // stroke is radially symmetric. Add moves raw weights by `amount` (signed); Erase moves
    // the paint layer toward zero.
    void paint(QPointF cell, double radius, float amount, Brush brush, bool rings);

    const lenia::Grid& paint_layer() const { return paint_; } // added to the shape's weights
    const lenia::Grid& raw() const { return raw_; }       // shape + paint, in [-1, 1]
    const lenia::Grid& kernel() const { return kernel_; } // normalized raw, what the GPU gets
    // The kernel with the shape warped (node graph): warped shape plus the paint layer scaled
    // with the radius, normalized the same way. Equals kernel() for a neutral warp.
    lenia::Grid build(const lenia::KernelWarp& warp) const;
    // Size, weight sums, and why normalization fell back to raw weights, if it did.
    QString summary() const;

signals:
    void changed();

private:
    void rebuild();

    lenia::KernelShape shape_;
    lenia::KernelNormalization normalization_ = lenia::KernelNormalization::Sum;
    lenia::Grid shape_grid_;
    lenia::Grid paint_; // same size as shape_grid_
    lenia::Grid raw_;
    lenia::Grid kernel_;
    QString normalization_error_;
};

// Plot of the radial profile K(r) with one bar per ring. Dragging with the left button sets
// ring weights in [-1, 1] (shift snaps to 0.25 steps), the right button zeroes a ring. Dots
// show every raw kernel cell at its radius, so painted strokes are visible too.
class RingProfileWidget : public QWidget {
    Q_OBJECT
public:
    explicit RingProfileWidget(KernelModel* model, QWidget* parent = nullptr);
    QSize sizeHint() const override { return {320, 160}; }

signals:
    // A drag ended (button released): one undo step (docs/undo.md).
    void edited(const QString& text);

protected:
    void paintEvent(QPaintEvent*) override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;

private:
    QRectF plot_rect() const;
    void edit(QPointF pos, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);

    KernelModel* model_;
    bool editing_ = false; // a drag is setting ring weights
};

// Kernel cells drawn with a signed palette (raw weights, -1 blue .. +1 red). Left drag adds
// `strength`, right drag subtracts it, ctrl + drag erases paint.
class KernelCanvas : public QWidget {
    Q_OBJECT
public:
    explicit KernelCanvas(KernelModel* model, QWidget* parent = nullptr);
    QSize sizeHint() const override { return {320, 320}; }
    bool hasHeightForWidth() const override { return true; }
    int heightForWidth(int w) const override { return w; }

    void set_brush(double radius, float strength, bool rings);

signals:
    // A stroke ended (button released): one undo step however many dabs it painted.
    void edited(const QString& text);

protected:
    void paintEvent(QPaintEvent*) override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;

private:
    QRectF grid_rect() const;
    void stroke(QPointF pos, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);

    KernelModel* model_;
    double brush_radius_ = 1.5;
    float strength_ = 0.1f;
    bool rings_ = false;
    QPointF hover_{-1, -1};
    bool erasing_ = false; // the stroke in progress erases (named in its undo step)
    bool stroking_ = false;
};

} // namespace workbench
