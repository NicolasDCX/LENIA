#pragma once

#include "lenia/convolve.hpp"
#include "lenia/display_graph.hpp"
#include "lenia/environment.hpp"
#include "lenia/fft_convolve.hpp"
#include "lenia/rule.hpp"
#include "lenia/step.hpp"

#include <QtGui/qtguiglobal.h>
#if !QT_CONFIG(vulkan)
#error "lenia_workbench needs a Qt 6 build with Vulkan support"
#endif

#include <QKeySequence>
#include <QPointF>
#include <QString>
#include <QVulkanWindow>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>

namespace workbench {

// Everything one generation is computed with. The kernel and rule are compared by pointer:
// handing out the same shared_ptr again means "unchanged" and costs no upload.
struct GenerationInputs {
    std::shared_ptr<const lenia::Grid> kernel; // odd-sized, at most LeniaView::kMaxKernelSize per side
    lenia::GrowthParams growth;
    lenia::Environment environment; // strength 0: none
    // The node graph's per-cell rule (at most lenia::kMaxRuleOps ops); null = the built-in
    // Lenia update with `growth` and `environment`, which a rule replaces entirely.
    std::shared_ptr<const lenia::RuleProgram> rule;
    // A register of `rule` (a Convolve node's U, lenia::RuleCompile::convolve_registers) that the
    // Neighborhood field shows instead of U; kNoRegister: U.
    std::uint32_t preview = lenia::kNoRegister;
};

// Called with the generation about to be computed (the one the step reads from). Must be a
// pure function of the generation and of state whose changes are announced with
// LeniaView::inputs_changed, so that runs replay identically.
using InputsProvider = std::function<GenerationInputs(std::uint64_t generation)>;

// The display graph's program at a generation (at most lenia::kMaxDisplayOps ops and
// lenia::kMaxDisplayStops stops), or null to draw with the ViewSettings palette. Compared by
// pointer like GenerationInputs: the same shared_ptr again costs no upload. Same purity rules
// as InputsProvider; announce changes with LeniaView::display_changed.
using DisplayProvider = std::function<std::shared_ptr<const lenia::DisplayProgram>(std::uint64_t generation)>;

// The world's state fields read back from the GPU by LeniaView::capture_world.
struct WorldCapture {
    lenia::Grid density;
    std::uint64_t generation = 0; // the generation `density` is at
    QString error;                // non-empty: the capture failed and the fields are empty
    lenia::Grid nutrient;         // the same generation's nutrient
};
// Called exactly once per capture_world, on the GUI thread. On success it runs inside the frame
// that read the world back, before anything else can change the setup, so the caller can record
// the configuration that goes with it; it must not open dialogs or spin an event loop (defer
// those, e.g. with QTimer::singleShot).
using CaptureCallback = std::function<void(const WorldCapture& capture)>;

enum class Field : std::uint32_t { Density = 0, Neighborhood = 1, Growth = 2, Nutrient = 3 };

// The state field the world brush paints (lenia::WorldState).
enum class BrushField { Density = 0, Nutrient = 1 };

struct ViewSettings {
    Field field = Field::Density;
    int palette = 0;
    float lo = 0.0f; // field value at the first palette entry
    float hi = 1.0f; // field value at the last palette entry
};

class LeniaRenderer;

// Viewport and simulation in one QVulkanWindow: the world lives in device buffers on the
// window's device and is stepped by compute dispatches recorded into each frame's command
// buffer, ahead of the render pass that draws it. The world is both state fields
// (lenia::WorldState): each cell buffer holds the density followed by the nutrient, so a
// generation updates both at once. Edits (seeds, kernels, painted cells) are
// queued here and uploaded with vkCmdUpdateBuffer in the next frame. Nothing is read back to
// the CPU during playback; only capture_world copies the fields out, on demand. Kernel, growth, environment and the cell rule come from an InputsProvider queried
// before every generation's dispatch, so they can change from one generation to the next
// within a frame. The picture comes from a DisplayProvider queried once per frame (the display
// graph's program, run per pixel by workbench_view.frag) or, without one, from ViewSettings.
//
// Mouse: left drag paints the world with the brush (shift erases), alt + left click asks to
// inspect the cell (cell_probed), right or middle drag pans, the wheel zooms about the cursor.
class LeniaView : public QVulkanWindow {
    Q_OBJECT
public:
    LeniaView();

    QVulkanWindowRenderer* createRenderer() override;

    // Replaces the density (resizing the world if needed) and sets the generation counter: 0 for
    // a reseed, the snapshot's generation when resuming one. The nutrient stays as it is (all 0
    // after a resize); the second form replaces it too (it must be the density's size).
    void upload_world(const lenia::Grid& world, std::uint64_t generation = 0);
    void upload_world(const lenia::Grid& density, const lenia::Grid& nutrient, std::uint64_t generation);
    // Reads both fields back at the next frame, after that frame's uploads and steps (so pending
    // edits are included), and passes them with their generation to `done`. One capture at a time;
    // fails at once (calling `done` before returning) if one is pending or the view is not
    // exposed, and later if the device fails or is reset first.
    void capture_world(CaptureCallback done);
    bool capture_pending() const { return static_cast<bool>(capture_); }
    // Number of completed GPU readbacks (captures) so far; playback alone never adds to it.
    std::uint64_t readback_count() const { return readbacks_; }
    // Clears the density to a single 1.0 cell in the middle and resets the generation counter;
    // the nutrient stays as it is.
    void seed_impulse();
    void set_inputs(InputsProvider provider);
    // The provider's answers changed (kernel, growth or graph edit): re-analyzes the current
    // state and uses the new inputs from the next generation on.
    void inputs_changed();
    // Used for every frame, running or paused; a null provider or program means ViewSettings.
    void set_display(DisplayProvider provider);
    // The display provider's answers changed (display graph edit): redraws, also while paused.
    void display_changed();
    void set_view(const ViewSettings& view);

    void set_paused(bool paused);
    bool paused() const { return paused_; }
    // Queues n single generations; used while paused.
    void step(int n = 1);
    void set_steps_per_frame(int n);

    // How the neighborhood U is computed. Direct (the default) sums the kernel per cell. Fft
    // (shaders/fft_convolve.comp) is used for a generation when the world's sides are powers of
    // two the device's FFT supports and the kernel has at least kFftMinTaps nonzero weights;
    // otherwise that generation runs Direct. Results agree up to float rounding. The kernel's
    // spectrum is recomputed only when the kernel (by identity, see GenerationInputs) or the
    // world size changes.
    void set_convolution(lenia::ConvolutionMethod method);
    lenia::ConvolutionMethod convolution() const { return convolution_; }
    // Below this many nonzero kernel weights Direct is cheap; an estimate, not a measurement:
    // lenia_bench shows where FFT overtakes Direct on a given device.
    static constexpr std::size_t kFftMinTaps = 100;

    // Brush for left-drag painting: cells within `radius` get `value` (shift paints 0).
    void set_brush(int radius, float value);
    void set_brush_field(BrushField field) { brush_field_ = field; }
    void fit_view();

    QSize world_size() const { return {world_width_, world_height_}; }
    std::uint64_t generation() const { return generation_; }

    static constexpr int kMaxKernelSize = 2 * 64 + 1;

signals:
    void generation_changed(quint64 generation);
    void device_ready(const QString& description);
    void vulkan_failed(const QString& message);
    // The convolution actually used changed: true once generations run through the FFT path,
    // false when they run Direct again (see set_convolution).
    void convolution_used(bool fft);
    // Unmodified key presses, for the main window's shortcuts.
    void key_pressed(int key);
    // Key presses with Ctrl held (Ctrl+O, Ctrl+S, …), for the main window's menu shortcuts.
    void shortcut_pressed(const QKeySequence& keys);
    // Alt + left click on cell (x, y) of the world: the user wants to inspect it.
    void cell_probed(int x, int y);

protected:
    void keyPressEvent(QKeyEvent* e) override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;

private:
    friend class LeniaRenderer;

    // Grid placement in framebuffer pixels (swapchain size, not logical size).
    struct Placement {
        QPointF origin;          // pixel of the grid's top-left corner
        double pixels_per_cell;
    };
    Placement placement(QSizeF pixels) const;
    QPointF to_cell(QPointF logical_pos) const;
    void paint_line(QPointF from_cell, QPointF to_cell, float value);
    void paint_disk(int cx, int cy, float value);
    // Hands the capture to its callback (taken out first, so it may request another).
    void finish_capture(WorldCapture capture);

    // Edits waiting for the next frame. A world upload or impulse seed supersedes earlier
    // pokes; pokes are keyed by buffer index (a density cell, or width * height plus a nutrient
    // cell) so the last value written to a cell wins.
    struct Pending {
        std::optional<lenia::Grid> world;
        std::optional<lenia::Grid> nutrient;
        bool impulse = false;
        std::map<std::uint32_t, float> pokes;
        int steps = 0;
        bool analyze = true; // recompute neighborhood / growth without stepping
    };
    Pending pending_;

    InputsProvider inputs_;
    DisplayProvider display_;
    ViewSettings view_;
    bool paused_ = false;
    int steps_per_frame_ = 1;
    lenia::ConvolutionMethod convolution_ = lenia::ConvolutionMethod::Direct;
    int world_width_ = 256;
    int world_height_ = 256;
    std::uint64_t generation_ = 0;
    CaptureCallback capture_; // pending capture_world, served by the next frame
    std::uint64_t readbacks_ = 0;

    int brush_radius_ = 4;
    float brush_value_ = 1.0f;
    BrushField brush_field_ = BrushField::Density;

    double zoom_ = 1.0;        // 1 fits the world in the window
    QPointF center_{128, 128}; // world cell at the window center
    enum class Drag { None, Paint, Pan } drag_ = Drag::None;
    QPointF last_mouse_;       // logical pixels
    QPointF last_cell_;
    float drag_value_ = 1.0f;
};

} // namespace workbench
