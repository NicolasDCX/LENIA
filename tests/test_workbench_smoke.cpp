// Qt + Vulkan smoke test of the real workbench window (MainWindow, LeniaView, the GPU step and
// the display shaders), driven through the same entry points as the File menu:
//
// - opens a known, time-modulated experiment (Orbium under the breathing kernel graph): it is
//   paused at generation 0 and the GPU world equals the CPU seed bit for bit;
// - steps it, then changes the display graph while paused: the rendered colors change
//   substantially, while generation and density stay bit-identical;
// - saves a snapshot, steps on, checks a damaged snapshot is refused without touching the world,
//   opens the snapshot (paused at its generation even if running) and steps it again: the result
//   equals the original continuation bit for bit, and differs if the generation is not restored;
// - plays for a while: playback alone performs no GPU readbacks.
//
// Needs a desktop session and a Vulkan device; exits with 77 (skipped) without them.

#include "main_window.hpp"

#include "lenia/display_graph.hpp"
#include "lenia/node_graph.hpp"
#include "lenia/pattern.hpp"
#include "lenia/project.hpp"
#include "lenia/snapshot.hpp"

#include <QApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QImage>
#include <QMessageBox>
#include <QSet>
#include <QTemporaryDir>
#include <QThread>
#include <QTimer>
#include <QVulkanInstance>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

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

// Runs the event loop (frames render, signals arrive) until `done` or the timeout.
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

int captures = 0; // completed capture_world calls, to check LeniaView::readback_count

// The GPU world through LeniaView::capture_world.
std::optional<workbench::WorldCapture> capture(workbench::LeniaView* view, const char* what) {
    // Shared, so a capture that completes after a timeout does not write to a dead frame.
    auto result = std::make_shared<std::optional<workbench::WorldCapture>>();
    view->capture_world([result](const workbench::WorldCapture& c) { *result = c; });
    if (!wait_until([&] { return result->has_value(); })) {
        std::printf("FAILED: capture (%s) timed out\n", what);
        ++failures;
        return std::nullopt;
    }
    if (!(*result)->error.isEmpty()) {
        std::printf("FAILED: capture (%s): %s\n", what, qPrintable((*result)->error));
        ++failures;
        return std::nullopt;
    }
    ++captures;
    return *result;
}

bool wait_generation(workbench::LeniaView* view, std::uint64_t g) {
    const bool ok = wait_until([&] { return view->generation() == g; });
    if (!ok) {
        std::printf("FAILED: generation %llu not reached (at %llu)\n", static_cast<unsigned long long>(g),
                    static_cast<unsigned long long>(view->generation()));
        ++failures;
    }
    return ok;
}

// Fraction of pixels whose color differs by more than `tolerance` in some channel.
double changed_fraction(const QImage& a, const QImage& b, int tolerance) {
    if (a.size() != b.size() || a.isNull()) return 0.0;
    const QImage x = a.convertToFormat(QImage::Format_RGB32);
    const QImage y = b.convertToFormat(QImage::Format_RGB32);
    long long changed = 0;
    for (int row = 0; row < x.height(); ++row) {
        const auto* p = reinterpret_cast<const QRgb*>(x.constScanLine(row));
        const auto* q = reinterpret_cast<const QRgb*>(y.constScanLine(row));
        for (int col = 0; col < x.width(); ++col) {
            const int d = std::max({std::abs(qRed(p[col]) - qRed(q[col])), std::abs(qGreen(p[col]) - qGreen(q[col])),
                                    std::abs(qBlue(p[col]) - qBlue(q[col]))});
            if (d > tolerance) ++changed;
        }
    }
    return static_cast<double>(changed) / (static_cast<double>(x.width()) * x.height());
}

int distinct_colors(const QImage& image) {
    const QImage x = image.convertToFormat(QImage::Format_RGB32);
    QSet<QRgb> colors;
    for (int row = 0; row < x.height(); row += 4) {
        const auto* p = reinterpret_cast<const QRgb*>(x.constScanLine(row));
        for (int col = 0; col < x.width(); col += 4) colors.insert(p[col] & 0xffffff);
    }
    return static_cast<int>(colors.size());
}

bool write_bytes(const QString& path, const QByteArray& bytes) {
    QFile f(path);
    return f.open(QIODevice::WriteOnly) && f.write(bytes) == bytes.size();
}

QByteArray read_bytes(const QString& path) {
    QFile f(path);
    return f.open(QIODevice::ReadOnly) ? f.readAll() : QByteArray();
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

    // The workbench reports problems in message boxes; any that opens is a failure, closed so
    // the test goes on (timers also fire inside a dialog's own event loop).
    QTimer dialog_watch;
    QObject::connect(&dialog_watch, &QTimer::timeout, [] {
        if (QWidget* modal = QApplication::activeModalWidget()) {
            const auto* box = qobject_cast<QMessageBox*>(modal);
            std::printf("FAILED: the workbench opened a dialog: %s\n",
                        qPrintable(box ? box->text() : modal->windowTitle()));
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
    QObject::connect(view, &LeniaView::device_ready, [&](const QString& d) {
        device_ready = true;
        std::printf("device: %s\n", qPrintable(d));
    });
    QObject::connect(view, &LeniaView::vulkan_failed,
                     [](const QString& m) { std::printf("Vulkan failure: %s\n", qPrintable(m)); });
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

    // --- A known, paused, time-modulated world. ----------------------------------------------
    lenia::Project known;
    known.kernel = orbium.kernel;
    known.growth = orbium.growth;
    known.world_size = 128;
    known.seed = lenia::SeedKind::Pattern;
    known.pattern = orbium;
    known.graph = lenia::make_preset(lenia::GraphPreset::BreathingKernel);
    known.graph.set_param(4, lenia::environment_node::Strength, 0.01f);
    expect(!(lenia::evaluate(known.graph, 7).kernel == lenia::evaluate(known.graph, 0).kernel),
           "setup: the graph warps the kernel differently per generation");
    const QString experiment_path = dir.filePath(QStringLiteral("known.leniaexp"));
    expect(write_bytes(experiment_path, QByteArray::fromStdString(lenia::write_project(known))), "setup: experiment written");

    const QString open_error = window.open_experiment_file(experiment_path);
    expect(open_error.isEmpty(), "open the known experiment");
    if (!open_error.isEmpty()) std::printf("  %s\n", qPrintable(open_error));
    expect(view->paused() && view->generation() == 0, "it opens paused at generation 0");

    lenia::Grid seed(128, 128, 0.0f);
    lenia::stamp_pattern(seed, orbium.cells, 64, 64);
    if (const auto c0 = capture(view, "seed")) {
        expect(c0->generation == 0 && same(c0->density, seed), "GPU world at generation 0 equals the CPU seed");
    }

    view->step(7);
    wait_generation(view, 7);
    const std::optional<WorldCapture> at7 = capture(view, "generation 7");
    if (!at7) return 1;
    expect(at7->generation == 7 && !same(at7->density, seed), "stepped to generation 7 on the GPU");

    // --- The display graph recolors the paused world without touching it. -------------------
    bool grab_checked = false;
    if (view->supportsGrab()) {
        const QImage before = view->grab();
        window.load_display_preset(lenia::DisplayPreset::GrowthOverlay);
        settle(200);
        const QImage after = view->grab();
        const int colors = distinct_colors(before);
        const double changed = changed_fraction(before, after, 24);
        std::printf("viewport %dx%d, %d distinct colors before, %.1f%% of pixels recolored\n", before.width(),
                    before.height(), colors, 100.0 * changed);
        expect(!before.isNull() && colors >= 8, "the viewport renders the world (not blank)");
        expect(changed > 0.10, "the display graph change recolors the viewport");
        grab_checked = true;
    } else {
        std::printf("note: the swapchain does not support grab(); rendered colors not compared\n");
        window.load_display_preset(lenia::DisplayPreset::GrowthOverlay);
        settle(200);
    }
    expect(view->paused() && view->generation() == 7, "still paused at generation 7 after the display change");
    if (const auto again = capture(view, "after the display change")) {
        expect(again->generation == 7 && same(again->density, at7->density),
               "density bit-identical after the display change");
    }

    // --- Snapshot: save, continue, refuse a damaged one, reload and continue identically. ----
    const QString snap_path = dir.filePath(QStringLiteral("gen7.leniasnap"));
    expect(write_bytes(snap_path, "old contents"), "setup: a file to replace");
    std::optional<QString> saved;
    window.save_snapshot_file(snap_path, [&](const QString& error) { saved = error; });
    expect(wait_until([&] { return saved.has_value(); }) && saved->isEmpty(), "snapshot saved");
    if (saved && !saved->isEmpty()) std::printf("  %s\n", qPrintable(*saved));
    const QByteArray snap_bytes = read_bytes(snap_path);
    try {
        const lenia::Snapshot s = lenia::parse_snapshot(
            std::string_view(snap_bytes.constData(), static_cast<std::size_t>(snap_bytes.size())));
        expect(s.generation == 7 && same(s.density, at7->density), "the file holds generation 7's exact density");
        expect(lenia::compile_display(s.project.display_graph, 7).linked, "the file holds the edited display graph");
        expect(s.project.graph.nodes().size() == known.graph.nodes().size(), "the file holds the experiment's graph");
    } catch (const std::exception& e) {
        std::printf("FAILED: the saved snapshot does not parse: %s\n", e.what());
        ++failures;
    }
    expect(QDir(dir.path()).entryList(QDir::Files).size() == 2, "atomic save left no temporary files behind");

    view->step(5);
    wait_generation(view, 12);
    const std::optional<WorldCapture> at12 = capture(view, "generation 12");
    if (!at12) return 1;

    QByteArray damaged = snap_bytes;
    damaged[damaged.size() / 2] = static_cast<char>(damaged[damaged.size() / 2] ^ 0x40);
    const QString damaged_path = dir.filePath(QStringLiteral("damaged.leniasnap"));
    write_bytes(damaged_path, damaged);
    const QString damaged_error = window.open_snapshot_file(damaged_path);
    std::printf("damaged snapshot: %s\n", qPrintable(damaged_error));
    expect(!damaged_error.isEmpty(), "a damaged snapshot is refused");
    if (const auto c = capture(view, "after the refused snapshot")) {
        expect(c->generation == 12 && same(c->density, at12->density), "the refused snapshot left the world unchanged");
    }

    // Playback alone never reads the world back.
    const std::uint64_t readbacks = view->readback_count();
    window.set_playing(true);
    expect(wait_until([&] { return view->generation() >= 40; }), "playback runs");
    expect(view->readback_count() == readbacks, "no GPU readbacks during playback");

    // Opening while running: paused at the snapshot's generation, and it stays there.
    const QString snap_error = window.open_snapshot_file(snap_path);
    expect(snap_error.isEmpty(), "open the snapshot");
    if (!snap_error.isEmpty()) std::printf("  %s\n", qPrintable(snap_error));
    expect(view->paused() && view->generation() == 7, "the snapshot opens paused at generation 7");
    settle(300);
    if (const auto c = capture(view, "loaded snapshot")) {
        expect(c->generation == 7 && same(c->density, at7->density), "the loaded world is generation 7's, bit for bit");
    }
    view->step(5);
    wait_generation(view, 12);
    if (const auto c = capture(view, "snapshot continued")) {
        expect(c->generation == 12 && same(c->density, at12->density),
               "the snapshot continued to 12 equals the original run bit for bit");
    }

    // The restored generation matters: the same world stepped from generation 0 diverges.
    view->upload_world(at7->density, 0);
    view->step(5);
    wait_generation(view, 5);
    if (const auto c = capture(view, "generation not restored")) {
        expect(!same(c->density, at12->density), "stepping the same world from generation 0 diverges");
    }

    // FFT convolution: one generation from the same world matches Direct up to float rounding
    // (one only: Orbium's steep growth compounds rounding), and the view reports using it. The
    // breathing kernel is a new kernel every generation, so its spectrum is recomputed each time.
    {
        bool fft_reported = false;
        const QMetaObject::Connection watch =
            QObject::connect(view, &LeniaView::convolution_used, [&](bool fft) { fft_reported = fft_reported || fft; });
        view->upload_world(at7->density, 7);
        view->step(1);
        wait_generation(view, 8);
        const std::optional<WorldCapture> direct8 = capture(view, "direct generation 8");
        view->set_convolution(lenia::ConvolutionMethod::Fft);
        view->upload_world(at7->density, 7);
        view->step(1);
        wait_generation(view, 8);
        const std::optional<WorldCapture> fft8 = capture(view, "FFT generation 8");
        if (direct8 && fft8 && fft8->density.cells.size() == direct8->density.cells.size()) {
            float max_diff = 0.0f;
            for (std::size_t i = 0; i < fft8->density.cells.size(); ++i) {
                max_diff = std::max(max_diff, std::fabs(fft8->density.cells[i] - direct8->density.cells[i]));
            }
            std::printf("FFT vs direct, generation 7 -> 8: max difference %.3g\n", static_cast<double>(max_diff));
            expect(max_diff <= 1e-4f, "an FFT generation matches a direct one up to float rounding");
        } else {
            expect(false, "an FFT generation matches a direct one up to float rounding");
        }
        expect(wait_until([&] { return fft_reported; }, 5000), "the view reports using the FFT");
        view->step(3); // breathing kernel: a new spectrum each generation
        expect(wait_until([&] { return view->generation() == 11; }), "FFT keeps stepping with a changing kernel");
        view->set_convolution(lenia::ConvolutionMethod::Direct);
        QObject::disconnect(watch);
    }

    // Saving the snapshot also requests one capture through the public window action.
    expect(view->readback_count() == static_cast<std::uint64_t>(captures) + 1,
           "one GPU readback per explicit capture and one for the snapshot save");

    window.close();
    settle(100);
    if (failures == 0) {
        std::printf(grab_checked ? "All workbench smoke tests passed.\n"
                                 : "Workbench smoke tests passed, rendered colors not checked (skipped).\n");
        return grab_checked ? 0 : kSkip;
    }
    std::printf("%d failure(s).\n", failures);
    return 1;
}
