#include "main_window.hpp"

#include <QApplication>
#include <QLoggingCategory>
#include <QMessageBox>
#include <QPalette>
#include <QStyleFactory>
#include <QStyleHints>
#include <QToolTip>
#include <QVulkanInstance>

namespace {

// Dark Fusion theme for every widget; NodeCanvas (node_editor.cpp) uses matching colors.
void apply_dark_theme() {
    QApplication::setStyle(QStyleFactory::create(QStringLiteral("Fusion")));
#if QT_VERSION >= QT_VERSION_CHECK(6, 8, 0)
    QGuiApplication::styleHints()->setColorScheme(Qt::ColorScheme::Dark); // dark title bars
#endif

    const QColor window(45, 47, 52);
    const QColor base(30, 31, 35);
    const QColor text(225, 225, 228);
    const QColor disabled(120, 122, 128);
    QPalette p;
    p.setColor(QPalette::Window, window);
    p.setColor(QPalette::WindowText, text);
    p.setColor(QPalette::Base, base);
    p.setColor(QPalette::AlternateBase, QColor(40, 42, 47));
    p.setColor(QPalette::ToolTipBase, QColor(24, 25, 28));
    p.setColor(QPalette::ToolTipText, text);
    p.setColor(QPalette::PlaceholderText, QColor(140, 143, 150));
    p.setColor(QPalette::Text, text);
    p.setColor(QPalette::Button, QColor(56, 59, 65));
    p.setColor(QPalette::ButtonText, text);
    p.setColor(QPalette::BrightText, QColor(255, 96, 96));
    p.setColor(QPalette::Light, QColor(78, 81, 88));
    p.setColor(QPalette::Midlight, QColor(64, 67, 73));
    p.setColor(QPalette::Mid, QColor(86, 89, 96));
    p.setColor(QPalette::Dark, QColor(24, 25, 28));
    p.setColor(QPalette::Shadow, QColor(10, 10, 12));
    p.setColor(QPalette::Highlight, QColor(62, 126, 200));
    p.setColor(QPalette::HighlightedText, Qt::white);
    p.setColor(QPalette::Link, QColor(110, 165, 240));
    p.setColor(QPalette::LinkVisited, QColor(160, 130, 230));
    for (QPalette::ColorRole role : {QPalette::WindowText, QPalette::Text, QPalette::ButtonText}) {
        p.setColor(QPalette::Disabled, role, disabled);
    }
    p.setColor(QPalette::Disabled, QPalette::Highlight, QColor(70, 72, 78));
    p.setColor(QPalette::Disabled, QPalette::HighlightedText, disabled);
    QApplication::setPalette(p);
    QToolTip::setPalette(p);
}

} // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QApplication::setApplicationName(QStringLiteral("Lenia kernel workbench"));
    apply_dark_theme();

    QVulkanInstance instance;
    // Set LENIA_VK_VALIDATION=1 to enable the Khronos validation layer (Vulkan SDK).
    if (qEnvironmentVariableIntValue("LENIA_VK_VALIDATION") != 0) {
        QLoggingCategory::setFilterRules(QStringLiteral("qt.vulkan=true"));
        instance.setLayers({"VK_LAYER_KHRONOS_validation"});
    }
    if (!instance.create()) {
        QMessageBox::critical(nullptr, QApplication::applicationName(),
                              QObject::tr("Could not create a Vulkan instance (VkResult %1). "
                                          "Is a Vulkan driver installed?")
                                  .arg(static_cast<int>(instance.errorCode())));
        return 1;
    }

    workbench::MainWindow window(&instance);
    window.show();
    return app.exec();
}
