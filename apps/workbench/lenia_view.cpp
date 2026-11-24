#include "lenia_view.hpp"

#include "palettes.hpp"

#include <QKeyEvent>
#include <QMouseEvent>
#include <QVulkanDeviceFunctions>
#include <QVulkanFunctions>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

namespace workbench {

namespace {

// SPIR-V words generated at build time from shaders/.
const std::uint32_t kStepCode[] = {
#include "workbench_step.spv.inc"
};
const std::uint32_t kFftCode[] = {
#include "fft_convolve.spv.inc"
};
const std::uint32_t kVertCode[] = {
#include "workbench_view.vert.spv.inc"
};
const std::uint32_t kFragCode[] = {
#include "workbench_view.frag.spv.inc"
};

// Must match local_size_x / local_size_y in workbench_step.comp.
constexpr std::uint32_t kLocalSize = 16;
// vkCmdUpdateBuffer accepts at most 65536 bytes per call.
constexpr VkDeviceSize kUpdateChunk = 65536;

// Must match the push_constant block in workbench_step.comp.
struct StepPush {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t kwidth;
    std::uint32_t kheight;
    float mu;
    float sigma;
    float dt;
    std::uint32_t advance;
    std::uint32_t growth;      // lenia::GrowthFunction
    std::uint32_t env_pattern; // lenia::EnvironmentPattern + 1, 0 = none
    float env_strength;
    float env_scale;
    float env_cos;
    float env_sin;
    float env_offset;
    std::uint32_t rule;    // 1: the Rule buffer gives the next state
    std::uint32_t u_ready; // 1: the Neighborhood buffer already holds U (FFT path)
    std::uint32_t preview; // rule register shown as the neighborhood, or lenia::kNoRegister
};

// The kernel buffer (binding 1): the active kernel at 0, up to kMaxKernelSize² weights, then
// the rule's Convolve kernels (lenia::pack_rule_kernels) at this fixed offset, in floats.
constexpr std::uint32_t kRuleKernelBase =
    static_cast<std::uint32_t>(LeniaView::kMaxKernelSize) * LeniaView::kMaxKernelSize;

// fft_convolve.comp: threads per line (local_size_x_id 1) if the device allows them, and its
// specialization constants (FFT_MAX, id 0, and the threads).
constexpr std::uint32_t kFftThreads = 256;
struct FftSpecialization {
    std::uint32_t max_size;
    std::uint32_t threads;
};

// The Rule buffer (binding 5): a lenia::pack_rule program of up to kMaxRuleOps ops.
constexpr VkDeviceSize kRuleBytes = lenia::kRuleBufferWords * sizeof(std::uint32_t);

bool valid_kernel(const lenia::Grid* k) {
    constexpr auto kMax = static_cast<std::size_t>(LeniaView::kMaxKernelSize);
    return k && k->width % 2 == 1 && k->height % 2 == 1 && k->width <= kMax && k->height <= kMax &&
           k->cells.size() == k->width * k->height;
}

// Fits the Rule buffer and the kernel buffer's space for Convolve kernels.
bool valid_rule(const lenia::RuleProgram& r) {
    if (r.ops.size() > lenia::kMaxRuleOps || r.kernels.size() > lenia::kMaxConvolutions) return false;
    std::size_t weights = 0;
    for (const lenia::Grid& k : r.kernels) {
        if (k.width % 2 == 0 || k.height % 2 == 0 || k.cells.size() != k.width * k.height) return false;
        weights += k.cells.size();
    }
    return weights <= lenia::kMaxConvolveWeights;
}

// Must match the push_constant block in workbench_view.frag.
struct ViewPush {
    float origin[2];
    float cells_per_pixel;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t field;
    std::uint32_t palette;
    float lo;
    float hi;
    std::uint32_t linear_target;
    std::uint32_t program; // 1: the Display buffer's program colors the cells
};

// The Display buffer (view binding 4): a lenia::pack_display program.
constexpr VkDeviceSize kDisplayBytes = lenia::kDisplayBufferWords * sizeof(std::uint32_t);

bool valid_display(const lenia::DisplayProgram& p) {
    return p.ops.size() <= lenia::kMaxDisplayOps && p.stops.size() <= lenia::kMaxDisplayStops;
}

bool is_srgb(VkFormat format) {
    return format == VK_FORMAT_B8G8R8A8_SRGB || format == VK_FORMAT_R8G8B8A8_SRGB ||
           format == VK_FORMAT_A8B8G8R8_SRGB_PACK32;
}

} // namespace

class LeniaRenderer : public QVulkanWindowRenderer {
public:
    explicit LeniaRenderer(LeniaView* view) : view_(view) {}

    void initResources() override;
    void initSwapChainResources() override;
    void releaseResources() override;
    void startNextFrame() override;

private:
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
    };

    bool check(VkResult result, const char* what);
    void fail(const QString& message);
    // Device-local storage buffer (transfer source too, for captures); with `host_readback` a
    // host-visible transfer destination the CPU maps instead.
    bool create_buffer(Buffer& b, VkDeviceSize size, bool host_readback = false);
    void destroy_buffer(Buffer& b);
    VkShaderModule create_shader(const std::uint32_t* code, std::size_t bytes);
    bool create_pipelines();
    bool create_world(int width, int height);
    void destroy_world();
    void write_descriptors();
    // The optional FFT path: failures only disable it (with a warning), never the renderer.
    void create_fft_pipeline();
    void destroy_fft();
    // Spectrum buffers for a world the FFT supports; none otherwise.
    void create_fft_world();
    // Whether this generation's neighborhood goes through the FFT path.
    bool fft_enabled(const std::shared_ptr<const lenia::Grid>& kernel);
    // U for the generation reading cells_[cur_] into neighborhood_, transforming `kernel` first
    // if khat_ holds another one's spectrum.
    void record_fft(VkCommandBuffer cb, const std::shared_ptr<const lenia::Grid>& kernel);

    void barrier(VkCommandBuffer cb, VkPipelineStageFlags src_stage, VkAccessFlags src_access,
                 VkPipelineStageFlags dst_stage, VkAccessFlags dst_access);
    void update(VkCommandBuffer cb, const Buffer& b, VkDeviceSize offset, const void* data, VkDeviceSize bytes);
    void record_uploads(VkCommandBuffer cb);
    void record_steps(VkCommandBuffer cb);
    // Uploads this frame's display program if it changed (outside the render pass) and returns
    // whether the draw uses it.
    bool record_display(VkCommandBuffer cb);
    void record_draw(VkCommandBuffer cb, bool program);
    // A pending capture_world: copies the current density into staging_ after this frame's
    // uploads and steps. Returns false (and fails the capture) if it cannot.
    bool record_capture(VkCommandBuffer cb);
    // After the frame is submitted: waits for the queue, reads staging_ and hands the result on.
    void finish_capture();

    LeniaView* view_;
    QVulkanDeviceFunctions* df_ = nullptr;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory_props_{};
    bool failed_ = false;
    bool linear_target_ = false;

    // cells_[cur_] is the current density; the step writes cells_[1 - cur_].
    Buffer cells_[2], neighborhood_, growth_, kernel_, palette_, rule_, display_;
    int width_ = 0;
    int height_ = 0;
    int cur_ = 0;
    // The kernel now in kernel_ and the program in rule_ (by identity, see GenerationInputs);
    // null after device loss.
    std::shared_ptr<const lenia::Grid> uploaded_kernel_;
    std::shared_ptr<const lenia::RuleProgram> uploaded_rule_;
    std::shared_ptr<const lenia::DisplayProgram> uploaded_display_; // the program in display_
    bool warned_bad_rule_ = false;
    bool warned_bad_display_ = false;
    bool warned_bad_kernel_ = false;
    bool clear_world_ = false;
    bool upload_palette_ = false;

    // Host-visible copy of the density, alive only between record_capture and finish_capture.
    Buffer staging_;
    bool staging_coherent_ = true;
    int capture_width_ = 0;
    int capture_height_ = 0;
    std::uint64_t capture_generation_ = 0;
    // While set, fail() records the message here instead of failing the renderer.
    QString* soft_error_ = nullptr;

    VkDescriptorSetLayout step_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout view_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout step_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout view_layout_ = VK_NULL_HANDLE;
    VkPipeline step_pipeline_ = VK_NULL_HANDLE;
    VkPipeline view_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    // step_sets_[i] reads cells_[i] and writes cells_[1 - i]; view_sets_[i] draws cells_[i].
    VkDescriptorSet step_sets_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet view_sets_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};

    // FFT convolution (fft_convolve.comp, LeniaView::set_convolution); a null pipeline if the
    // device cannot run it. fft_sets_[i] reads cells_[i] and writes U into neighborhood_.
    VkDescriptorSetLayout fft_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout fft_layout_ = VK_NULL_HANDLE;
    VkPipeline fft_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool fft_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet fft_sets_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::size_t fft_max_ = 0;        // longest side the pipeline handles
    Buffer spec_, khat_, twiddles_;  // spec_ and khat_ exist only for worlds the FFT supports
    VkDeviceSize spectrum_bytes_ = 0;
    bool upload_twiddles_ = false;
    std::shared_ptr<const lenia::Grid> fft_kernel_; // whose spectrum khat_ holds; null: none
    std::shared_ptr<const lenia::Grid> counted_kernel_; // the kernel counted_taps_ was counted for
    std::size_t counted_taps_ = 0;
    bool fft_in_use_ = false; // what convolution_used last reported
};

// ---------------------------------------------------------------------------------------------
// LeniaRenderer: resources

void LeniaRenderer::fail(const QString& message) {
    if (soft_error_) { // a capture's own resources: fail the capture, not the renderer
        if (soft_error_->isEmpty()) *soft_error_ = message;
        return;
    }
    if (!failed_) {
        failed_ = true;
        emit view_->vulkan_failed(message);
    }
    if (view_->capture_) view_->finish_capture({{}, 0, message});
}

bool LeniaRenderer::check(VkResult result, const char* what) {
    if (result == VK_SUCCESS) return true;
    fail(QStringLiteral("%1 failed (VkResult %2)").arg(QLatin1String(what)).arg(static_cast<int>(result)));
    return false;
}

void LeniaRenderer::initResources() {
    failed_ = false;
    QVulkanInstance* inst = view_->vulkanInstance();
    dev_ = view_->device();
    df_ = inst->deviceFunctions(dev_);
    QVulkanFunctions* f = inst->functions();

    // Compute is recorded into the frame's command buffer, so the graphics queue must do both.
    // Vulkan guarantees some family does, but QVulkanWindow picks the first graphics family.
    std::uint32_t family_count = 0;
    f->vkGetPhysicalDeviceQueueFamilyProperties(view_->physicalDevice(), &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    f->vkGetPhysicalDeviceQueueFamilyProperties(view_->physicalDevice(), &family_count, families.data());
    const std::uint32_t family = view_->graphicsQueueFamilyIndex();
    if (family >= family_count || (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
        fail(QStringLiteral("The graphics queue of %1 does not support compute.")
                 .arg(QString::fromUtf8(view_->physicalDeviceProperties()->deviceName)));
        return;
    }
    f->vkGetPhysicalDeviceMemoryProperties(view_->physicalDevice(), &memory_props_);

    if (!create_pipelines()) return;
    create_fft_pipeline();

    const VkDeviceSize kernel_bytes = (static_cast<VkDeviceSize>(kRuleKernelBase) + lenia::kMaxConvolveWeights) * sizeof(float);
    const std::vector<float> palettes = palette_table();
    if (!create_buffer(kernel_, kernel_bytes)) return;
    if (!create_buffer(palette_, palettes.size() * sizeof(float))) return;
    if (!create_buffer(rule_, kRuleBytes)) return;
    if (!create_buffer(display_, kDisplayBytes)) return;
    if (!create_world(view_->world_width_, view_->world_height_)) return;
    write_descriptors();
    upload_palette_ = true;
    // A recreated device has lost the kernel, rule and display program (re-uploaded when next
    // used); the world was cleared by create_world.
    uploaded_kernel_.reset();
    uploaded_rule_.reset();
    uploaded_display_.reset();
    view_->pending_.analyze = true;

    const VkPhysicalDeviceProperties* props = view_->physicalDeviceProperties();
    emit view_->device_ready(QStringLiteral("%1, queue family %2 (graphics + compute)")
                                 .arg(QString::fromUtf8(props->deviceName))
                                 .arg(family));
}

void LeniaRenderer::initSwapChainResources() { linear_target_ = is_srgb(view_->colorFormat()); }

VkShaderModule LeniaRenderer::create_shader(const std::uint32_t* code, std::size_t bytes) {
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = bytes;
    info.pCode = code;
    VkShaderModule module = VK_NULL_HANDLE;
    check(df_->vkCreateShaderModule(dev_, &info, nullptr, &module), "vkCreateShaderModule");
    return module;
}

bool LeniaRenderer::create_pipelines() {
    // Set layouts: compute reads src, ker and writes dst, neighborhood, growth, and reads the
    // rule program; the fragment shader reads density, neighborhood, growth, palettes and the
    // display program. All are storage buffers.
    const auto make_set_layout = [&](std::uint32_t count, VkShaderStageFlags stage, VkDescriptorSetLayout* out) {
        std::vector<VkDescriptorSetLayoutBinding> bindings(count);
        for (std::uint32_t b = 0; b < count; ++b) {
            bindings[b] = {};
            bindings[b].binding = b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = stage;
        }
        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = count;
        info.pBindings = bindings.data();
        return check(df_->vkCreateDescriptorSetLayout(dev_, &info, nullptr, out), "vkCreateDescriptorSetLayout");
    };
    if (!make_set_layout(6, VK_SHADER_STAGE_COMPUTE_BIT, &step_set_layout_)) return false;
    if (!make_set_layout(5, VK_SHADER_STAGE_FRAGMENT_BIT, &view_set_layout_)) return false;

    const auto make_layout = [&](VkDescriptorSetLayout set_layout, VkShaderStageFlags stage, std::uint32_t push_size,
                                 VkPipelineLayout* out) {
        VkPushConstantRange range{stage, 0, push_size};
        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        info.setLayoutCount = 1;
        info.pSetLayouts = &set_layout;
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges = &range;
        return check(df_->vkCreatePipelineLayout(dev_, &info, nullptr, out), "vkCreatePipelineLayout");
    };
    if (!make_layout(step_set_layout_, VK_SHADER_STAGE_COMPUTE_BIT, sizeof(StepPush), &step_layout_)) return false;
    if (!make_layout(view_set_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(ViewPush), &view_layout_)) return false;

    // Compute pipeline.
    VkShaderModule step_module = create_shader(kStepCode, sizeof(kStepCode));
    if (!step_module) return false;
    VkComputePipelineCreateInfo compute{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    compute.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compute.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compute.stage.module = step_module;
    compute.stage.pName = "main";
    compute.layout = step_layout_;
    const bool compute_ok = check(
        df_->vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &compute, nullptr, &step_pipeline_),
        "vkCreateComputePipelines");
    df_->vkDestroyShaderModule(dev_, step_module, nullptr);
    if (!compute_ok) return false;

    // Graphics pipeline: one fullscreen triangle into QVulkanWindow's default render pass.
    VkShaderModule vert = create_shader(kVertCode, sizeof(kVertCode));
    VkShaderModule frag = create_shader(kFragCode, sizeof(kFragCode));
    bool graphics_ok = vert && frag;
    if (graphics_ok) {
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName = "main";
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = view_->sampleCountFlagBits();
        VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState blend_attachment{};
        blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1;
        blend.pAttachments = &blend_attachment;
        const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = 2;
        dynamic.pDynamicStates = dynamic_states;

        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        info.stageCount = 2;
        info.pStages = stages;
        info.pVertexInputState = &vertex_input;
        info.pInputAssemblyState = &assembly;
        info.pViewportState = &viewport;
        info.pRasterizationState = &raster;
        info.pMultisampleState = &multisample;
        info.pDepthStencilState = &depth;
        info.pColorBlendState = &blend;
        info.pDynamicState = &dynamic;
        info.layout = view_layout_;
        info.renderPass = view_->defaultRenderPass();
        graphics_ok = check(df_->vkCreateGraphicsPipelines(dev_, VK_NULL_HANDLE, 1, &info, nullptr, &view_pipeline_),
                            "vkCreateGraphicsPipelines");
    }
    if (vert) df_->vkDestroyShaderModule(dev_, vert, nullptr);
    if (frag) df_->vkDestroyShaderModule(dev_, frag, nullptr);
    if (!graphics_ok) return false;

    // Two sets each, for the two ping-pong directions.
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * 6 + 2 * 5};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 4;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (!check(df_->vkCreateDescriptorPool(dev_, &pool_info, nullptr, &pool_), "vkCreateDescriptorPool")) {
        return false;
    }
    const VkDescriptorSetLayout layouts[4] = {step_set_layout_, step_set_layout_, view_set_layout_, view_set_layout_};
    VkDescriptorSet sets[4];
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = pool_;
    set_info.descriptorSetCount = 4;
    set_info.pSetLayouts = layouts;
    if (!check(df_->vkAllocateDescriptorSets(dev_, &set_info, sets), "vkAllocateDescriptorSets")) return false;
    step_sets_[0] = sets[0];
    step_sets_[1] = sets[1];
    view_sets_[0] = sets[2];
    view_sets_[1] = sets[3];
    return true;
}

bool LeniaRenderer::create_buffer(Buffer& b, VkDeviceSize size, bool host_readback) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = host_readback ? VK_BUFFER_USAGE_TRANSFER_DST_BIT
                               : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!check(df_->vkCreateBuffer(dev_, &info, nullptr, &b.buffer), "vkCreateBuffer")) return false;

    VkMemoryRequirements req;
    df_->vkGetBufferMemoryRequirements(dev_, b.buffer, &req);
    // Storage buffers: device-local if possible, the CPU never maps them, it only records
    // updates. Readback buffers: host-visible, preferably cached (fast CPU reads) and coherent.
    const auto score = [&](VkMemoryPropertyFlags flags) {
        if (!host_readback) return (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 2 : 1;
        if (!(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) return 0;
        return 1 + ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? 2 : 0) +
               ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? 1 : 0);
    };
    std::uint32_t type = memory_props_.memoryTypeCount;
    int best = 0;
    for (std::uint32_t i = 0; i < memory_props_.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) == 0) continue;
        const int s = score(memory_props_.memoryTypes[i].propertyFlags);
        if (s > best) {
            best = s;
            type = i;
        }
    }
    if (type == memory_props_.memoryTypeCount) {
        fail(host_readback ? QStringLiteral("No host-visible memory type for reading the world back")
                           : QStringLiteral("No memory type for a storage buffer"));
        return false;
    }
    if (host_readback) {
        staging_coherent_ = (memory_props_.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    }
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    if (!check(df_->vkAllocateMemory(dev_, &alloc, nullptr, &b.memory), "vkAllocateMemory")) return false;
    return check(df_->vkBindBufferMemory(dev_, b.buffer, b.memory, 0), "vkBindBufferMemory");
}

void LeniaRenderer::destroy_buffer(Buffer& b) {
    if (b.buffer) df_->vkDestroyBuffer(dev_, b.buffer, nullptr);
    if (b.memory) df_->vkFreeMemory(dev_, b.memory, nullptr);
    b = Buffer{};
}

bool LeniaRenderer::create_world(int width, int height) {
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * sizeof(float);
    // Cell buffers hold both state fields: the density, then the nutrient.
    for (Buffer* b : {&cells_[0], &cells_[1]}) {
        if (!create_buffer(*b, 2 * bytes)) return false;
    }
    for (Buffer* b : {&neighborhood_, &growth_}) {
        if (!create_buffer(*b, bytes)) return false;
    }
    width_ = width;
    height_ = height;
    cur_ = 0;
    clear_world_ = true; // new memory is undefined
    create_fft_world();
    return true;
}

void LeniaRenderer::destroy_world() {
    for (Buffer* b : {&cells_[0], &cells_[1], &neighborhood_, &growth_, &spec_, &khat_}) destroy_buffer(*b);
    width_ = height_ = 0;
    spectrum_bytes_ = 0;
    fft_kernel_.reset();
}

void LeniaRenderer::create_fft_pipeline() {
    const VkPhysicalDeviceLimits& limits = view_->physicalDeviceProperties()->limits;
    const std::size_t max_size = lenia::fft_max_size(limits.maxComputeSharedMemorySize);
    if (max_size == 0) return;
    std::uint32_t threads = kFftThreads;
    while (threads > 1 && (threads > limits.maxComputeWorkGroupInvocations || threads > limits.maxComputeWorkGroupSize[0])) {
        threads /= 2;
    }

    QString error;
    soft_error_ = &error;
    bool ok = true;
    VkDescriptorSetLayoutBinding bindings[6]{};
    for (std::uint32_t b = 0; b < 6; ++b) {
        bindings[b].binding = b;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 6;
    set_info.pBindings = bindings;
    ok = check(df_->vkCreateDescriptorSetLayout(dev_, &set_info, nullptr, &fft_set_layout_), "vkCreateDescriptorSetLayout");
    if (ok) {
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(lenia::FftPushConstants)};
        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        info.setLayoutCount = 1;
        info.pSetLayouts = &fft_set_layout_;
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges = &range;
        ok = check(df_->vkCreatePipelineLayout(dev_, &info, nullptr, &fft_layout_), "vkCreatePipelineLayout");
    }
    if (ok) {
        VkShaderModule module = create_shader(kFftCode, sizeof(kFftCode));
        ok = module != VK_NULL_HANDLE;
        if (ok) {
            const FftSpecialization values{static_cast<std::uint32_t>(max_size), threads};
            const VkSpecializationMapEntry entries[2] = {
                {0, offsetof(FftSpecialization, max_size), sizeof(std::uint32_t)},
                {1, offsetof(FftSpecialization, threads), sizeof(std::uint32_t)}};
            VkSpecializationInfo specialization{};
            specialization.mapEntryCount = 2;
            specialization.pMapEntries = entries;
            specialization.dataSize = sizeof(values);
            specialization.pData = &values;
            VkComputePipelineCreateInfo compute{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            compute.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            compute.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            compute.stage.module = module;
            compute.stage.pName = "main";
            compute.stage.pSpecializationInfo = &specialization;
            compute.layout = fft_layout_;
            ok = check(df_->vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &compute, nullptr, &fft_pipeline_),
                       "vkCreateComputePipelines");
            df_->vkDestroyShaderModule(dev_, module, nullptr);
        }
    }
    if (ok) {
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * 6};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 2;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        ok = check(df_->vkCreateDescriptorPool(dev_, &pool_info, nullptr, &fft_pool_), "vkCreateDescriptorPool");
    }
    if (ok) {
        const VkDescriptorSetLayout layouts[2] = {fft_set_layout_, fft_set_layout_};
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = fft_pool_;
        alloc.descriptorSetCount = 2;
        alloc.pSetLayouts = layouts;
        ok = check(df_->vkAllocateDescriptorSets(dev_, &alloc, fft_sets_), "vkAllocateDescriptorSets");
    }
    // Twiddles for the longest line, uploaded with the next frame's edits.
    if (ok) ok = create_buffer(twiddles_, static_cast<VkDeviceSize>(max_size) * sizeof(float));
    soft_error_ = nullptr;
    if (!ok) {
        qWarning("LeniaView: FFT convolution unavailable (%s); using direct convolution", qPrintable(error));
        destroy_fft();
        return;
    }
    fft_max_ = max_size;
    upload_twiddles_ = true;
}

void LeniaRenderer::destroy_fft() {
    destroy_buffer(spec_);
    destroy_buffer(khat_);
    destroy_buffer(twiddles_);
    if (fft_pool_) df_->vkDestroyDescriptorPool(dev_, fft_pool_, nullptr); // frees the sets
    if (fft_pipeline_) df_->vkDestroyPipeline(dev_, fft_pipeline_, nullptr);
    if (fft_layout_) df_->vkDestroyPipelineLayout(dev_, fft_layout_, nullptr);
    if (fft_set_layout_) df_->vkDestroyDescriptorSetLayout(dev_, fft_set_layout_, nullptr);
    fft_pool_ = VK_NULL_HANDLE;
    fft_pipeline_ = VK_NULL_HANDLE;
    fft_layout_ = VK_NULL_HANDLE;
    fft_set_layout_ = VK_NULL_HANDLE;
    fft_sets_[0] = fft_sets_[1] = VK_NULL_HANDLE;
    fft_max_ = 0;
    spectrum_bytes_ = 0;
    upload_twiddles_ = false;
    fft_kernel_.reset();
}

void LeniaRenderer::create_fft_world() {
    fft_kernel_.reset(); // a new khat_, or none
    if (!fft_pipeline_) return;
    const auto w = static_cast<std::size_t>(width_), h = static_cast<std::size_t>(height_);
    if (!lenia::fft_size_supported(w, h, fft_max_)) return;
    const VkPhysicalDeviceLimits& limits = view_->physicalDeviceProperties()->limits;
    const VkDeviceSize bytes = lenia::fft_spectrum_size(w, h) * 2 * sizeof(float);
    if (bytes > limits.maxStorageBufferRange ||
        lenia::fft_pass_groups(lenia::FftPass::ConvolveColumns, w, h) > limits.maxComputeWorkGroupCount[0] ||
        lenia::fft_pass_groups(lenia::FftPass::RowsForward, w, h) > limits.maxComputeWorkGroupCount[0]) {
        return;
    }
    QString error;
    soft_error_ = &error;
    const bool ok = create_buffer(spec_, bytes) && create_buffer(khat_, bytes);
    soft_error_ = nullptr;
    if (!ok) {
        qWarning("LeniaView: no FFT buffers for this world (%s); using direct convolution", qPrintable(error));
        destroy_buffer(spec_);
        destroy_buffer(khat_);
        return;
    }
    spectrum_bytes_ = bytes;
}

bool LeniaRenderer::fft_enabled(const std::shared_ptr<const lenia::Grid>& kernel) {
    if (view_->convolution_ != lenia::ConvolutionMethod::Fft || !fft_pipeline_ || !spec_.buffer) return false;
    if (kernel != counted_kernel_) {
        counted_kernel_ = kernel;
        counted_taps_ = static_cast<std::size_t>(
            std::count_if(kernel->cells.begin(), kernel->cells.end(), [](float w) { return w != 0.0f; }));
    }
    return counted_taps_ >= LeniaView::kFftMinTaps;
}

void LeniaRenderer::record_fft(VkCommandBuffer cb, const std::shared_ptr<const lenia::Grid>& kernel) {
    const auto w = static_cast<std::size_t>(width_), h = static_cast<std::size_t>(height_);
    lenia::FftPushConstants pc{static_cast<std::uint32_t>(w),
                               static_cast<std::uint32_t>(h),
                               lenia::fft_log2(w),
                               lenia::fft_log2(h),
                               lenia::fft_log2(fft_max_),
                               static_cast<std::uint32_t>(kernel->width),
                               static_cast<std::uint32_t>(kernel->height),
                               0,
                               0, // RowsInverse writes U; workbench_step.comp does the update
                               0.0f,
                               1.0f,
                               0.0f,
                               0};
    // Each pass reads what the previous one wrote and may overwrite what it read.
    const auto pass = [&](lenia::FftPass p) {
        pc.pass = static_cast<std::uint32_t>(p);
        df_->vkCmdPushConstants(cb, fft_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        df_->vkCmdDispatch(cb, lenia::fft_pass_groups(p, w, h), 1, 1);
        barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    };
    df_->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, fft_pipeline_);
    df_->vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, fft_layout_, 0, 1, &fft_sets_[cur_], 0, nullptr);
    if (fft_kernel_ != kernel) {
        pass(lenia::FftPass::KernelRowsForward);
        pass(lenia::FftPass::KernelColumns);
        fft_kernel_ = kernel;
    }
    pass(lenia::FftPass::RowsForward);
    pass(lenia::FftPass::ConvolveColumns);
    pass(lenia::FftPass::RowsInverse);
    df_->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, step_pipeline_);
}

void LeniaRenderer::write_descriptors() {
    constexpr int descriptor_count = 2 * (6 + 5 + 6); // two generations; compute, view and FFT bindings
    VkDescriptorBufferInfo infos[descriptor_count];
    VkWriteDescriptorSet writes[descriptor_count]{};
    std::uint32_t n = 0;
    const auto add = [&](VkDescriptorSet set, std::uint32_t binding, const Buffer& b) {
        infos[n] = {b.buffer, 0, VK_WHOLE_SIZE};
        writes[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[n].dstSet = set;
        writes[n].dstBinding = binding;
        writes[n].descriptorCount = 1;
        writes[n].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[n].pBufferInfo = &infos[n];
        ++n;
    };
    for (int i = 0; i < 2; ++i) {
        add(step_sets_[i], 0, cells_[i]);
        add(step_sets_[i], 1, kernel_);
        add(step_sets_[i], 2, cells_[1 - i]);
        add(step_sets_[i], 3, neighborhood_);
        add(step_sets_[i], 4, growth_);
        add(step_sets_[i], 5, rule_);
        add(view_sets_[i], 0, cells_[i]);
        add(view_sets_[i], 1, neighborhood_);
        add(view_sets_[i], 2, growth_);
        add(view_sets_[i], 3, palette_);
        add(view_sets_[i], 4, display_);
        if (fft_pipeline_ && spec_.buffer) {
            add(fft_sets_[i], 0, cells_[i]);
            add(fft_sets_[i], 1, kernel_);
            add(fft_sets_[i], 2, neighborhood_);
            add(fft_sets_[i], 3, spec_);
            add(fft_sets_[i], 4, khat_);
            add(fft_sets_[i], 5, twiddles_);
        }
    }
    df_->vkUpdateDescriptorSets(dev_, n, writes, 0, nullptr);
}

void LeniaRenderer::releaseResources() {
    // The world is about to be lost (device reset, window closing): a pending capture fails.
    if (view_->capture_) view_->finish_capture({{}, 0, QStringLiteral("The Vulkan device was reset before the capture")});
    if (!df_) return;
    df_->vkDeviceWaitIdle(dev_);
    destroy_buffer(staging_);
    destroy_world();
    destroy_fft();
    counted_kernel_.reset();
    destroy_buffer(kernel_);
    destroy_buffer(palette_);
    destroy_buffer(rule_);
    destroy_buffer(display_);
    if (pool_) df_->vkDestroyDescriptorPool(dev_, pool_, nullptr); // frees the sets
    if (step_pipeline_) df_->vkDestroyPipeline(dev_, step_pipeline_, nullptr);
    if (view_pipeline_) df_->vkDestroyPipeline(dev_, view_pipeline_, nullptr);
    if (step_layout_) df_->vkDestroyPipelineLayout(dev_, step_layout_, nullptr);
    if (view_layout_) df_->vkDestroyPipelineLayout(dev_, view_layout_, nullptr);
    if (step_set_layout_) df_->vkDestroyDescriptorSetLayout(dev_, step_set_layout_, nullptr);
    if (view_set_layout_) df_->vkDestroyDescriptorSetLayout(dev_, view_set_layout_, nullptr);
    pool_ = VK_NULL_HANDLE;
    step_pipeline_ = view_pipeline_ = VK_NULL_HANDLE;
    step_layout_ = view_layout_ = VK_NULL_HANDLE;
    step_set_layout_ = view_set_layout_ = VK_NULL_HANDLE;
    step_sets_[0] = step_sets_[1] = view_sets_[0] = view_sets_[1] = VK_NULL_HANDLE;
    uploaded_kernel_.reset();
    uploaded_rule_.reset();
    uploaded_display_.reset();
}

// ---------------------------------------------------------------------------------------------
// LeniaRenderer: per-frame recording. Everything goes into QVulkanWindow's frame command
// buffer, which the window submits on its graphics (+ compute) queue.

void LeniaRenderer::barrier(VkCommandBuffer cb, VkPipelineStageFlags src_stage, VkAccessFlags src_access,
                            VkPipelineStageFlags dst_stage, VkAccessFlags dst_access) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    df_->vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 1, &b, 0, nullptr, 0, nullptr);
}

void LeniaRenderer::update(VkCommandBuffer cb, const Buffer& b, VkDeviceSize offset, const void* data,
                           VkDeviceSize bytes) {
    const auto* src = static_cast<const char*>(data);
    while (bytes > 0) {
        const VkDeviceSize chunk = std::min(bytes, kUpdateChunk);
        df_->vkCmdUpdateBuffer(cb, b.buffer, offset, chunk, src);
        offset += chunk;
        src += chunk;
        bytes -= chunk;
    }
}

void LeniaRenderer::startNextFrame() {
    VkCommandBuffer cb = view_->currentCommandBuffer();

    if (!failed_ && (width_ != view_->world_width_ || height_ != view_->world_height_)) {
        // Earlier frames may still reference the old buffers.
        df_->vkDeviceWaitIdle(dev_);
        destroy_world();
        if (create_world(view_->world_width_, view_->world_height_)) write_descriptors();
    }
    bool program = false;
    bool capture = false;
    if (!failed_) {
        record_uploads(cb);
        record_steps(cb);
        capture = view_->capture_ && record_capture(cb);
        program = record_display(cb);
    } else if (view_->capture_) {
        view_->finish_capture({{}, 0, QStringLiteral("The Vulkan viewport has failed")});
    }
    record_draw(cb, program);

    // Submits this frame's command buffer (QVulkanWindow ends the frame right here), so the
    // capture's copy is on the queue when finish_capture waits for it.
    view_->frameReady();
    if (capture) finish_capture();
    if (!failed_ && !view_->paused_) view_->requestUpdate();
}

bool LeniaRenderer::record_capture(VkCommandBuffer cb) {
    // Both state fields: density, then nutrient.
    const VkDeviceSize bytes =
        2 * static_cast<VkDeviceSize>(width_) * static_cast<VkDeviceSize>(height_) * sizeof(float);
    QString error;
    soft_error_ = &error;
    destroy_buffer(staging_);
    const bool ok = width_ > 0 && create_buffer(staging_, bytes, true);
    soft_error_ = nullptr;
    if (!ok) {
        destroy_buffer(staging_);
        view_->finish_capture({{}, 0, error.isEmpty() ? QStringLiteral("No world to capture") : error});
        return false;
    }

    // cells_[cur_] is final for this frame: the uploads (transfer writes) and steps (compute
    // writes) recorded above, and earlier frames' work on this queue, must be complete and
    // visible before the copy reads it.
    barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_READ_BIT);
    const VkBufferCopy region{0, 0, bytes};
    df_->vkCmdCopyBuffer(cb, cells_[cur_].buffer, staging_.buffer, 1, &region);
    // The copy's writes must be visible to the host once the queue is idle.
    barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            VK_ACCESS_HOST_READ_BIT);
    capture_width_ = width_;
    capture_height_ = height_;
    capture_generation_ = view_->generation_; // after this frame's steps
    return true;
}

void LeniaRenderer::finish_capture() {
    // A device loss detected while ending the frame releases the resources (and fails the
    // capture) before frameReady returns.
    if (!staging_.buffer || !view_->capture_) {
        destroy_buffer(staging_);
        return;
    }
    WorldCapture result;
    // A one-off stall, only when a capture was asked for: the CPU waits for this frame.
    const VkResult wait = df_->vkQueueWaitIdle(view_->graphicsQueue());
    void* mapped = nullptr;
    if (wait != VK_SUCCESS) {
        result.error = QStringLiteral("Waiting for the GPU failed (VkResult %1)").arg(static_cast<int>(wait));
    } else if (const VkResult map = df_->vkMapMemory(dev_, staging_.memory, 0, VK_WHOLE_SIZE, 0, &mapped);
               map != VK_SUCCESS) {
        result.error = QStringLiteral("Mapping the readback buffer failed (VkResult %1)").arg(static_cast<int>(map));
    } else {
        if (!staging_coherent_) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = staging_.memory;
            range.size = VK_WHOLE_SIZE;
            df_->vkInvalidateMappedMemoryRanges(dev_, 1, &range);
        }
        const auto w = static_cast<std::size_t>(capture_width_), h = static_cast<std::size_t>(capture_height_);
        result.density = lenia::Grid(w, h);
        result.nutrient = lenia::Grid(w, h);
        const std::size_t plane = w * h * sizeof(float);
        std::memcpy(result.density.cells.data(), mapped, plane);
        std::memcpy(result.nutrient.cells.data(), static_cast<const char*>(mapped) + plane, plane);
        df_->vkUnmapMemory(dev_, staging_.memory);
        result.generation = capture_generation_;
        ++view_->readbacks_;
    }
    destroy_buffer(staging_); // the queue is idle (or lost): nothing uses it any more
    view_->finish_capture(std::move(result));
}

void LeniaRenderer::record_uploads(VkCommandBuffer cb) {
    LeniaView::Pending& p = view_->pending_;
    const bool any = clear_world_ || upload_palette_ || upload_twiddles_ || p.world || p.nutrient || p.impulse ||
                     !p.pokes.empty();
    if (!any) return;
    // The density's size in bytes; the nutrient follows it in each cell buffer.
    const VkDeviceSize plane = static_cast<VkDeviceSize>(width_) * static_cast<VkDeviceSize>(height_) * sizeof(float);

    constexpr VkPipelineStageFlags kShaders = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    constexpr VkAccessFlags kShaderAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    const auto transfer_to_transfer = [&] {
        barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT);
    };

    // Earlier frames' shaders (same queue, earlier submissions) must be done with these
    // buffers before they are overwritten.
    barrier(cb, kShaders, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);

    // Phase 1: clears.
    bool cleared = false;
    if (clear_world_) {
        for (Buffer* b : {&cells_[0], &cells_[1], &neighborhood_, &growth_}) {
            df_->vkCmdFillBuffer(cb, b->buffer, 0, VK_WHOLE_SIZE, 0);
        }
        clear_world_ = false;
        cleared = true;
    }
    if (p.impulse) {
        df_->vkCmdFillBuffer(cb, cells_[cur_].buffer, 0, plane, 0); // the density only
        cleared = true;
    }
    if (cleared) transfer_to_transfer();

    // Phase 2: whole-buffer uploads.
    bool uploaded = false;
    if (upload_palette_) {
        const std::vector<float> table = palette_table();
        update(cb, palette_, 0, table.data(), table.size() * sizeof(float));
        upload_palette_ = false;
        uploaded = true;
    }
    if (upload_twiddles_) {
        const std::vector<float> table = lenia::fft_twiddles(fft_max_);
        update(cb, twiddles_, 0, table.data(), table.size() * sizeof(float));
        upload_twiddles_ = false;
        uploaded = true;
    }
    if (p.world) {
        update(cb, cells_[cur_], 0, p.world->cells.data(), p.world->cells.size() * sizeof(float));
        p.world.reset();
        uploaded = true;
    }
    if (p.nutrient) {
        if (p.nutrient->cells.size() * sizeof(float) == plane) {
            update(cb, cells_[cur_], plane, p.nutrient->cells.data(), plane);
            uploaded = true;
        }
        p.nutrient.reset();
    }
    if (p.impulse) {
        const std::uint32_t center = static_cast<std::uint32_t>((height_ / 2) * width_ + width_ / 2);
        const float one = 1.0f;
        update(cb, cells_[cur_], center * sizeof(float), &one, sizeof(float));
        p.impulse = false;
        uploaded = true;
    }

    // Phase 3: painted cells, coalesced into runs of consecutive indices.
    if (!p.pokes.empty()) {
        if (uploaded) transfer_to_transfer();
        const std::uint32_t cell_count = static_cast<std::uint32_t>(2 * width_ * height_); // both fields
        std::vector<float> run;
        for (auto it = p.pokes.begin(); it != p.pokes.end();) {
            const std::uint32_t start = it->first;
            run.clear();
            while (it != p.pokes.end() && it->first == start + run.size() && it->first < cell_count) {
                run.push_back(it->second);
                ++it;
            }
            if (run.empty()) break; // index out of range; the rest are larger still
            update(cb, cells_[cur_], static_cast<VkDeviceSize>(start) * sizeof(float), run.data(),
                   run.size() * sizeof(float));
        }
        p.pokes.clear();
    }

    p.analyze = true;
    barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, kShaders, kShaderAccess);
}

void LeniaRenderer::record_steps(VkCommandBuffer cb) {
    LeniaView::Pending& p = view_->pending_;
    const int steps = view_->paused_ ? p.steps : view_->steps_per_frame_;
    p.steps = 0;
    if (!view_->inputs_ || width_ == 0) return;
    if (steps == 0 && !p.analyze) return;
    p.analyze = false;

    // An analysis-only dispatch (advance = 0) refreshes neighborhood and growth for the
    // current state; stepping dispatches refresh them for the state they step from.
    const int dispatches = std::max(steps, 1);
    const std::uint32_t advance = steps > 0 ? 1u : 0u;
    const std::uint32_t groups_x = (static_cast<std::uint32_t>(width_) + kLocalSize - 1) / kLocalSize;
    const std::uint32_t groups_y = (static_cast<std::uint32_t>(height_) + kLocalSize - 1) / kLocalSize;

    df_->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, step_pipeline_);
    int done = 0;
    bool used_fft = fft_in_use_;
    for (int i = 0; i < dispatches; ++i) {
        // Inputs of the generation this dispatch reads from, so a frame of n steps computes
        // exactly what n single steps would.
        const GenerationInputs in = view_->inputs_(view_->generation_ + static_cast<std::uint64_t>(advance ? i : 0));
        if (!valid_kernel(in.kernel.get())) {
            if (!warned_bad_kernel_) qWarning("LeniaView: provider returned no valid kernel; not stepping");
            warned_bad_kernel_ = true;
            break;
        }
        warned_bad_kernel_ = false;
        if (in.rule && !valid_rule(*in.rule)) {
            if (!warned_bad_rule_) qWarning("LeniaView: rule program too large; not stepping");
            warned_bad_rule_ = true;
            break;
        }
        warned_bad_rule_ = false;
        const bool new_kernel = in.kernel != uploaded_kernel_;
        const bool new_rule = in.rule && in.rule != uploaded_rule_;
        if (new_kernel || new_rule) {
            // The previous dispatch (or an earlier frame) may still read the old kernel or
            // program: a write-after-read hazard, so an execution dependency is enough.
            barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT);
            if (new_kernel) {
                update(cb, kernel_, 0, in.kernel->cells.data(), in.kernel->cells.size() * sizeof(float));
                uploaded_kernel_ = in.kernel;
            }
            if (new_rule) {
                const std::vector<std::uint32_t> words = lenia::pack_rule(*in.rule, kRuleKernelBase);
                update(cb, rule_, 0, words.data(), words.size() * sizeof(std::uint32_t));
                const std::vector<float> kernels = lenia::pack_rule_kernels(*in.rule);
                if (!kernels.empty()) {
                    update(cb, kernel_, static_cast<VkDeviceSize>(kRuleKernelBase) * sizeof(float), kernels.data(),
                           kernels.size() * sizeof(float));
                }
                uploaded_rule_ = in.rule;
            }
            barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_READ_BIT);
        }

        // FFT: U into neighborhood_ first (after the kernel upload above, which it may transform).
        used_fft = fft_enabled(in.kernel);
        if (used_fft) record_fft(cb, in.kernel);

        const lenia::Environment& env = in.environment;
        const bool env_on = env.strength != 0.0f;
        const StepPush pc{static_cast<std::uint32_t>(width_),
                          static_cast<std::uint32_t>(height_),
                          static_cast<std::uint32_t>(in.kernel->width),
                          static_cast<std::uint32_t>(in.kernel->height),
                          in.growth.mu,
                          in.growth.sigma,
                          in.growth.dt,
                          advance,
                          static_cast<std::uint32_t>(in.growth.function),
                          env_on ? static_cast<std::uint32_t>(env.pattern) + 1u : 0u,
                          env.strength,
                          env.scale,
                          std::cos(env.angle),
                          std::sin(env.angle),
                          env.offset,
                          in.rule ? 1u : 0u,
                          used_fft ? 1u : 0u,
                          in.rule ? in.preview : lenia::kNoRegister};
        df_->vkCmdPushConstants(cb, step_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        df_->vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, step_layout_, 0, 1, &step_sets_[cur_], 0,
                                     nullptr);
        df_->vkCmdDispatch(cb, groups_x, groups_y, 1);
        // The next dispatch reads this one's output and overwrites what it read; the fragment
        // shader reads the final fields.
        barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        if (advance) cur_ = 1 - cur_;
        ++done;
    }
    if (advance && done > 0) {
        view_->generation_ += static_cast<std::uint64_t>(done);
        emit view_->generation_changed(view_->generation_);
    }
    if (done > 0 && used_fft != fft_in_use_) {
        fft_in_use_ = used_fft;
        emit view_->convolution_used(used_fft);
    }
}

bool LeniaRenderer::record_display(VkCommandBuffer cb) {
    if (!view_->display_ || !display_.buffer) return false;
    // The generation on screen: after this frame's steps, the state the fields were computed for.
    const std::shared_ptr<const lenia::DisplayProgram> program = view_->display_(view_->generation_);
    if (!program) return false;
    if (!valid_display(*program)) {
        if (!warned_bad_display_) qWarning("LeniaView: display program too large; drawing the palette view");
        warned_bad_display_ = true;
        return false;
    }
    warned_bad_display_ = false;
    if (program != uploaded_display_) {
        // Earlier frames' fragment shaders may still read the old program (write after read).
        barrier(cb, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT);
        const std::vector<std::uint32_t> words = lenia::pack_display(*program);
        update(cb, display_, 0, words.data(), words.size() * sizeof(std::uint32_t));
        barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT);
        uploaded_display_ = program;
    }
    return true;
}

void LeniaRenderer::record_draw(VkCommandBuffer cb, bool program) {
    const QSize size = view_->swapChainImageSize();

    VkClearValue clear[3]{};
    clear[0].color = clear[2].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clear[1].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = view_->defaultRenderPass();
    rp.framebuffer = view_->currentFramebuffer();
    rp.renderArea.extent.width = static_cast<std::uint32_t>(size.width());
    rp.renderArea.extent.height = static_cast<std::uint32_t>(size.height());
    rp.clearValueCount = view_->sampleCountFlagBits() > VK_SAMPLE_COUNT_1_BIT ? 3 : 2;
    rp.pClearValues = clear;
    df_->vkCmdBeginRenderPass(cb, &rp, VK_SUBPASS_CONTENTS_INLINE);

    if (!failed_ && view_pipeline_ && width_ > 0) {
        const VkViewport viewport{0.0f, 0.0f, static_cast<float>(size.width()), static_cast<float>(size.height()),
                                  0.0f, 1.0f};
        df_->vkCmdSetViewport(cb, 0, 1, &viewport);
        df_->vkCmdSetScissor(cb, 0, 1, &rp.renderArea);
        df_->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, view_pipeline_);
        df_->vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, view_layout_, 0, 1, &view_sets_[cur_], 0,
                                     nullptr);

        const LeniaView::Placement place = view_->placement(QSizeF(size));
        const ViewSettings& v = view_->view_;
        const ViewPush pc{{static_cast<float>(place.origin.x()), static_cast<float>(place.origin.y())},
                          static_cast<float>(1.0 / place.pixels_per_cell),
                          static_cast<std::uint32_t>(width_),
                          static_cast<std::uint32_t>(height_),
                          static_cast<std::uint32_t>(v.field),
                          static_cast<std::uint32_t>(std::clamp(v.palette, 0, PaletteCount - 1)),
                          v.lo,
                          v.hi,
                          linear_target_ ? 1u : 0u,
                          program ? 1u : 0u};
        df_->vkCmdPushConstants(cb, view_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
        df_->vkCmdDraw(cb, 3, 1, 0, 0);
    }
    df_->vkCmdEndRenderPass(cb);
}

// ---------------------------------------------------------------------------------------------
// LeniaView

LeniaView::LeniaView() {
    center_ = QPointF(world_width_ / 2.0, world_height_ / 2.0);
}

QVulkanWindowRenderer* LeniaView::createRenderer() { return new LeniaRenderer(this); }

void LeniaView::upload_world(const lenia::Grid& world, std::uint64_t generation) {
    if (world.width == 0 || world.height == 0 || world.cells.size() != world.width * world.height) {
        qWarning("upload_world: malformed grid ignored");
        return;
    }
    const bool resized = static_cast<int>(world.width) != world_width_ || static_cast<int>(world.height) != world_height_;
    world_width_ = static_cast<int>(world.width);
    world_height_ = static_cast<int>(world.height);
    pending_.world = world;
    if (resized) pending_.nutrient.reset(); // sized for the old world; the new one starts at 0
    pending_.impulse = false;
    pending_.pokes.clear();
    pending_.steps = 0; // steps queued for the old world do not apply to this one
    generation_ = generation;
    emit generation_changed(generation_);
    if (resized) fit_view();
    requestUpdate();
}

void LeniaView::upload_world(const lenia::Grid& density, const lenia::Grid& nutrient, std::uint64_t generation) {
    if (nutrient.width != density.width || nutrient.height != density.height ||
        nutrient.cells.size() != density.cells.size()) {
        qWarning("upload_world: nutrient of another size ignored");
        upload_world(density, generation);
        return;
    }
    upload_world(density, generation);
    if (static_cast<int>(density.width) == world_width_ && static_cast<int>(density.height) == world_height_) {
        pending_.nutrient = nutrient;
    }
}

void LeniaView::capture_world(CaptureCallback done) {
    if (!done) return;
    if (capture_) {
        done({{}, 0, tr("A capture is already in progress")});
        return;
    }
    if (!isExposed() || !isValid()) {
        done({{}, 0, tr("The viewport is not visible (minimized?) or has no Vulkan device, so the GPU world cannot be "
                        "read back")});
        return;
    }
    capture_ = std::move(done);
    requestUpdate();
}

void LeniaView::finish_capture(WorldCapture capture) {
    CaptureCallback done = std::move(capture_);
    capture_ = nullptr;
    if (done) done(capture);
}

void LeniaView::seed_impulse() {
    pending_.world.reset();
    pending_.impulse = true;
    pending_.pokes.clear();
    pending_.steps = 0;
    generation_ = 0;
    emit generation_changed(generation_);
    requestUpdate();
}

void LeniaView::set_inputs(InputsProvider provider) {
    inputs_ = std::move(provider);
    inputs_changed();
}

void LeniaView::inputs_changed() {
    pending_.analyze = true;
    requestUpdate();
}

void LeniaView::set_display(DisplayProvider provider) {
    display_ = std::move(provider);
    requestUpdate();
}

void LeniaView::display_changed() { requestUpdate(); }

void LeniaView::set_view(const ViewSettings& view) {
    view_ = view;
    requestUpdate();
}

void LeniaView::set_paused(bool paused) {
    paused_ = paused;
    requestUpdate();
}

void LeniaView::step(int n) {
    pending_.steps += std::max(n, 0);
    requestUpdate();
}

void LeniaView::set_steps_per_frame(int n) { steps_per_frame_ = std::max(n, 0); }

void LeniaView::set_convolution(lenia::ConvolutionMethod method) {
    if (method == convolution_) return;
    convolution_ = method;
    pending_.analyze = true; // refresh U (and G) of a paused world with the chosen method
    requestUpdate();
}

void LeniaView::set_brush(int radius, float value) {
    brush_radius_ = std::max(radius, 0);
    brush_value_ = value;
}

void LeniaView::fit_view() {
    zoom_ = 1.0;
    center_ = QPointF(world_width_ / 2.0, world_height_ / 2.0);
    requestUpdate();
}

LeniaView::Placement LeniaView::placement(QSizeF pixels) const {
    const double fit = std::min(pixels.width() / world_width_, pixels.height() / world_height_);
    const double ppc = std::max(fit * zoom_, 1e-6);
    return {QPointF(pixels.width() / 2.0 - center_.x() * ppc, pixels.height() / 2.0 - center_.y() * ppc), ppc};
}

QPointF LeniaView::to_cell(QPointF logical_pos) const {
    const qreal dpr = devicePixelRatio();
    const QSize swap = swapChainImageSize();
    const QSizeF pixels = swap.isEmpty() ? QSizeF(size()) * dpr : QSizeF(swap);
    const Placement place = placement(pixels);
    return (logical_pos * dpr - place.origin) / place.pixels_per_cell;
}

void LeniaView::paint_disk(int cx, int cy, float value) {
    if (cx < 0 || cy < 0 || cx >= world_width_ || cy >= world_height_) return;
    const int r = brush_radius_;
    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            if (dx * dx + dy * dy > r * r) continue;
            const int x = ((cx + dx) % world_width_ + world_width_) % world_width_;
            const int y = ((cy + dy) % world_height_ + world_height_) % world_height_;
            const int base = brush_field_ == BrushField::Nutrient ? world_width_ * world_height_ : 0;
            pending_.pokes[static_cast<std::uint32_t>(base + y * world_width_ + x)] = value;
        }
    }
}

void LeniaView::paint_line(QPointF from, QPointF to, float value) {
    const QPointF d = to - from;
    const double length = std::hypot(d.x(), d.y());
    const double spacing = std::max(0.5, brush_radius_ / 2.0);
    const int samples = static_cast<int>(std::ceil(length / spacing));
    for (int i = 0; i <= samples; ++i) {
        const QPointF c = samples == 0 ? to : from + d * (static_cast<double>(i) / samples);
        paint_disk(static_cast<int>(std::floor(c.x())), static_cast<int>(std::floor(c.y())), value);
    }
    pending_.analyze = true;
    requestUpdate();
}

void LeniaView::mousePressEvent(QMouseEvent* e) {
    last_mouse_ = e->position();
    if (e->button() == Qt::LeftButton && (e->modifiers() & Qt::AltModifier)) {
        const QPointF cell = to_cell(e->position());
        const int x = static_cast<int>(std::floor(cell.x()));
        const int y = static_cast<int>(std::floor(cell.y()));
        if (x >= 0 && y >= 0 && x < world_width_ && y < world_height_) emit cell_probed(x, y);
    } else if (e->button() == Qt::LeftButton) {
        drag_ = Drag::Paint;
        drag_value_ = (e->modifiers() & Qt::ShiftModifier) ? 0.0f : brush_value_;
        last_cell_ = to_cell(e->position());
        paint_line(last_cell_, last_cell_, drag_value_);
    } else if (e->button() == Qt::RightButton || e->button() == Qt::MiddleButton) {
        drag_ = Drag::Pan;
    }
}

void LeniaView::mouseMoveEvent(QMouseEvent* e) {
    if (drag_ == Drag::Paint) {
        const QPointF cell = to_cell(e->position());
        paint_line(last_cell_, cell, drag_value_);
        last_cell_ = cell;
    } else if (drag_ == Drag::Pan) {
        const QSize swap = swapChainImageSize();
        const QSizeF pixels = swap.isEmpty() ? QSizeF(size()) * devicePixelRatio() : QSizeF(swap);
        const double ppc = placement(pixels).pixels_per_cell;
        center_ -= (e->position() - last_mouse_) * devicePixelRatio() / ppc;
        requestUpdate();
    }
    last_mouse_ = e->position();
}

void LeniaView::mouseReleaseEvent(QMouseEvent*) { drag_ = Drag::None; }

void LeniaView::keyPressEvent(QKeyEvent* e) {
    if (e->modifiers() == Qt::NoModifier && !e->isAutoRepeat()) emit key_pressed(e->key());
    if (e->modifiers().testFlag(Qt::ControlModifier) && !e->isAutoRepeat()) {
        emit shortcut_pressed(QKeySequence(e->keyCombination()));
    }
    QVulkanWindow::keyPressEvent(e);
}

void LeniaView::wheelEvent(QWheelEvent* e) {
    const double notches = e->angleDelta().y() / 120.0;
    if (notches == 0.0) return;
    // Keep the cell under the cursor in place.
    const QPointF before = to_cell(e->position());
    zoom_ = std::clamp(zoom_ * std::pow(1.15, notches), 0.25, 128.0);
    const QPointF after = to_cell(e->position());
    center_ += before - after;
    requestUpdate();
}

} // namespace workbench
