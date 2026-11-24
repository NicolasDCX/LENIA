#include "lenia/vulkan_step.hpp"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace lenia {

namespace {
// SPIR-V words generated from shaders/lenia_step.comp, rule_step.comp and fft_convolve.comp at
// build time.
const std::uint32_t kStepCode[] = {
#include "lenia_step.spv.inc"
};
const std::uint32_t kRuleCode[] = {
#include "rule_step.spv.inc"
};
const std::uint32_t kFftCode[] = {
#include "fft_convolve.spv.inc"
};

// Must match local_size_x / local_size_y in the shaders.
constexpr std::uint32_t kLocalSize = 16;

// Threads per FFT line (fft_convolve.comp's local_size_x_id), if the device allows them.
constexpr std::uint32_t kFftThreads = 256;

// Specialization constants of fft_convolve.comp: FFT_MAX (id 0) and the threads (id 1).
struct FftSpecialization {
    std::uint32_t max_size;
    std::uint32_t threads;
};

// Timestamps: after the upload, after the FFT kernel transform (if any), after the generations.
constexpr std::uint32_t kQueries = 3;

// Must match the push_constant block in lenia_step.comp.
struct PushConstants {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t kwidth;
    std::uint32_t kheight;
    float mu;
    float sigma;
    float dt;
    std::uint32_t growth; // lenia::GrowthFunction
};

// Must match the push_constant block in rule_step.comp.
struct RulePushConstants {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t kwidth;
    std::uint32_t kheight;
};

void check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string("Vulkan: ") + what + " failed (VkResult " +
                                 std::to_string(static_cast<int>(result)) + ")");
    }
}

void validate(const Grid& g, const char* name) {
    if (g.cells.size() != g.width * g.height) {
        throw std::invalid_argument(std::string(name) + " cell count does not match width * height");
    }
}

const char* device_type_name(VkPhysicalDeviceType type) {
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
    default: return "other";
    }
}

// Offsets of the kernel and rule program in the staging buffer, after the grid.
constexpr VkDeviceSize kStagingAlign = 256;
VkDeviceSize align_up(VkDeviceSize n) { return (n + kStagingAlign - 1) / kStagingAlign * kStagingAlign; }
} // namespace

struct VulkanStepper::Impl {
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        VkDeviceSize size = 0;
    };

    // One compute shader with its layouts. Bindings 0, 1, 2 are src, ker, dst; a rule pipeline
    // also has the program at binding 3, the FFT pipeline spec, khat and twiddles at 3, 4, 5.
    struct Pipeline {
        std::uint32_t bindings = 0;
        VkShaderModule shader = VK_NULL_HANDLE;
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        // sets[i] reads cells[i] and writes cells[1 - i].
        VkDescriptorSet sets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    };

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkDevice device = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    std::uint32_t timestamp_bits = 0; // timestampValidBits of queue_family
    VkQueue queue = VK_NULL_HANDLE;
    Pipeline lenia, rule;
    Pipeline fft; // null pipeline if the device cannot run fft_convolve.comp
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    // kQueries timestamps around a batch's dispatches; null without timestamp support.
    VkQueryPool query_pool = VK_NULL_HANDLE;
    // Device-local working buffers, and the host-visible staging buffer that fills and reads them.
    // spec, khat and twiddles are the FFT path's half spectra (scratch and kernel) and its table.
    Buffer cells[2], ker, program, staging, spec, khat, twiddles;
    bool twiddles_uploaded = false;
    // The kernel and grid size whose spectrum khat holds, if fft_valid.
    Grid fft_kernel;
    std::size_t fft_width = 0, fft_height = 0;
    bool fft_valid = false;
    ConvolutionMethod method = ConvolutionMethod::Direct;
    ConvolutionMethod last_method = ConvolutionMethod::Direct;
    std::string device_name;
    VulkanDeviceInfo info;
    std::optional<double> last_gpu_ms;
    std::optional<double> last_transform_ms;

    Impl() {
        try {
            init();
        } catch (...) {
            destroy();
            throw;
        }
    }
    ~Impl() { destroy(); }

    void init() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "lenia";
        app.apiVersion = VK_API_VERSION_1_0;
        VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instance_info.pApplicationInfo = &app;
        check(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");

        pick_physical_device();
        vkGetPhysicalDeviceProperties(physical, &properties);
        device_name = properties.deviceName;
        info.name = device_name;
        info.type = device_type_name(properties.deviceType);
        info.vendor_id = properties.vendorID;
        info.device_id = properties.deviceID;
        info.driver_version = properties.driverVersion;
        info.api_version = properties.apiVersion;
        info.timestamps = timestamp_bits > 0 && properties.limits.timestampPeriod > 0.0f;
        info.timestamp_period_ns = properties.limits.timestampPeriod;
        info.device_local = has_device_local_memory();

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = queue_family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        check(vkCreateDevice(physical, &device_info, nullptr, &device), "vkCreateDevice");
        vkGetDeviceQueue(device, queue_family, 0, &queue);

        create_pipeline(lenia, kStepCode, sizeof(kStepCode), 3, sizeof(PushConstants));
        create_pipeline(rule, kRuleCode, sizeof(kRuleCode), 4, sizeof(RulePushConstants));
        create_fft_pipeline();

        VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        command_pool_info.queueFamilyIndex = queue_family;
        check(vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool), "vkCreateCommandPool");

        VkCommandBufferAllocateInfo command_buffer_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_buffer_info.commandPool = command_pool;
        command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_buffer_info.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device, &command_buffer_info, &command_buffer),
              "vkAllocateCommandBuffers");

        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(device, &fence_info, nullptr, &fence), "vkCreateFence");

        if (info.timestamps) {
            VkQueryPoolCreateInfo query_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
            query_info.queryCount = kQueries;
            check(vkCreateQueryPool(device, &query_info, nullptr, &query_pool), "vkCreateQueryPool");
        }
    }

    // fft_convolve.comp, sized for this device: the longest line whose elements fit shared
    // memory, and as many threads per line (up to kFftThreads) as a workgroup may have. Leaves
    // `fft` null and info.fft_max_size 0, so step_lenia stays Direct, if the device cannot.
    void create_fft_pipeline() {
        const VkPhysicalDeviceLimits& limits = properties.limits;
        const std::size_t max_size = fft_max_size(limits.maxComputeSharedMemorySize);
        std::uint32_t threads = kFftThreads;
        while (threads > 1 && (threads > limits.maxComputeWorkGroupInvocations || threads > limits.maxComputeWorkGroupSize[0])) {
            threads /= 2;
        }
        if (max_size == 0) return;
        const FftSpecialization values{static_cast<std::uint32_t>(max_size), threads};
        const VkSpecializationMapEntry entries[2] = {{0, offsetof(FftSpecialization, max_size), sizeof(std::uint32_t)},
                                                     {1, offsetof(FftSpecialization, threads), sizeof(std::uint32_t)}};
        VkSpecializationInfo specialization{};
        specialization.mapEntryCount = 2;
        specialization.pMapEntries = entries;
        specialization.dataSize = sizeof(values);
        specialization.pData = &values;
        try {
            create_pipeline(fft, kFftCode, sizeof(kFftCode), 6, sizeof(FftPushConstants), &specialization);
            info.fft_max_size = max_size;
        } catch (const std::runtime_error&) {
            destroy_pipeline(fft); // the Direct path does not need it
        }
    }

    void create_pipeline(Pipeline& p, const std::uint32_t* code, std::size_t code_bytes, std::uint32_t bindings,
                         std::uint32_t push_size, const VkSpecializationInfo* specialization = nullptr) {
        p.bindings = bindings;
        VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader_info.codeSize = code_bytes;
        shader_info.pCode = code;
        check(vkCreateShaderModule(device, &shader_info, nullptr, &p.shader), "vkCreateShaderModule");

        std::vector<VkDescriptorSetLayoutBinding> layout_bindings(bindings);
        for (std::uint32_t b = 0; b < bindings; ++b) {
            layout_bindings[b] = {};
            layout_bindings[b].binding = b;
            layout_bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            layout_bindings[b].descriptorCount = 1;
            layout_bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo set_layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set_layout_info.bindingCount = bindings;
        set_layout_info.pBindings = layout_bindings.data();
        check(vkCreateDescriptorSetLayout(device, &set_layout_info, nullptr, &p.set_layout),
              "vkCreateDescriptorSetLayout");

        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.size = push_size;
        VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &p.set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        check(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &p.layout), "vkCreatePipelineLayout");

        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = p.shader;
        pipeline_info.stage.pName = "main";
        pipeline_info.stage.pSpecializationInfo = specialization;
        pipeline_info.layout = p.layout;
        check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &p.pipeline),
              "vkCreateComputePipelines");

        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * bindings};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 2;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        check(vkCreateDescriptorPool(device, &pool_info, nullptr, &p.pool), "vkCreateDescriptorPool");

        const VkDescriptorSetLayout set_layouts[2] = {p.set_layout, p.set_layout};
        VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = p.pool;
        set_info.descriptorSetCount = 2;
        set_info.pSetLayouts = set_layouts;
        check(vkAllocateDescriptorSets(device, &set_info, p.sets), "vkAllocateDescriptorSets");
    }

    void destroy_pipeline(Pipeline& p) {
        if (p.pool) vkDestroyDescriptorPool(device, p.pool, nullptr); // frees the sets
        if (p.pipeline) vkDestroyPipeline(device, p.pipeline, nullptr);
        if (p.layout) vkDestroyPipelineLayout(device, p.layout, nullptr);
        if (p.set_layout) vkDestroyDescriptorSetLayout(device, p.set_layout, nullptr);
        if (p.shader) vkDestroyShaderModule(device, p.shader, nullptr);
        p = Pipeline{};
    }

    void pick_physical_device() {
        std::uint32_t count = 0;
        check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
        std::vector<VkPhysicalDevice> devices(count);
        check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "vkEnumeratePhysicalDevices");

        bool found = false;
        for (VkPhysicalDevice candidate : devices) {
            std::uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());

            for (std::uint32_t f = 0; f < family_count; ++f) {
                if ((families[f].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(candidate, &props);
                const bool discrete = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
                if (!found || discrete) {
                    physical = candidate;
                    queue_family = f;
                    timestamp_bits = families[f].timestampValidBits;
                    found = true;
                }
                if (discrete) return;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("Vulkan: no physical device with a compute queue");
        }
    }

    // First memory type allowed by `type_bits` with all of `flags`, or -1.
    int find_memory_type(std::uint32_t type_bits, VkMemoryPropertyFlags flags) const {
        VkPhysicalDeviceMemoryProperties mem;
        vkGetPhysicalDeviceMemoryProperties(physical, &mem);
        for (std::uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
            if ((type_bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & flags) == flags) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    bool has_device_local_memory() const {
        return find_memory_type(~0u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) >= 0;
    }

    // Grows but never shrinks. A `host` buffer is host-visible, coherent and persistently
    // mapped (staging); otherwise it is a storage buffer in device-local memory if possible.
    void ensure_buffer(Buffer& b, VkDeviceSize size, bool host) {
        if (b.size >= size) return;
        destroy_buffer(b);

        VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size = size;
        buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (!host) buffer_info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device, &buffer_info, nullptr, &b.buffer), "vkCreateBuffer");

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, b.buffer, &req);
        int type = -1;
        if (host) {
            type = find_memory_type(req.memoryTypeBits,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (type < 0) throw std::runtime_error("Vulkan: no host-visible coherent memory type");
        } else {
            type = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (type < 0) {
                info.device_local = false;
                type = find_memory_type(req.memoryTypeBits, 0);
            }
            if (type < 0) throw std::runtime_error("Vulkan: no memory type for a storage buffer");
        }
        VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc_info.allocationSize = req.size;
        alloc_info.memoryTypeIndex = static_cast<std::uint32_t>(type);
        check(vkAllocateMemory(device, &alloc_info, nullptr, &b.memory), "vkAllocateMemory");
        check(vkBindBufferMemory(device, b.buffer, b.memory, 0), "vkBindBufferMemory");
        if (host) check(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped), "vkMapMemory");
        b.size = size;
    }

    void destroy_buffer(Buffer& b) {
        if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
        if (b.memory) vkFreeMemory(device, b.memory, nullptr); // implicitly unmaps
        b = Buffer{};
    }

    void destroy() {
        if (device) {
            vkDeviceWaitIdle(device);
            destroy_buffer(cells[0]);
            destroy_buffer(cells[1]);
            destroy_buffer(ker);
            destroy_buffer(program);
            destroy_buffer(staging);
            destroy_buffer(spec);
            destroy_buffer(khat);
            destroy_buffer(twiddles);
            if (query_pool) vkDestroyQueryPool(device, query_pool, nullptr);
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
            destroy_pipeline(lenia);
            destroy_pipeline(rule);
            destroy_pipeline(fft);
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
        if (instance) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
    }

    bool fft_supported(std::size_t width, std::size_t height) const {
        if (!fft.pipeline || !fft_size_supported(width, height, info.fft_max_size)) return false;
        const VkPhysicalDeviceLimits& limits = properties.limits;
        const std::uint64_t spectrum_bytes = fft_spectrum_size(width, height) * 2 * sizeof(float);
        return spectrum_bytes <= limits.maxStorageBufferRange &&
               static_cast<std::uint64_t>(width) * height * sizeof(float) <= limits.maxStorageBufferRange &&
               fft_pass_groups(FftPass::RowsForward, width, height) <= limits.maxComputeWorkGroupCount[0] &&
               fft_pass_groups(FftPass::ConvolveColumns, width, height) <= limits.maxComputeWorkGroupCount[0];
    }

    // Runs `generations` dispatches of `p` ping-ponging between cells[0] and cells[1]. A rule
    // pipeline passes its packed program (`words`, binding 3). With `fft_growth` (and `p` the
    // fft pipeline, `push` unused) each generation is fft_convolve.comp's RowsForward,
    // ConvolveColumns and fused RowsInverse, after the kernel transform if khat is stale.
    Grid run(Pipeline& p, const Grid& grid, const Grid& kernel, const void* push, std::uint32_t push_size,
             std::size_t generations, const std::vector<std::uint32_t>* words, const GrowthParams* fft_growth) {
        last_gpu_ms.reset();
        last_transform_ms.reset();
        const bool use_fft = fft_growth != nullptr;
        last_method = use_fft ? ConvolutionMethod::Fft : ConvolutionMethod::Direct;
        if (kernel.width % 2 == 0 || kernel.height % 2 == 0) {
            throw std::invalid_argument("kernel dimensions must be odd");
        }
        validate(grid, "grid");
        validate(kernel, "kernel");

        if (generations == 0) return grid;
        Grid next(grid.width, grid.height, 0.0f);
        if (next.cells.empty()) return next;

        // The shaders index cells with 32-bit unsigned arithmetic.
        constexpr std::size_t max_u32 = std::numeric_limits<std::uint32_t>::max();
        if (grid.cells.size() > max_u32 / 2 || kernel.cells.size() > max_u32 / 2) {
            throw std::runtime_error("Vulkan: grid or kernel too large for 32-bit indexing");
        }
        const std::uint32_t groups_x = static_cast<std::uint32_t>((grid.width + kLocalSize - 1) / kLocalSize);
        const std::uint32_t groups_y = static_cast<std::uint32_t>((grid.height + kLocalSize - 1) / kLocalSize);
        if (groups_x > properties.limits.maxComputeWorkGroupCount[0] ||
            groups_y > properties.limits.maxComputeWorkGroupCount[1]) {
            throw std::runtime_error("Vulkan: grid exceeds maxComputeWorkGroupCount");
        }

        const VkDeviceSize grid_bytes = grid.cells.size() * sizeof(float);
        const VkDeviceSize kernel_bytes = kernel.cells.size() * sizeof(float);
        const VkDeviceSize program_bytes = words ? words->size() * sizeof(std::uint32_t) : 0;
        const VkDeviceSize kernel_offset = align_up(grid_bytes);
        const VkDeviceSize program_offset = kernel_offset + align_up(kernel_bytes);
        // FFT: half spectra (complex floats) and the twiddle table, uploaded once per stepper.
        const VkDeviceSize spectrum_bytes = use_fft ? fft_spectrum_size(grid.width, grid.height) * 2 * sizeof(float) : 0;
        const VkDeviceSize twiddle_bytes = use_fft ? info.fft_max_size * sizeof(float) : 0; // max / 2 complex
        const bool upload_twiddles = use_fft && !twiddles_uploaded;
        const VkDeviceSize twiddle_offset = program_offset + align_up(program_bytes);
        ensure_buffer(cells[0], grid_bytes, false);
        ensure_buffer(cells[1], grid_bytes, false);
        ensure_buffer(ker, kernel_bytes, false);
        if (words) ensure_buffer(program, kRuleBufferWords * sizeof(std::uint32_t), false);
        if (use_fft) {
            const VkBuffer old_khat = khat.buffer;
            ensure_buffer(spec, spectrum_bytes, false);
            ensure_buffer(khat, spectrum_bytes, false);
            ensure_buffer(twiddles, twiddle_bytes, false);
            if (khat.buffer != old_khat) fft_valid = false;
        }
        ensure_buffer(staging, upload_twiddles ? twiddle_offset + twiddle_bytes : program_offset + program_bytes, true);
        // The only upload of the batch; later generations stay on the device.
        auto* staged = static_cast<unsigned char*>(staging.mapped);
        std::memcpy(staged, grid.cells.data(), grid_bytes);
        std::memcpy(staged + kernel_offset, kernel.cells.data(), kernel_bytes);
        if (words) std::memcpy(staged + program_offset, words->data(), program_bytes);
        if (upload_twiddles) {
            const std::vector<float> table = fft_twiddles(info.fft_max_size);
            std::memcpy(staged + twiddle_offset, table.data(), twiddle_bytes);
        }
        // The kernel's spectrum is recomputed only for a new kernel or grid size.
        const bool transform = use_fft && !(fft_valid && fft_width == grid.width && fft_height == grid.height &&
                                            fft_kernel.width == kernel.width && fft_kernel.height == kernel.height &&
                                            fft_kernel.cells == kernel.cells);

        // Safe to rewrite every call: the previous submission was waited on.
        VkDescriptorBufferInfo infos[2][6];
        VkWriteDescriptorSet writes[2][6]{};
        for (std::uint32_t s = 0; s < 2; ++s) {
            infos[s][0] = {cells[s].buffer, 0, grid_bytes};
            infos[s][1] = {ker.buffer, 0, kernel_bytes};
            infos[s][2] = {cells[1 - s].buffer, 0, grid_bytes};
            if (words) infos[s][3] = {program.buffer, 0, program_bytes};
            if (use_fft) {
                infos[s][3] = {spec.buffer, 0, spectrum_bytes};
                infos[s][4] = {khat.buffer, 0, spectrum_bytes};
                infos[s][5] = {twiddles.buffer, 0, twiddle_bytes};
            }
            for (std::uint32_t b = 0; b < p.bindings; ++b) {
                writes[s][b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[s][b].dstSet = p.sets[s];
                writes[s][b].dstBinding = b;
                writes[s][b].descriptorCount = 1;
                writes[s][b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[s][b].pBufferInfo = &infos[s][b];
            }
        }
        // One call per set: a pipeline with 3 bindings leaves each row's last entry unused.
        vkUpdateDescriptorSets(device, p.bindings, writes[0], 0, nullptr);
        vkUpdateDescriptorSets(device, p.bindings, writes[1], 0, nullptr);

        check(vkResetCommandBuffer(command_buffer, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(command_buffer, &begin), "vkBeginCommandBuffer");
        if (query_pool) vkCmdResetQueryPool(command_buffer, query_pool, 0, kQueries);

        // Upload. Host writes to the staging buffer are made visible to the device implicitly
        // by vkQueueSubmit.
        const VkBufferCopy grid_copy{0, 0, grid_bytes};
        const VkBufferCopy kernel_copy{kernel_offset, 0, kernel_bytes};
        const VkBufferCopy program_copy{program_offset, 0, program_bytes};
        const VkBufferCopy twiddle_copy{twiddle_offset, 0, twiddle_bytes};
        vkCmdCopyBuffer(command_buffer, staging.buffer, cells[0].buffer, 1, &grid_copy);
        vkCmdCopyBuffer(command_buffer, staging.buffer, ker.buffer, 1, &kernel_copy);
        if (words) vkCmdCopyBuffer(command_buffer, staging.buffer, program.buffer, 1, &program_copy);
        if (upload_twiddles) vkCmdCopyBuffer(command_buffer, staging.buffer, twiddles.buffer, 1, &twiddle_copy);
        // Also covers a cached kernel spectrum (khat) an earlier submission's shaders wrote.
        VkMemoryBarrier uploaded{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        uploaded.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        uploaded.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &uploaded, 0, nullptr, 0, nullptr);

        // A dispatch reads what the previous one wrote (read-after-write) and may overwrite what
        // it read (write-after-read), so wait for it and make its writes visible.
        const auto between_dispatches = [&] {
            VkMemoryBarrier between{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            between.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            between.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &between, 0, nullptr, 0, nullptr);
        };
        FftPushConstants fpc{};
        if (use_fft) {
            fpc = {static_cast<std::uint32_t>(grid.width),
                   static_cast<std::uint32_t>(grid.height),
                   fft_log2(grid.width),
                   fft_log2(grid.height),
                   fft_log2(info.fft_max_size),
                   static_cast<std::uint32_t>(kernel.width),
                   static_cast<std::uint32_t>(kernel.height),
                   0,
                   1, // RowsInverse writes the next state
                   fft_growth->mu,
                   fft_growth->sigma,
                   fft_growth->dt,
                   static_cast<std::uint32_t>(fft_growth->function)};
        }
        const auto fft_pass = [&](FftPass pass) {
            fpc.pass = static_cast<std::uint32_t>(pass);
            vkCmdPushConstants(command_buffer, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(fpc), &fpc);
            vkCmdDispatch(command_buffer, fft_pass_groups(pass, grid.width, grid.height), 1, 1);
        };

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
        if (!use_fft) vkCmdPushConstants(command_buffer, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);
        // Bottom of pipe: written once every earlier command, the upload included, has
        // finished, so the intervals cover the dispatches only.
        if (query_pool) vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 0);
        if (transform) {
            // The kernel's spectrum into khat (the sets differ only in src and dst, unused here).
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.sets[0], 0,
                                    nullptr);
            fft_pass(FftPass::KernelRowsForward);
            between_dispatches();
            fft_pass(FftPass::KernelColumns);
        }
        if (query_pool) vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 1);
        for (std::size_t g = 0; g < generations; ++g) {
            if (g > 0 || transform) between_dispatches();
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.sets[g % 2], 0,
                                    nullptr);
            if (use_fft) {
                fft_pass(FftPass::RowsForward);
                between_dispatches();
                fft_pass(FftPass::ConvolveColumns);
                between_dispatches();
                fft_pass(FftPass::RowsInverse);
            } else {
                vkCmdDispatch(command_buffer, groups_x, groups_y, 1);
            }
        }
        if (query_pool) vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 2);

        // Readback of the last generation, cells[generations % 2], into the start of the
        // staging buffer (the upload copies from it finished before the first dispatch).
        VkMemoryBarrier computed{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        computed.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        computed.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             1, &computed, 0, nullptr, 0, nullptr);
        vkCmdCopyBuffer(command_buffer, cells[generations % 2].buffer, staging.buffer, 1, &grid_copy);
        // Make the copy visible to the host read below.
        VkMemoryBarrier downloaded{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        downloaded.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        downloaded.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                             &downloaded, 0, nullptr, 0, nullptr);
        check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

        // khat is being rewritten: stale until this submission is known to have completed.
        if (transform) fft_valid = false;
        check(vkResetFences(device, 1, &fence), "vkResetFences");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_buffer;
        check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
        check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        if (upload_twiddles) twiddles_uploaded = true;
        if (transform) {
            fft_kernel = kernel;
            fft_width = grid.width;
            fft_height = grid.height;
            fft_valid = true;
        }

        // The only readback of the batch.
        std::memcpy(next.cells.data(), staging.mapped, grid_bytes);

        if (query_pool) {
            std::uint64_t ticks[kQueries] = {0, 0, 0};
            check(vkGetQueryPoolResults(device, query_pool, 0, kQueries, sizeof(ticks), ticks, sizeof(std::uint64_t),
                                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                  "vkGetQueryPoolResults");
            // Only the low timestamp_bits are meaningful; differences wrap within them.
            const std::uint64_t mask = timestamp_bits >= 64 ? ~0ull : (1ull << timestamp_bits) - 1;
            const auto ms = [&](std::uint64_t from, std::uint64_t to) {
                return static_cast<double>((to - from) & mask) * properties.limits.timestampPeriod * 1e-6;
            };
            last_gpu_ms = ms(ticks[1], ticks[2]);
            if (transform) last_transform_ms = ms(ticks[0], ticks[1]);
        }
        return next;
    }

    Grid step(const Grid& grid, const Grid& kernel, const GrowthParams& params, std::size_t generations) {
        if (method == ConvolutionMethod::Fft && fft_supported(grid.width, grid.height)) {
            return run(fft, grid, kernel, nullptr, 0, generations, nullptr, &params);
        }
        const PushConstants pc{static_cast<std::uint32_t>(grid.width),   static_cast<std::uint32_t>(grid.height),
                               static_cast<std::uint32_t>(kernel.width), static_cast<std::uint32_t>(kernel.height),
                               params.mu,                                params.sigma,
                               params.dt,                                static_cast<std::uint32_t>(params.function)};
        return run(lenia, grid, kernel, &pc, sizeof(pc), generations, nullptr, nullptr);
    }

    Grid step_rule(const Grid& grid, const Grid& kernel, const RuleProgram& rule_program, std::size_t generations) {
        last_gpu_ms.reset();
        if (rule_program.ops.size() > kMaxRuleOps) throw std::invalid_argument("rule program has too many ops");
        const std::vector<std::uint32_t> words = pack_rule(rule_program);
        const RulePushConstants pc{static_cast<std::uint32_t>(grid.width), static_cast<std::uint32_t>(grid.height),
                                   static_cast<std::uint32_t>(kernel.width), static_cast<std::uint32_t>(kernel.height)};
        return run(rule, grid, kernel, &pc, sizeof(pc), generations, &words, nullptr);
    }
};

VulkanStepper::VulkanStepper() : impl_(std::make_unique<Impl>()) {}
VulkanStepper::~VulkanStepper() = default;
VulkanStepper::VulkanStepper(VulkanStepper&&) noexcept = default;
VulkanStepper& VulkanStepper::operator=(VulkanStepper&&) noexcept = default;

Grid VulkanStepper::step_lenia(const Grid& grid, const Grid& kernel, const GrowthParams& params) {
    return impl_->step(grid, kernel, params, 1);
}

Grid VulkanStepper::step_lenia(const Grid& grid, const Grid& kernel, const GrowthParams& params,
                               std::size_t generations) {
    return impl_->step(grid, kernel, params, generations);
}

Grid VulkanStepper::step_rule(const Grid& grid, const Grid& kernel, const RuleProgram& program,
                              std::size_t generations) {
    return impl_->step_rule(grid, kernel, program, generations);
}

void VulkanStepper::set_convolution(ConvolutionMethod method) { impl_->method = method; }

ConvolutionMethod VulkanStepper::convolution() const { return impl_->method; }

bool VulkanStepper::fft_supported(std::size_t width, std::size_t height) const {
    return impl_->fft_supported(width, height);
}

ConvolutionMethod VulkanStepper::last_convolution() const { return impl_->last_method; }

const std::string& VulkanStepper::device_name() const { return impl_->device_name; }

const VulkanDeviceInfo& VulkanStepper::device_info() const { return impl_->info; }

std::optional<double> VulkanStepper::last_gpu_ms() const { return impl_->last_gpu_ms; }

std::optional<double> VulkanStepper::last_kernel_transform_ms() const { return impl_->last_transform_ms; }

} // namespace lenia
