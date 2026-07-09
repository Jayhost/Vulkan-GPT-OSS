#include "vulkan_backend.h"
#include <iostream>
#include <cstring>
#include <stdexcept>

uint32_t findMemoryType(VkPhysicalDevice phys, uint32_t typeFilter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type!");
}

void VulkanBackend::fillBuffer(VkBuffer buffer, VkDeviceSize size, uint32_t data) {
    vkCmdFillBuffer(commandBuffer_, buffer, 0, size, data);
}

void VulkanBackend::uploadToDeviceLocal(VkBuffer dst, const void* data, VkDeviceSize size) {
    // Create temporary staging buffer in System RAM
    VkBuffer staging = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    copyToBuffer(staging, data, size);
    
    // Copy from System RAM to VRAM
    beginCommandBuffer();
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(commandBuffer_, staging, dst, 1, &region);
    endAndSubmitCommandBuffer();
    waitIdle(); // Wait for copy to finish
    
    // Free staging buffer
    vkDestroyBuffer(device_, staging, nullptr);
    vkFreeMemory(device_, bufferMemMap_[staging], nullptr);
    bufferMemMap_.erase(staging);
}

void VulkanBackend::init() {
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.apiVersion = VK_API_VERSION_1_1;
    
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&ici, nullptr, &instance_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance");
    }

    uint32_t gpuCount = 0;
    vkEnumeratePhysicalDevices(instance_, &gpuCount, nullptr);
    std::vector<VkPhysicalDevice> gpus(gpuCount);
    vkEnumeratePhysicalDevices(instance_, &gpuCount, gpus.data());
    physicalDevice_ = gpus[0];

    uint32_t qFamCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &qFamCount, nullptr);
    std::vector<VkQueueFamilyProperties> qFams(qFamCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &qFamCount, qFams.data());
    
    uint32_t computeQFam = 0;
    for (uint32_t i = 0; i < qFamCount; i++) {
        if (qFams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { computeQFam = i; break; }
    }

    float qPrio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = computeQFam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &qPrio;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    vkCreateDevice(physicalDevice_, &dci, nullptr, &device_);
    vkGetDeviceQueue(device_, computeQFam, 0, &computeQueue_);

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = computeQFam;
    vkCreateCommandPool(device_, &cpci, nullptr, &commandPool_);

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = commandPool_;
    cbai.commandBufferCount = 1;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    vkAllocateCommandBuffers(device_, &cbai, &commandBuffer_);

    VkDescriptorPoolSize poolSizes[] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 } };
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 1000;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = poolSizes;
    vkCreateDescriptorPool(device_, &dpci, nullptr, &descPool_);

    VkDescriptorSetLayoutBinding bindings[8] = {};
    for (int i=0; i<8; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 8;
    dslci.pBindings = bindings;
    vkCreateDescriptorSetLayout(device_, &dslci, nullptr, &descSetLayout_);

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size = 128;
    
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &descSetLayout_;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(device_, &plci, nullptr, &pipelineLayout_);

    // FIX: Allocate large enough dummy buffer (e.g. 4096 bytes for up to 1024 heads) and fully zero it
    dummy_buf = createBuffer(4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    std::vector<uint8_t> zero(4096, 0);
    copyToBuffer(dummy_buf, zero.data(), 4096);
}       

VkBuffer VulkanBackend::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buf;
    if (vkCreateBuffer(device_, &bci, nullptr, &buf) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer");
    }
    
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device_, buf, &memReq);
    
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = memReq.size;
    mai.memoryTypeIndex = findMemoryType(physicalDevice_, memReq.memoryTypeBits, props);
    
    VkDeviceMemory mem;
    if (vkAllocateMemory(device_, &mai, nullptr, &mem) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate buffer memory");
    }
    vkBindBufferMemory(device_, buf, mem, 0);
    
    bufferMemMap_[buf] = mem;
    return buf;
}

void VulkanBackend::copyToBuffer(VkBuffer buffer, const void* data, VkDeviceSize size) {
    void* mapped;
    vkMapMemory(device_, bufferMemMap_[buffer], 0, size, 0, &mapped);
    memcpy(mapped, data, (size_t)size);
    vkUnmapMemory(device_, bufferMemMap_[buffer]);
}

void VulkanBackend::copyFromBuffer(VkBuffer buffer, void* data, VkDeviceSize size) {
    void* mapped;
    vkMapMemory(device_, bufferMemMap_[buffer], 0, size, 0, &mapped);
    memcpy(data, mapped, (size_t)size);
    vkUnmapMemory(device_, bufferMemMap_[buffer]);
}

VkShaderModule VulkanBackend::createShader(const std::vector<uint32_t>& spirv) {
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spirv.size() * 4;
    smci.pCode = spirv.data();
    VkShaderModule mod;
    vkCreateShaderModule(device_, &smci, nullptr, &mod);
    return mod;
}

void VulkanBackend::beginCommandBuffer() {
    vkResetCommandBuffer(commandBuffer_, 0);
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer_, &cbbi);
}

void VulkanBackend::dispatch(VkShaderModule shader, uint32_t groupX, uint32_t groupY, uint32_t groupZ,
                             const std::vector<VkBuffer>& bindings, const void* pushConstData, size_t pushConstSize) {
    // 1. Fetch or Create Pipeline
    VkPipeline pipeline;
    auto it = pipelineCache_.find(shader);
    if (it != pipelineCache_.end()) {
        pipeline = it->second;
    } else {
        VkComputePipelineCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pci.stage.module = shader;
        pci.stage.pName = "main";
        pci.layout = pipelineLayout_;
        
        if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute pipeline");
        }
        pipelineCache_[shader] = pipeline;
    }

    // 2. Allocate Descriptor Set
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = descPool_;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &descSetLayout_;
    VkDescriptorSet descSet;
    if (vkAllocateDescriptorSets(device_, &dsai, &descSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }

    std::vector<VkDescriptorBufferInfo> bufInfos;
    std::vector<VkWriteDescriptorSet> writes;
    bufInfos.reserve(bindings.size());
    writes.reserve(bindings.size());
    
    for (size_t i = 0; i < bindings.size(); i++) {
        VkBuffer buf = (bindings[i] != VK_NULL_HANDLE) ? bindings[i] : dummy_buf;
        bufInfos.push_back({buf, 0, VK_WHOLE_SIZE});
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET});
        writes.back().dstSet = descSet;
        writes.back().dstBinding = i;
        writes.back().descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes.back().descriptorCount = 1;
        writes.back().pBufferInfo = &bufInfos.back();
    }
    vkUpdateDescriptorSets(device_, writes.size(), writes.data(), 0, nullptr);

    // 3. Record commands
    vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &descSet, 0, nullptr);
    if (pushConstData && pushConstSize > 0) {
        vkCmdPushConstants(commandBuffer_, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstSize, pushConstData);
    }
    vkCmdDispatch(commandBuffer_, groupX, groupY, groupZ);

    // 4. Memory Barrier
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(commandBuffer_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

    pendingDescSets_.push_back(descSet); // Pipelines no longer pushed, they are cached!
}


void VulkanBackend::endAndSubmitCommandBuffer() {
    vkEndCommandBuffer(commandBuffer_);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &commandBuffer_;
    vkQueueSubmit(computeQueue_, 1, &si, VK_NULL_HANDLE);
}

void VulkanBackend::waitIdle() {
    vkQueueWaitIdle(computeQueue_);
    for (auto p : pendingPipelines_) vkDestroyPipeline(device_, p, nullptr);
    for (auto d : pendingDescSets_) vkFreeDescriptorSets(device_, descPool_, 1, &d);
    pendingPipelines_.clear();
    pendingDescSets_.clear();
}

void VulkanBackend::cleanup() {
    waitIdle();
    for (auto& pair : pipelineCache_) {
        vkDestroyPipeline(device_, pair.second, nullptr);
    }
    for (auto& pair : bufferMemMap_) {
        vkDestroyBuffer(device_, pair.first, nullptr);
        vkFreeMemory(device_, pair.second, nullptr);
    }
    vkDestroyDescriptorPool(device_, descPool_, nullptr);
    vkDestroyDescriptorSetLayout(device_, descSetLayout_, nullptr);
    vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    vkDestroyCommandPool(device_, commandPool_, nullptr);
    vkDestroyDevice(device_, nullptr);
    vkDestroyInstance(instance_, nullptr);
}