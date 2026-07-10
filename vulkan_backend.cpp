#include "vulkan_backend.h"
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <algorithm>

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

    VkDescriptorPoolSize poolSizes[] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100000 } };
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 10000;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = poolSizes;
    vkCreateDescriptorPool(device_, &dpci, nullptr, &descPool_);

    VkDescriptorSetLayoutBinding bindings[8] = {};
    for (int i=0; i<8; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
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

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(device_, &fci, nullptr, &fence_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fence");
    }

    dummy_buf = createBuffer(4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    void* mapped_dummy;
    vkMapMemory(device_, bufferMemMap_[dummy_buf], 0, 4096, 0, &mapped_dummy);
    memset(mapped_dummy, 0, 4096);
    vkUnmapMemory(device_, bufferMemMap_[dummy_buf]);

    // Persistently mapped Staging Buffer
    stagingBuffer_ = createBuffer(stagingSize_, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    stagingMem_ = bufferMemMap_[stagingBuffer_];
    vkMapMemory(device_, stagingMem_, 0, stagingSize_, 0, &stagingMapped_);

    // Persistently mapped Readback Buffer (4 MB, enough for 2x 128k vocab floats)
    VkDeviceSize rbSize = 4 * 1024 * 1024;
    readbackBuffer_ = createBuffer(rbSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    readbackMem_ = bufferMemMap_[readbackBuffer_];
    vkMapMemory(device_, readbackMem_, 0, rbSize, 0, &readbackMapped_);
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

void VulkanBackend::uploadData(VkBuffer dst, const void* data, VkDeviceSize size) {
    VkDeviceSize offset = 0;
    while (offset < size) {
        stagingOffset_ = (stagingOffset_ + 255) & ~255; // Align to 256 bytes
        if (stagingOffset_ + size > stagingSize_) {
            endAndSubmitCommandBuffer();
            waitFence();
            beginCommandBuffer();
            stagingOffset_ = (stagingOffset_ + 255) & ~255;
        }
        
        VkDeviceSize chunk = std::min(size - offset, stagingSize_ - stagingOffset_);
        memcpy((uint8_t*)stagingMapped_ + stagingOffset_, (const uint8_t*)data + offset, chunk);
        
        VkBufferCopy region{stagingOffset_, offset, chunk};
        vkCmdCopyBuffer(commandBuffer_, stagingBuffer_, dst, 1, &region);
        
        stagingOffset_ += chunk;
        offset += chunk;
    }
}

void VulkanBackend::readbackData(VkBuffer src, VkDeviceSize dst_offset, VkDeviceSize size) {
    VkBufferCopy region{0, dst_offset, size};
    vkCmdCopyBuffer(commandBuffer_, src, readbackBuffer_, 1, &region);
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
    waitFence();
    vkResetFences(device_, 1, &fence_);
    vkResetCommandBuffer(commandBuffer_, 0);
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer_, &cbbi);
    stagingOffset_ = 0; // Safe to reset since GPU is idle
}

void VulkanBackend::barrier(const std::vector<VkBuffer>& bufs) {
    if (bufs.empty()) return;

    std::vector<VkBufferMemoryBarrier> bmb;
    bmb.reserve(bufs.size());
    for (VkBuffer b : bufs) {
        if (b == VK_NULL_HANDLE) continue;
        VkBufferMemoryBarrier bar{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.buffer = b;
        bar.offset = 0;
        bar.size = VK_WHOLE_SIZE;
        bmb.push_back(bar);
    }
    if (bmb.empty()) return;

    vkCmdPipelineBarrier(commandBuffer_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr,
        (uint32_t)bmb.size(), bmb.data(),
        0, nullptr);
}

void VulkanBackend::dispatch(VkShaderModule shader, uint32_t groupX, uint32_t groupY, uint32_t groupZ,
                             const std::vector<VkBuffer>& bindings, const void* pushConstData, size_t pushConstSize) {
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

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = descPool_;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &descSetLayout_;
    VkDescriptorSet descSet;
    if (vkAllocateDescriptorSets(device_, &dsai, &descSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }

    VkDescriptorBufferInfo bufInfos[8];
    VkWriteDescriptorSet writes[8];
    for (size_t i = 0; i < 8; i++) {
        VkBuffer buf = (i < bindings.size() && bindings[i] != VK_NULL_HANDLE) ? bindings[i] : dummy_buf;
        bufInfos[i] = {buf, 0, VK_WHOLE_SIZE};
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = descSet;
        writes[i].dstBinding = i;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(device_, 8, writes, 0, nullptr);

    vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &descSet, 0, nullptr);
    if (pushConstData && pushConstSize > 0) {
        vkCmdPushConstants(commandBuffer_, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstSize, pushConstData);
    }
    vkCmdDispatch(commandBuffer_, groupX, groupY, groupZ);

    pendingDescSets_.push_back(descSet);
}

void VulkanBackend::endAndSubmitCommandBuffer() {
    vkEndCommandBuffer(commandBuffer_);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &commandBuffer_;
    vkQueueSubmit(computeQueue_, 1, &si, fence_);
}

void VulkanBackend::waitFence() {
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    for (auto p : pendingPipelines_) vkDestroyPipeline(device_, p, nullptr);
    for (auto d : pendingDescSets_) vkFreeDescriptorSets(device_, descPool_, 1, &d);
    pendingPipelines_.clear();
    pendingDescSets_.clear();
}

void VulkanBackend::waitIdle() {
    waitFence();
}

void VulkanBackend::cleanup() {
    waitIdle();
    
    if (stagingMapped_) vkUnmapMemory(device_, stagingMem_);
    if (readbackMapped_) vkUnmapMemory(device_, readbackMem_);

    if (fence_) vkDestroyFence(device_, fence_, nullptr);
    for (auto& pair : pipelineCache_) vkDestroyPipeline(device_, pair.second, nullptr);
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