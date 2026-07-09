#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <unordered_map>

class VulkanBackend {
public:
    void init();
    void cleanup();

    VkBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props);
    void copyToBuffer(VkBuffer buffer, const void* data, VkDeviceSize size);
    void copyFromBuffer(VkBuffer buffer, void* data, VkDeviceSize size);
    
    // New: Fast GPU-side buffer zeroing
    void fillBuffer(VkBuffer buffer, VkDeviceSize size, uint32_t data);
    // New: Upload to VRAM using a staging buffer
    void uploadToDeviceLocal(VkBuffer dst, const void* data, VkDeviceSize size);

    VkShaderModule createShader(const std::vector<uint32_t>& spirv);
    void destroyShader(VkShaderModule shader) { if (shader) vkDestroyShaderModule(device_, shader, nullptr); }
    
    void beginCommandBuffer();
    void dispatch(VkShaderModule shader, uint32_t groupX, uint32_t groupY, uint32_t groupZ,
                  const std::vector<VkBuffer>& bindings, const void* pushConstData, size_t pushConstSize);
    void endAndSubmitCommandBuffer();
    void waitIdle();

    VkDevice device() const { return device_; }
    VkBuffer dummy_buf;

private:
    VkInstance instance_;
    VkPhysicalDevice physicalDevice_;
    VkDevice device_;
    VkQueue computeQueue_;
    VkCommandPool commandPool_;
    VkCommandBuffer commandBuffer_;
    VkDescriptorPool descPool_;
    VkDescriptorSetLayout descSetLayout_;
    VkPipelineLayout pipelineLayout_;

    std::unordered_map<VkBuffer, VkDeviceMemory> bufferMemMap_;
    std::vector<VkPipeline> pendingPipelines_;
    std::vector<VkDescriptorSet> pendingDescSets_;
    // New: Pipeline cache
    std::unordered_map<VkShaderModule, VkPipeline> pipelineCache_;
};