#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <unordered_map>

struct DescSetKey {
    VkShaderModule shader;
    VkBuffer b[8];
    bool operator==(const DescSetKey& o) const {
        if (shader != o.shader) return false;
        for (int i = 0; i < 8; i++) if (b[i] != o.b[i]) return false;
        return true;
    }
};

struct DescSetKeyHash {
    size_t operator()(const DescSetKey& k) const {
        size_t h = std::hash<uint64_t>()((uint64_t)k.shader);
        for (int i = 0; i < 8; i++) {
            h ^= std::hash<uint64_t>()((uint64_t)k.b[i]) << i;
        }
        return h;
    }
};

class VulkanBackend {
public:
    void init();
    void cleanup();

    VkBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props);
    void uploadData(VkBuffer dst, const void* data, VkDeviceSize size);
    void readbackData(VkBuffer src, VkDeviceSize dst_offset, VkDeviceSize size);
    void* getReadbackPtr() const { return readbackMapped_; }
    
    void fillBuffer(VkBuffer buffer, VkDeviceSize size, uint32_t data);

    VkShaderModule createShader(const std::vector<uint32_t>& spirv);
    void destroyShader(VkShaderModule shader) { if (shader) vkDestroyShaderModule(device_, shader, nullptr); }
    
    void beginCommandBuffer();
    void dispatch(VkShaderModule shader, uint32_t groupX, uint32_t groupY, uint32_t groupZ,
                  const std::vector<VkBuffer>& bindings, const void* pushConstData, size_t pushConstSize);
    // FIXED: added include_transfer parameter
    void barrier(const std::vector<VkBuffer>& bufs, bool include_transfer = false);
    void endAndSubmitCommandBuffer();
    
    void waitFence();
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
    VkFence fence_;

    std::unordered_map<VkBuffer, VkDeviceMemory> bufferMemMap_;
    std::vector<VkPipeline> pendingPipelines_;
    std::unordered_map<VkShaderModule, VkPipeline> pipelineCache_;
    
    // Cache for descriptor sets
    std::unordered_map<DescSetKey, VkDescriptorSet, DescSetKeyHash> descSetCache_;

    // Staging ring buffer for host-to-device transfers
    VkBuffer stagingBuffer_;
    VkDeviceMemory stagingMem_;
    void* stagingMapped_ = nullptr;
    VkDeviceSize stagingSize_ = 16 * 1024 * 1024; // 16 MB
    VkDeviceSize stagingOffset_ = 0;

    // Readback buffer for device-to-host transfers
    VkBuffer readbackBuffer_;
    VkDeviceMemory readbackMem_;
    void* readbackMapped_ = nullptr;
};