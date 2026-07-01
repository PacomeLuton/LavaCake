#include <LavaCake/Raytracing/ShaderBindingTable.h>

namespace LavaCake {
  namespace RayTracing {

    void ShaderBindingTable::compile(const Framework::Queue& queue, Framework::CommandBuffer& cmdBuff, VkPipeline raytracingPipeline) {
      Framework::Device* d = Framework::Device::getDevice();
      VkDevice logical = d->getLogicalDevice();
      VkPhysicalDevice physical = d->getPhysicalDevice();

      VkPhysicalDeviceRayTracingPipelinePropertiesKHR  rayTracingPipelineProperties{};

      rayTracingPipelineProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
      VkPhysicalDeviceProperties2 deviceProperties2{};
      deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
      deviceProperties2.pNext = &rayTracingPipelineProperties;
      vkGetPhysicalDeviceProperties2(physical, &deviceProperties2);

      const uint32_t handleSize = rayTracingPipelineProperties.shaderGroupHandleSize;
      const uint32_t handleSizeAligned = alignedSize(rayTracingPipelineProperties.shaderGroupHandleSize, rayTracingPipelineProperties.shaderGroupHandleAlignment);
      const uint32_t groupCount = static_cast<uint32_t>(m_rayGen.size() + m_miss.size() + m_hitGroup.size() + m_callable.size());
      const uint32_t sbtSize = groupCount * handleSizeAligned;

      std::vector<uint8_t> shaderHandleStorage(sbtSize);
      vkGetRayTracingShaderGroupHandlesKHR(logical, raytracingPipeline, 0, groupCount, sbtSize, shaderHandleStorage.data());

      // Create buffer to hold all shader handles for the SBT
      
      const VkBufferUsageFlags sbtUsage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      const VkMemoryPropertyFlags sbtMemory = VkMemoryPropertyFlagBits(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

      auto allocSbtBuffer = [&](std::shared_ptr<Framework::Buffer>& buf, VkStridedDeviceAddressRegionKHR& region,
                                const std::vector<entry>& entries) -> void* {
        if (entries.empty()) return nullptr;
        buf = std::make_shared<Framework::Buffer>(handleSize * entries.size(), sbtUsage, sbtMemory);
        region = {};
        region.deviceAddress = buf->getBufferDeviceAddress();
        region.stride        = handleSizeAligned;
        region.size          = entries.size() * handleSizeAligned;
        void* mem = nullptr;
        vkMapMemory(logical, buf->getMemory(), 0, VK_WHOLE_SIZE, 0, &mem);
        return mem;
      };

      void* raygenMem   = allocSbtBuffer(m_raygenBuffer,   m_raygenShaderBindingTable,   m_rayGen);
      void* missMem     = allocSbtBuffer(m_missBuffer,     m_missShaderBindingTable,     m_miss);
      void* hitMem      = allocSbtBuffer(m_hitBuffer,      m_hitShaderBindingTable,      m_hitGroup);
      void* callableMem = allocSbtBuffer(m_callableBuffer, m_callableShaderBindingTable, m_callable);

      // Copy handles
      if (raygenMem)
        memcpy(raygenMem,   shaderHandleStorage.data() + handleSizeAligned * 0,                                                    handleSize * m_rayGen.size());
      if (missMem)
        memcpy(missMem,     shaderHandleStorage.data() + handleSizeAligned * m_rayGen.size(),                                      handleSize * m_miss.size());
      if (hitMem)
        memcpy(hitMem,      shaderHandleStorage.data() + handleSizeAligned * (m_rayGen.size() + m_miss.size()),                    handleSize * m_hitGroup.size());
      if (callableMem)
        memcpy(callableMem, shaderHandleStorage.data() + handleSizeAligned * (m_rayGen.size() + m_miss.size() + m_hitGroup.size()), handleSize * m_callable.size());
    }
  }
}