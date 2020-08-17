// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and / or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The below copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Vulkan Cookbook
// ISBN: 9781786468154
// � Packt Publishing Limited
//
// Author:   Pawel Lapinski
// LinkedIn: https://www.linkedin.com/in/pawel-lapinski-84522329

#ifndef VULKAN_DEVICE
#define VULKAN_DEVICE

#include "Common.h"
#include "VulkanQueue.h"

namespace LavaCake {
	namespace Device {
		namespace Physical {
			bool EnumerateAvailablePhysicalDevices(VkInstance                      instance,
				std::vector<VkPhysicalDevice> & available_devices);

			bool CheckAvailableDeviceExtensions(VkPhysicalDevice                     physical_device,
				std::vector<VkExtensionProperties> & available_extensions);

			void GetFeaturesAndPropertiesOfPhysicalDevice(VkPhysicalDevice             physical_device,
				VkPhysicalDeviceFeatures   & device_features,
				VkPhysicalDeviceProperties & device_properties);
		}
		namespace Logical {
			bool CreateLogicalDevice(VkPhysicalDevice                  physical_device,
				std::vector< Queue::QueueInfo >          queue_infos,
				std::vector<char const *> const & desired_extensions,
				VkPhysicalDeviceFeatures        * desired_features,
				VkDevice                        & logical_device);

			bool CreateLogicalDeviceWithGeometryShadersAndGraphicsAndComputeQueues(VkInstance   instance,
				VkDevice   & logical_device,
				VkQueue    & graphics_queue,
				VkQueue    & compute_queue);

			void DestroyLogicalDevice(VkDevice & logical_device);
		}
	}
}

#endif