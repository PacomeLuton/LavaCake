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

#include "VulkanFence.h" 

namespace LavaCake {

	namespace Fence {
		bool CreateFence(VkDevice   logical_device,
			bool       signaled,
			VkFence  & fence) {
			VkFenceCreateInfo fence_create_info = {
				VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,          // VkStructureType        sType
				nullptr,                                      // const void           * pNext
				signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0u, // VkFenceCreateFlags     flags
			};

			VkResult result = vkCreateFence(logical_device, &fence_create_info, nullptr, &fence);
			if (VK_SUCCESS != result) {
				std::cout << "Could not create a fence." << std::endl;
				return false;
			}
			return true;
		}

		bool WaitForFences(VkDevice                     logical_device,
			std::vector<VkFence> const & fences,
			VkBool32                     wait_for_all,
			uint64_t                     timeout) {
			if (fences.size() > 0) {
				VkResult result = vkWaitForFences(logical_device, static_cast<uint32_t>(fences.size()), fences.data(), wait_for_all, timeout);
				if (VK_SUCCESS != result) {
					std::cout << "Waiting on fence failed." << std::endl;
					return false;
				}
				return true;
			}
			return false;
		}

		bool ResetFences(VkDevice                     logical_device,
			std::vector<VkFence> const & fences) {
			if (fences.size() > 0) {
				VkResult result = vkResetFences(logical_device, static_cast<uint32_t>(fences.size()), fences.data());
				if (VK_SUCCESS != result) {
					std::cout << "Error occurred when tried to reset fences." << std::endl;
					return false;
				}
				return VK_SUCCESS == result;
			}
			return false;
		}

		void DestroyFence(VkDevice   logical_device,
			VkFence  & fence) {
			if (VK_NULL_HANDLE != fence) {
				vkDestroyFence(logical_device, fence, nullptr);
				fence = VK_NULL_HANDLE;
			}
		}
	}

}