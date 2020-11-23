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

#ifndef VULKAN_VIEWPORT
#define VULKAN_VIEWPORT

#include "Common.h"

namespace LavaCake {
	namespace Core {
		struct ViewportInfo {
			std::vector<VkViewport>   Viewports;
			std::vector<VkRect2D>     Scissors;
		};
		void SetViewportStateDynamically(VkCommandBuffer                 command_buffer,
			uint32_t                        first_viewport,
			std::vector<VkViewport> const & viewports);

		void SpecifyPipelineViewportAndScissorTestState(ViewportInfo const                & viewport_infos,
			VkPipelineViewportStateCreateInfo & viewport_state_create_info);

		void SetScissorStateDynamically(
			VkCommandBuffer               command_buffer,
			uint32_t                      first_scissor,
			std::vector<VkRect2D> const & scissors);
	}
}

#endif // !VULKAN_VIEWPORT
