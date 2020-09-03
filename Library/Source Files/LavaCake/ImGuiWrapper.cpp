#include "ImGuiWrapper.h"
#include <stdio.h>



namespace LavaCake {
  namespace Framework {

    static void ImGui_ImplGlfw_Mouse_Position(GLFWwindow* window, double xpos, double ypos) {
      ImGuiIO& io = ImGui::GetIO();
      io.MousePos = ImVec2((float)xpos, (float)ypos);
    }
    static const char* ImGui_ImplGlfw_GetClipboardText(void* user_data)
    {
      return glfwGetClipboardString((GLFWwindow*)user_data);
    }

    static void ImGui_ImplGlfw_SetClipboardText(void* user_data, const char* text)
    {
      glfwSetClipboardString((GLFWwindow*)user_data, text);
    }

    void ImGui_ImplGlfw_MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
    {
//      if (g_PrevUserCallbackMousebutton != NULL)
//        g_PrevUserCallbackMousebutton(window, button, action, mods);

      if (action == GLFW_PRESS && button >= 0) {
        ImGuiIO& io = ImGui::GetIO();
        io.MouseDown[button] = true;
      }
      if (action == GLFW_RELEASE && button >= 0) {
        ImGuiIO& io = ImGui::GetIO();
        io.MouseDown[button] = false;
      }
    }

    void ImGui_ImplGlfw_ScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
    {
//      if (g_PrevUserCallbackScroll != NULL)
//        g_PrevUserCallbackScroll(window, xoffset, yoffset);

      ImGuiIO& io = ImGui::GetIO();
      io.MouseWheelH += (float)xoffset;
      io.MouseWheel += (float)yoffset;
    }

    void ImGui_ImplGlfw_KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
//      if (g_PrevUserCallbackKey != NULL)
//        g_PrevUserCallbackKey(window, key, scancode, action, mods);

      ImGuiIO& io = ImGui::GetIO();
      if (action == GLFW_PRESS)
        io.KeysDown[key] = true;
      if (action == GLFW_RELEASE)
        io.KeysDown[key] = false;

      // Modifiers are not reliable across systems
      io.KeyCtrl = io.KeysDown[GLFW_KEY_LEFT_CONTROL] || io.KeysDown[GLFW_KEY_RIGHT_CONTROL];
      io.KeyShift = io.KeysDown[GLFW_KEY_LEFT_SHIFT] || io.KeysDown[GLFW_KEY_RIGHT_SHIFT];
      io.KeyAlt = io.KeysDown[GLFW_KEY_LEFT_ALT] || io.KeysDown[GLFW_KEY_RIGHT_ALT];
#ifdef _WIN32
      io.KeySuper = false;
#else
      io.KeySuper = io.KeysDown[GLFW_KEY_LEFT_SUPER] || io.KeysDown[GLFW_KEY_RIGHT_SUPER];
#endif
    }

    void ImGui_ImplGlfw_CharCallback(GLFWwindow* window, unsigned int c)
    {
//      if (g_PrevUserCallbackChar != NULL)
//        g_PrevUserCallbackChar(window, c);

      ImGuiIO& io = ImGui::GetIO();
      io.AddInputCharacter(c);
    }

    // glsl_shader.vert, compiled with:
    // # glslangValidator -V -x -o glsl_shader.vert.u32 glsl_shader.vert
    /*
    #version 450 core
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in vec2 aUV;
    layout(location = 2) in vec4 aColor;
    layout(push_constant) uniform uPushConstant { vec2 uScale; vec2 uTranslate; } pc;

    out gl_PerVertex { vec4 gl_Position; };
    layout(location = 0) out struct { vec4 Color; vec2 UV; } Out;

    void main()
    {
      Out.Color = aColor;
      Out.UV = aUV;
      gl_Position = vec4(aPos * pc.uScale + pc.uTranslate, 0, 1);
    }
    */



    static const uint32_t __glsl_shader_vert_spv[] =
    {
        0x07230203,0x00010000,0x00080001,0x0000002e,0x00000000,0x00020011,0x00000001,0x0006000b,
        0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
        0x000a000f,0x00000000,0x00000004,0x6e69616d,0x00000000,0x0000000b,0x0000000f,0x00000015,
        0x0000001b,0x0000001c,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,
        0x00000000,0x00030005,0x00000009,0x00000000,0x00050006,0x00000009,0x00000000,0x6f6c6f43,
        0x00000072,0x00040006,0x00000009,0x00000001,0x00005655,0x00030005,0x0000000b,0x0074754f,
        0x00040005,0x0000000f,0x6c6f4361,0x0000726f,0x00030005,0x00000015,0x00565561,0x00060005,
        0x00000019,0x505f6c67,0x65567265,0x78657472,0x00000000,0x00060006,0x00000019,0x00000000,
        0x505f6c67,0x7469736f,0x006e6f69,0x00030005,0x0000001b,0x00000000,0x00040005,0x0000001c,
        0x736f5061,0x00000000,0x00060005,0x0000001e,0x73755075,0x6e6f4368,0x6e617473,0x00000074,
        0x00050006,0x0000001e,0x00000000,0x61635375,0x0000656c,0x00060006,0x0000001e,0x00000001,
        0x61725475,0x616c736e,0x00006574,0x00030005,0x00000020,0x00006370,0x00040047,0x0000000b,
        0x0000001e,0x00000000,0x00040047,0x0000000f,0x0000001e,0x00000002,0x00040047,0x00000015,
        0x0000001e,0x00000001,0x00050048,0x00000019,0x00000000,0x0000000b,0x00000000,0x00030047,
        0x00000019,0x00000002,0x00040047,0x0000001c,0x0000001e,0x00000000,0x00050048,0x0000001e,
        0x00000000,0x00000023,0x00000000,0x00050048,0x0000001e,0x00000001,0x00000023,0x00000008,
        0x00030047,0x0000001e,0x00000002,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,
        0x00030016,0x00000006,0x00000020,0x00040017,0x00000007,0x00000006,0x00000004,0x00040017,
        0x00000008,0x00000006,0x00000002,0x0004001e,0x00000009,0x00000007,0x00000008,0x00040020,
        0x0000000a,0x00000003,0x00000009,0x0004003b,0x0000000a,0x0000000b,0x00000003,0x00040015,
        0x0000000c,0x00000020,0x00000001,0x0004002b,0x0000000c,0x0000000d,0x00000000,0x00040020,
        0x0000000e,0x00000001,0x00000007,0x0004003b,0x0000000e,0x0000000f,0x00000001,0x00040020,
        0x00000011,0x00000003,0x00000007,0x0004002b,0x0000000c,0x00000013,0x00000001,0x00040020,
        0x00000014,0x00000001,0x00000008,0x0004003b,0x00000014,0x00000015,0x00000001,0x00040020,
        0x00000017,0x00000003,0x00000008,0x0003001e,0x00000019,0x00000007,0x00040020,0x0000001a,
        0x00000003,0x00000019,0x0004003b,0x0000001a,0x0000001b,0x00000003,0x0004003b,0x00000014,
        0x0000001c,0x00000001,0x0004001e,0x0000001e,0x00000008,0x00000008,0x00040020,0x0000001f,
        0x00000009,0x0000001e,0x0004003b,0x0000001f,0x00000020,0x00000009,0x00040020,0x00000021,
        0x00000009,0x00000008,0x0004002b,0x00000006,0x00000028,0x00000000,0x0004002b,0x00000006,
        0x00000029,0x3f800000,0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,
        0x00000005,0x0004003d,0x00000007,0x00000010,0x0000000f,0x00050041,0x00000011,0x00000012,
        0x0000000b,0x0000000d,0x0003003e,0x00000012,0x00000010,0x0004003d,0x00000008,0x00000016,
        0x00000015,0x00050041,0x00000017,0x00000018,0x0000000b,0x00000013,0x0003003e,0x00000018,
        0x00000016,0x0004003d,0x00000008,0x0000001d,0x0000001c,0x00050041,0x00000021,0x00000022,
        0x00000020,0x0000000d,0x0004003d,0x00000008,0x00000023,0x00000022,0x00050085,0x00000008,
        0x00000024,0x0000001d,0x00000023,0x00050041,0x00000021,0x00000025,0x00000020,0x00000013,
        0x0004003d,0x00000008,0x00000026,0x00000025,0x00050081,0x00000008,0x00000027,0x00000024,
        0x00000026,0x00050051,0x00000006,0x0000002a,0x00000027,0x00000000,0x00050051,0x00000006,
        0x0000002b,0x00000027,0x00000001,0x00070050,0x00000007,0x0000002c,0x0000002a,0x0000002b,
        0x00000028,0x00000029,0x00050041,0x00000011,0x0000002d,0x0000001b,0x0000000d,0x0003003e,
        0x0000002d,0x0000002c,0x000100fd,0x00010038
    };

    // glsl_shader.frag, compiled with:
    // # glslangValidator -V -x -o glsl_shader.frag.u32 glsl_shader.frag
    /*
    #version 450 core
    layout(location = 0) out vec4 fColor;
    layout(set=0, binding=0) uniform sampler2D sTexture;
    layout(location = 0) in struct { vec4 Color; vec2 UV; } In;
    void main()
    {
        fColor = In.Color * texture(sTexture, In.UV.st);
    }
    */
    static uint32_t __glsl_shader_frag_spv[] =
    {
        0x07230203,0x00010000,0x00080001,0x0000001e,0x00000000,0x00020011,0x00000001,0x0006000b,
        0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
        0x0007000f,0x00000004,0x00000004,0x6e69616d,0x00000000,0x00000009,0x0000000d,0x00030010,
        0x00000004,0x00000007,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,
        0x00000000,0x00040005,0x00000009,0x6c6f4366,0x0000726f,0x00030005,0x0000000b,0x00000000,
        0x00050006,0x0000000b,0x00000000,0x6f6c6f43,0x00000072,0x00040006,0x0000000b,0x00000001,
        0x00005655,0x00030005,0x0000000d,0x00006e49,0x00050005,0x00000016,0x78655473,0x65727574,
        0x00000000,0x00040047,0x00000009,0x0000001e,0x00000000,0x00040047,0x0000000d,0x0000001e,
        0x00000000,0x00040047,0x00000016,0x00000022,0x00000000,0x00040047,0x00000016,0x00000021,
        0x00000000,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00030016,0x00000006,
        0x00000020,0x00040017,0x00000007,0x00000006,0x00000004,0x00040020,0x00000008,0x00000003,
        0x00000007,0x0004003b,0x00000008,0x00000009,0x00000003,0x00040017,0x0000000a,0x00000006,
        0x00000002,0x0004001e,0x0000000b,0x00000007,0x0000000a,0x00040020,0x0000000c,0x00000001,
        0x0000000b,0x0004003b,0x0000000c,0x0000000d,0x00000001,0x00040015,0x0000000e,0x00000020,
        0x00000001,0x0004002b,0x0000000e,0x0000000f,0x00000000,0x00040020,0x00000010,0x00000001,
        0x00000007,0x00090019,0x00000013,0x00000006,0x00000001,0x00000000,0x00000000,0x00000000,
        0x00000001,0x00000000,0x0003001b,0x00000014,0x00000013,0x00040020,0x00000015,0x00000000,
        0x00000014,0x0004003b,0x00000015,0x00000016,0x00000000,0x0004002b,0x0000000e,0x00000018,
        0x00000001,0x00040020,0x00000019,0x00000001,0x0000000a,0x00050036,0x00000002,0x00000004,
        0x00000000,0x00000003,0x000200f8,0x00000005,0x00050041,0x00000010,0x00000011,0x0000000d,
        0x0000000f,0x0004003d,0x00000007,0x00000012,0x00000011,0x0004003d,0x00000014,0x00000017,
        0x00000016,0x00050041,0x00000019,0x0000001a,0x0000000d,0x00000018,0x0004003d,0x0000000a,
        0x0000001b,0x0000001a,0x00050057,0x00000007,0x0000001c,0x00000017,0x0000001b,0x00050085,
        0x00000007,0x0000001d,0x00000012,0x0000001c,0x0003003e,0x00000009,0x0000001d,0x000100fd,
        0x00010038
    };

    void ImGuiWrapper::initGui(Window* win, Queue* queue, CommandBuffer* cmdBuff) {
      IMGUI_CHECKVERSION();
      ImGui::CreateContext();
      ImGuiIO& io = ImGui::GetIO(); (void)io;
      ImGui::StyleColorsDark();
      prepareImput(win->m_window);

      Framework::Device* d = Framework::Device::getDevice();
      LavaCake::Framework::SwapChain* s = LavaCake::Framework::SwapChain::getSwapChain();
      VkExtent2D size = s->size();
      VkDevice logical = d->getLogicalDevice();

      unsigned char* pixels;
      int width, height;
      io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
      std::vector<unsigned char>* textureData = new std::vector<unsigned char>(pixels, pixels + width * height * 4);

      TextureBuffer* fontBuffer = new TextureBuffer(textureData, width, height, 4);
      fontBuffer->allocate(queue->getHandle(), cmdBuff->getHandle());

      
      pass = new RenderPass();


      mesh = {
            {
          // positions
          -1.0f, -1.0f,
           0.0f, 0.0f,
           0.0f,  0.0f, 0.0f, 0.0f,
          -1.0f,  1.0f,
           0.0f, 1.0f,
           0.0f,  0.0f, 0.0f, 0.0f,
           1.0f, -1.0f,
           1.0f, 0.0f,
           0.0f,  0.0f, 0.0f, 0.0f,
        },
        {0,1,2},
        {
          {
            uint32_t(0),
            uint32_t(3)
          }
        },
        true
      };;

      vertex_buffer = new Framework::VertexBuffer({ &mesh }, { 2,2,4 });
      vertex_buffer->allocate(queue->getHandle(), cmdBuff->getHandle());

      pushConstant = new PushConstant();
      pushConstant->addVariable("uScale", vec2f({0.0f,0.0f}));
      pushConstant->addVariable("uTranslate", vec2f({0.0f,0.0f}));

      std::vector<unsigned char>	vertSpirv(sizeof(__glsl_shader_vert_spv)/sizeof(unsigned char));
      memcpy(&vertSpirv[0], __glsl_shader_vert_spv, sizeof(__glsl_shader_vert_spv));
      std::vector<unsigned char>	fragSpirv(sizeof(__glsl_shader_frag_spv) / sizeof(unsigned char));
      memcpy(&fragSpirv[0], __glsl_shader_frag_spv, sizeof(__glsl_shader_frag_spv));

      pipeline = new GraphicPipeline({ 0,0,0 }, { float(size.width),float(size.height),1.0f }, { 0,0 }, { float(size.width),float(size.height) });
      VertexShaderModule* sphereVertex = new Framework::VertexShaderModule(vertSpirv);
      pipeline->setVextexShader(sphereVertex);

      FragmentShaderModule* sphereFrag = new Framework::FragmentShaderModule(fragSpirv);
      pipeline->setFragmentModule(sphereFrag);
      pipeline->setVeritices(vertex_buffer);

      pipeline->addPushContant(pushConstant, VK_SHADER_STAGE_VERTEX_BIT);
      pipeline->addTextureBuffer(fontBuffer, VK_SHADER_STAGE_FRAGMENT_BIT,0);
      pipeline->SetCullMode(VK_CULL_MODE_NONE);
      pass->addSubPass({ pipeline }, Framework::RenderPassFlag::SHOW_ON_SCREEN | Framework::RenderPassFlag::USE_COLOR | Framework::RenderPassFlag::OP_STORE_COLOR | Framework::RenderPassFlag::USE_DEPTH | Framework::RenderPassFlag::OP_STORE_DEPTH);

      pass->compile();

      frame = new FrameBuffer(size.width, size.height);
      pass->prepareOutputFrameBuffer(*frame);
    }
    
    void ImGuiWrapper::prepareGui(Queue* queue, CommandBuffer* cmdBuff, SwapChainImage* img) {
      LavaCake::Framework::SwapChain* s = LavaCake::Framework::SwapChain::getSwapChain();
      VkExtent2D size = s->size();
      ImGui::Render();
      ImDrawData* draw_data = ImGui::GetDrawData();

      size_t vertex_size = draw_data->TotalVtxCount * sizeof(ImDrawVert)/sizeof(float);
      size_t index_size = draw_data->TotalIdxCount ;

      std::vector<float> vertices;
      std::vector<uint16_t>indices;

      uint16_t offset = 0;
      for (int n = 0; n < draw_data->CmdListsCount; n++)
      {
        const ImDrawList* cmd_list = draw_data->CmdLists[n];
        for (int j = 0; j < cmd_list->VtxBuffer.Size; j++) {
          vertices.push_back(cmd_list->VtxBuffer.Data[j].pos[0]);
          vertices.push_back(cmd_list->VtxBuffer.Data[j].pos[1]);
          vertices.push_back(cmd_list->VtxBuffer.Data[j].uv[0]);
          vertices.push_back(cmd_list->VtxBuffer.Data[j].uv[1]);
          uint8_t a = static_cast<uint8_t>((cmd_list->VtxBuffer.Data[j].col & 0xFF000000) >> 24);
          uint8_t r = static_cast<uint8_t>((cmd_list->VtxBuffer.Data[j].col & 0x00FF0000) >> 16);
          uint8_t g = static_cast<uint8_t>((cmd_list->VtxBuffer.Data[j].col & 0x0000FF00) >> 8);
          uint8_t b = static_cast<uint8_t>(cmd_list->VtxBuffer.Data[j].col & 0x000000FF);
          vertices.push_back(float(b) / 255.0f);
          vertices.push_back(float(g) / 255.0f);
          vertices.push_back(float(r) / 255.0f);
          vertices.push_back(float(a) / 255.0f);
        }
        for (int j = 0; j < cmd_list->IdxBuffer.Size; j++) {
          indices.push_back(static_cast<uint16_t>(cmd_list->IdxBuffer.Data[j] + offset));
        }
        offset = vertices.size()/8;
      }

      mesh = {
        vertices,
        indices,
        {{uint32_t(0),uint32_t(draw_data->TotalIdxCount)}},
        true
      };

      vertex_buffer->swapMeshes({ &mesh });
      vertex_buffer->allocate(queue->getHandle(), cmdBuff->getHandle());

      vec2f scale = vec2f({ 2.0f / draw_data->DisplaySize.x , 2.0f / draw_data->DisplaySize.y });
      vec2f translate = vec2f({ -1.0f - draw_data->DisplayPos.x * scale[0] , -1.0f - draw_data->DisplayPos.y * scale[1] });
      pushConstant->setVariable("uScale", scale);
      pushConstant->setVariable("uTranslate", translate);



    }


    void ImGuiWrapper::drawGui(Queue* queue, CommandBuffer* cmdBuff, SwapChainImage* img) {
      LavaCake::Framework::SwapChain* s = LavaCake::Framework::SwapChain::getSwapChain();
      VkExtent2D size = s->size();

      pass->setSwapChainImage(*frame, *img);
      
      pass->draw(cmdBuff->getHandle(), frame->getFrameBuffer(), { 0,0 }, { size.width, size.height }, { {0.0f,0.0f,0.0f,0.0f}, { 1.0f, 0 }});
    }



    void prepareImput(GLFWwindow* window) {

      // Setup back-end capabilities flags
      ImGuiIO& io = ImGui::GetIO();
      io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;         // We can honor GetMouseCursor() values (optional)
      io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;          // We can honor io.WantSetMousePos requests (optional, rarely used)
      io.BackendPlatformName = "imgui_impl_glfw";

      // Keyboard mapping. ImGui will use those indices to peek into the io.KeysDown[] array.
      io.KeyMap[ImGuiKey_Tab] = GLFW_KEY_TAB;
      io.KeyMap[ImGuiKey_LeftArrow] = GLFW_KEY_LEFT;
      io.KeyMap[ImGuiKey_RightArrow] = GLFW_KEY_RIGHT;
      io.KeyMap[ImGuiKey_UpArrow] = GLFW_KEY_UP;
      io.KeyMap[ImGuiKey_DownArrow] = GLFW_KEY_DOWN;
      io.KeyMap[ImGuiKey_PageUp] = GLFW_KEY_PAGE_UP;
      io.KeyMap[ImGuiKey_PageDown] = GLFW_KEY_PAGE_DOWN;
      io.KeyMap[ImGuiKey_Home] = GLFW_KEY_HOME;
      io.KeyMap[ImGuiKey_End] = GLFW_KEY_END;
      io.KeyMap[ImGuiKey_Insert] = GLFW_KEY_INSERT;
      io.KeyMap[ImGuiKey_Delete] = GLFW_KEY_DELETE;
      io.KeyMap[ImGuiKey_Backspace] = GLFW_KEY_BACKSPACE;
      io.KeyMap[ImGuiKey_Space] = GLFW_KEY_SPACE;
      io.KeyMap[ImGuiKey_Enter] = GLFW_KEY_ENTER;
      io.KeyMap[ImGuiKey_Escape] = GLFW_KEY_ESCAPE;
      io.KeyMap[ImGuiKey_KeyPadEnter] = GLFW_KEY_KP_ENTER;
      io.KeyMap[ImGuiKey_A] = GLFW_KEY_A;
      io.KeyMap[ImGuiKey_C] = GLFW_KEY_C;
      io.KeyMap[ImGuiKey_V] = GLFW_KEY_V;
      io.KeyMap[ImGuiKey_X] = GLFW_KEY_X;
      io.KeyMap[ImGuiKey_Y] = GLFW_KEY_Y;
      io.KeyMap[ImGuiKey_Z] = GLFW_KEY_Z;

      //io.SetClipboardTextFn = ImGui_ImplGlfw_SetClipboardText;
      //io.GetClipboardTextFn = ImGui_ImplGlfw_GetClipboardText;
      //io.ClipboardUserData = g_Window;
#if defined(_WIN32)
      io.ImeWindowHandle = (void*)glfwGetWin32Window(window);
#endif

      // Create mouse cursors
      // (By design, on X11 cursors are user configurable and some cursors may be missing. When a cursor doesn't exist,
      // GLFW will emit an error which will often be printed by the app, so we temporarily disable error reporting.
      // Missing cursors will return NULL and our _UpdateMouseCursor() function will use the Arrow cursor instead.)
      GLFWerrorfun prev_error_callback = glfwSetErrorCallback(NULL);
/*      g_MouseCursors[ImGuiMouseCursor_Arrow] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
      g_MouseCursors[ImGuiMouseCursor_TextInput] = glfwCreateStandardCursor(GLFW_IBEAM_CURSOR);
      g_MouseCursors[ImGuiMouseCursor_ResizeNS] = glfwCreateStandardCursor(GLFW_VRESIZE_CURSOR);
      g_MouseCursors[ImGuiMouseCursor_ResizeEW] = glfwCreateStandardCursor(GLFW_HRESIZE_CURSOR);
      g_MouseCursors[ImGuiMouseCursor_Hand] = glfwCreateStandardCursor(GLFW_HAND_CURSOR);
#if GLFW_HAS_NEW_CURSORS
      g_MouseCursors[ImGuiMouseCursor_ResizeAll] = glfwCreateStandardCursor(GLFW_RESIZE_ALL_CURSOR);
      g_MouseCursors[ImGuiMouseCursor_ResizeNESW] = glfwCreateStandardCursor(GLFW_RESIZE_NESW_CURSOR);
      g_MouseCursors[ImGuiMouseCursor_ResizeNWSE] = glfwCreateStandardCursor(GLFW_RESIZE_NWSE_CURSOR);
      g_MouseCursors[ImGuiMouseCursor_NotAllowed] = glfwCreateStandardCursor(GLFW_NOT_ALLOWED_CURSOR);
#else
      g_MouseCursors[ImGuiMouseCursor_ResizeAll] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
      g_MouseCursors[ImGuiMouseCursor_ResizeNESW] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
      g_MouseCursors[ImGuiMouseCursor_ResizeNWSE] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
      g_MouseCursors[ImGuiMouseCursor_NotAllowed] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
#endif
      glfwSetErrorCallback(prev_error_callback);
      */
      // Chain GLFW callbacks: our callbacks will call the user's previously installed callbacks, if any.
      //g_PrevUserCallbackMousebutton = NULL;
      //g_PrevUserCallbackScroll = NULL;
      //g_PrevUserCallbackKey = NULL;
      //g_PrevUserCallbackChar = NULL;
      
      //g_InstalledCallbacks = true;
      glfwSetMouseButtonCallback(window, ImGui_ImplGlfw_MouseButtonCallback);
      glfwSetScrollCallback(window, ImGui_ImplGlfw_ScrollCallback);
      glfwSetKeyCallback(window, ImGui_ImplGlfw_KeyCallback);
      glfwSetCharCallback(window, ImGui_ImplGlfw_CharCallback);
      glfwSetCursorPosCallback(window, ImGui_ImplGlfw_Mouse_Position);
         
    }
  }
}