// Derived from imgui_sdl_gl.jai in the Jai ImGui module 

#import "Basic";
#import "SDL";
#import "Math";

cursor_arrow: *SDL_Cursor;
cursor_text_input: *SDL_Cursor;
cursor_move: *SDL_Cursor;
cursor_resize_ns: *SDL_Cursor;
cursor_resize_ew: *SDL_Cursor;
cursor_resize_nesw: *SDL_Cursor;
cursor_resize_nwse: *SDL_Cursor;

ImguiCBuffer :: struct {
    mvp: [4][4]float;
}

ImGuiState :: struct
{
    device            : *ID3D11Device;
    ctx               : *ID3D11DeviceContext;
    vb                : *ID3D11Buffer;
    ib                : *ID3D11Buffer;
    vb_size           : s32 = 5000;
    ib_size           : s32 = 10000;
    imgui_vs          : *ID3D11VertexShader;
    imgui_ps          : *ID3D11PixelShader;
    depth_state       : *ID3D11DepthStencilState;
    rasterizer_state  : *ID3D11RasterizerState;
    blend_state       : *ID3D11BlendState;
    input_layout      : *ID3D11InputLayout;
    cbuffer           : *ID3D11Buffer;
    font_texture      : *ID3D11Texture2D;
    font_sampler      : *ID3D11SamplerState;
    font_srv          : *ID3D11ShaderResourceView;

    time: u64 = 0;
    mouse_pressed := bool.[false, false, false];
    mouse_wheel: float = 0.0;
}

imgui_vs_source :: #string END
cbuffer vertexBuffer : register(b0)
{
  float4x4 ProjectionMatrix;
};
struct VS_INPUT
{
  float2 pos : POSITION;
  float4 col : COLOR0;
  float2 uv  : TEXCOORD0;
};

struct PS_INPUT
{
  float4 pos : SV_POSITION;
  float4 col : COLOR0;
  float2 uv  : TEXCOORD0;
};

PS_INPUT main(VS_INPUT input)
{
  PS_INPUT output;
  output.pos = mul( ProjectionMatrix, float4(input.pos.xy, 0.f, 1.f));
  output.col = input.col;
  output.uv  = input.uv;
  return output;
}
END;

imgui_ps_source :: #string END
struct PS_INPUT
{
    float4 pos : SV_POSITION;
    float4 col : COLOR0;
    float2 uv  : TEXCOORD0;
};
sampler sampler0;
Texture2D texture0;

float4 main(PS_INPUT input) : SV_Target
{
    float4 out_col = input.col * texture0.Sample(sampler0, input.uv);
    return out_col;
}
END;

ImGui_ImplDX11_SetupRenderState :: (state: *ImGuiState, draw_data: *ImGui.ImDrawData) {
    // Setup viewport
    vp: D3D11_VIEWPORT;
    vp.Width = draw_data.DisplaySize.x;
    vp.Height = draw_data.DisplaySize.y;
    vp.MinDepth = 0.0;
    vp.MaxDepth = 1.0;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    ID3D11DeviceContext_RSSetViewports(state.ctx, 1, *vp);

    // Setup shader and vertex buffers
    stride:u32 = size_of(ImGui.ImDrawVert);
    offset:u32;
    ID3D11DeviceContext_IASetInputLayout(state.ctx, state.input_layout);
    vbs: [1] *ID3D11Buffer;
    vbs[0] = state.vb;
    ID3D11DeviceContext_IASetVertexBuffers(state.ctx, 0, 1, vbs.data, *stride, *offset);
    idx_buf_type : DXGI_FORMAT = ifx size_of(ImGui.ImDrawIdx) == 2 then .R16_UINT else .R32_UINT;
    ID3D11DeviceContext_IASetIndexBuffer(state.ctx, state.ib, idx_buf_type, 0);
    ID3D11DeviceContext_IASetPrimitiveTopology(state.ctx, .TRIANGLELIST);
    ID3D11DeviceContext_VSSetShader(state.ctx, state.imgui_vs, null, 0);
    ID3D11DeviceContext_VSSetConstantBuffers(state.ctx, 0, 1, *state.cbuffer);
    ID3D11DeviceContext_PSSetShader(state.ctx, state.imgui_ps, null, 0);
    ID3D11DeviceContext_PSSetSamplers(state.ctx, 0, 1, *state.font_sampler);

    // Setup blend state
    blend_factor : [4]float = .[0, 0, 0, 0];
    ID3D11DeviceContext_OMSetBlendState(state.ctx, state.blend_state, *blend_factor, 0xffffffff);
    ID3D11DeviceContext_OMSetDepthStencilState(state.ctx, state.depth_state, 0);
    ID3D11DeviceContext_RSSetState(state.ctx, state.rasterizer_state);
}

// This is the main rendering function that you have to implement and provide to ImGui (via setting up 'RenderDrawListsFn' in the ImGuiIO structure)
// Note that this implementation is little overcomplicated because we are saving/setting up/restoring every OpenGL state explicitly, in order to be able to run within any OpenGL engine that doesn't do so.
// If text or lines are blurry when integrating ImGui in your engine: in your Render function, try translating your projection matrix by (0.5f,0.5f) or (0.375f,0.375f)
ImGui_ImplSdl_RenderDrawLists :: (state: *ImGuiState, draw_data: *ImGui.ImDrawData) -> bool {
    if draw_data.DisplaySize.x <= 0 || draw_data.DisplaySize.y <= 0
        return false;

    if !state.vb || state.vb_size < draw_data.TotalVtxCount {
        if state.vb != null { free(state.vb); state.vb = null; }
        state.vb_size = draw_data.TotalVtxCount + 5000;
        desc := D3D11_BUFFER_DESC.{
            Usage          = .DYNAMIC,
            BindFlags      = .VERTEX_BUFFER,
            CPUAccessFlags = .WRITE,
        };
        desc.ByteWidth = cast(u32)(state.vb_size * size_of(ImGui.ImDrawVert));
        if ID3D11Device_CreateBuffer(state.device, *desc, null, *state.vb) < 0 {
            log("[imgui_d3d11] Failed to create vertex buffer");
            return false;
        }
    }

    if !state.ib || state.ib_size < draw_data.TotalIdxCount {
        if state.ib != null { free(state.ib); state.ib = null; }
        state.ib_size = draw_data.TotalIdxCount + 10000;
        desc := D3D11_BUFFER_DESC.{
            Usage = .DYNAMIC,
            BindFlags = .INDEX_BUFFER,
            CPUAccessFlags = .WRITE,
        };
        desc.ByteWidth = cast(u32)(state.ib_size * size_of(ImGui.ImDrawIdx));
        if ID3D11Device_CreateBuffer(state.device, *desc, null, *state.ib) < 0 {
            log_error("[imgui_d3d11] Failed to create index buffer");
            return false;
        }
    }

    // Upload vertex/index data into a single contiguous GPU buffer
    vtx_resource, idx_resource: D3D11_MAPPED_SUBRESOURCE;
    if FAILED(ID3D11DeviceContext_Map(state.ctx, state.vb, 0, .WRITE_DISCARD, 0, *vtx_resource)) {
        return false;
    }
    if FAILED(ID3D11DeviceContext_Map(state.ctx, state.ib, 0, .WRITE_DISCARD, 0, *idx_resource)) {
        return false;
    }

    vtx_dst := cast(*ImGui.ImDrawVert)vtx_resource.pData;
    idx_dst := cast(*ImGui.ImDrawIdx)idx_resource.pData;
    for n: 0..draw_data.CmdListsCount - 1 {
        cmd_list := draw_data.CmdLists[n];
        memcpy(vtx_dst, cmd_list.VtxBuffer.Data, cmd_list.VtxBuffer.Size * size_of(ImGui.ImDrawVert));
        memcpy(idx_dst, cmd_list.IdxBuffer.Data, cmd_list.IdxBuffer.Size * size_of(ImGui.ImDrawIdx));
        vtx_dst += cmd_list.VtxBuffer.Size;
        idx_dst += cmd_list.IdxBuffer.Size;
    }
    ID3D11DeviceContext_Unmap(state.ctx, state.vb, 0);
    ID3D11DeviceContext_Unmap(state.ctx, state.ib, 0);
    {
        mapped_resource: D3D11_MAPPED_SUBRESOURCE;
        if FAILED(ID3D11DeviceContext_Map(state.ctx, state.cbuffer, 0, .WRITE_DISCARD, 0, *mapped_resource)) {
            return false;
        }

        constant_buffer := cast(*ImguiCBuffer)mapped_resource.pData;
        L:float = draw_data.DisplayPos.x;
        R:float = draw_data.DisplayPos.x + draw_data.DisplaySize.x;
        T:float = draw_data.DisplayPos.y;
        B:float = draw_data.DisplayPos.y + draw_data.DisplaySize.y;
        make_float4 :: (a: float, b: float, c: float, d: float) -> [4]float {
            f: [4]float = ---;
            f[0] = a;
            f[1] = b;
            f[2] = c;
            f[3] = d;
            return f;
        }
        mvp: [4][4]float;
        mvp[0] = make_float4(2.0/(R-L),   0.0,         0.0, 0.0);
        mvp[1] = make_float4(0.0,         2.0/(T-B),   0.0, 0.0);
        mvp[2] = make_float4(0.0,         0.0,         0.5, 0.0);
        mvp[3] = make_float4((R+L)/(L-R), (T+B)/(B-T), 0.5, 1.0);
        memcpy(*constant_buffer.mvp, mvp[0].data, size_of(type_of(mvp)));
        ID3D11DeviceContext_Unmap(state.ctx, state.cbuffer, 0);
    }
    
    ImGui_ImplDX11_SetupRenderState(state, draw_data);

    global_idx_offset:u32;
    global_vtx_offset:u32;
    clip_off := draw_data.DisplayPos;
    for n: 0..draw_data.CmdListsCount - 1 {
        cmd_list := draw_data.CmdLists[n];
        for cmd_i: 0..cmd_list.CmdBuffer.Size - 1 {
            pcmd := *cmd_list.CmdBuffer.Data[cmd_i];
            if pcmd.UserCallback != null {
                // User callback, registered via ImDrawList::AddCallback()
                // (ImDrawCallback_ResetRenderState is a special callback value used by the user to request the renderer to reset render state.)
                ImDrawCallback_ResetRenderState :: -1;
                if cast(int)cast(*void)pcmd.UserCallback == ImDrawCallback_ResetRenderState
                    ImGui_ImplDX11_SetupRenderState(state, draw_data);
                else
                    pcmd.UserCallback(cmd_list, pcmd);
            } else {
                r: D3D11_RECT;
                r.left   = cast(s32)(pcmd.ClipRect.x - clip_off.x);
                r.top    = cast(s32)(pcmd.ClipRect.y - clip_off.y);
                r.right  = cast(s32)(pcmd.ClipRect.z - clip_off.x);
                r.bottom = cast(s32)(pcmd.ClipRect.w - clip_off.y);
                ID3D11DeviceContext_RSSetScissorRects(state.ctx, 1, *r);
                
                texture_srv := cast(*ID3D11ShaderResourceView)pcmd.TextureId;
                ID3D11DeviceContext_PSSetShaderResources(state.ctx, 0, 1, *texture_srv);
                ID3D11DeviceContext_DrawIndexed(state.ctx, pcmd.ElemCount, pcmd.IdxOffset + global_idx_offset, cast(s32)(pcmd.VtxOffset + global_vtx_offset));
            }
        }
        global_idx_offset += cast(u32)cmd_list.IdxBuffer.Size;
        global_vtx_offset += cast(u32)cmd_list.VtxBuffer.Size;
    }

    return true;
}

ImGui_ImplSdl_GetClipboardText :: (data: *void) -> *u8 #c_call {
    return SDL_GetClipboardText();
}

ImGui_ImplSdl_SetClipboardText :: (data: *void, text: *u8) #c_call {
    SDL_SetClipboardText(text);
}

// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
// Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
ImGui_ImplSdl_ProcessEvent :: (state: *ImGuiState, event: *SDL_Event) -> bool {
    io := ImGui.GetIO();
    if event.type == {
        case SDL_MOUSEWHEEL;
            if event.wheel.y > 0 state.mouse_wheel = 1;
            if event.wheel.y < 0 state.mouse_wheel = -1;
            return true;
        case SDL_MOUSEBUTTONDOWN;
            if event.button.button == SDL_BUTTON_LEFT   state.mouse_pressed[0] = true;
            if event.button.button == SDL_BUTTON_RIGHT  state.mouse_pressed[1] = true;
            if event.button.button == SDL_BUTTON_MIDDLE state.mouse_pressed[2] = true;
            return true;
        case SDL_TEXTINPUT;
            io.AddInputCharactersUTF8(io, xx event.text.text.data);
            return true;
        case SDL_KEYDOWN; #through;
        case SDL_KEYUP;
            //@@key := event.key.keysym.sym & ~SDLK_SCANCODE_MASK;
            key := event.key.keysym.sym & (xx ~(1<<30));
            io.KeysDown[key] = (event.type == SDL_KEYDOWN);
            kmod := SDL_GetModState();
            io.KeyShift = ((kmod & KMOD_SHIFT) != 0);
            io.KeyCtrl = ((kmod & KMOD_CTRL) != 0);
            io.KeyAlt = ((kmod & KMOD_ALT) != 0);
            io.KeySuper = ((kmod & KMOD_GUI) != 0);
            return true;
    }
    return false;
}

ImGui_ImplDX11_CreateFontsTexture :: (state: *ImGuiState) -> bool {
    io := ImGui.GetIO();
    pixels: *u8;
    width, height: s32;
    io.Fonts.GetTexDataAsRGBA32(io.Fonts, *pixels, *width, *height);
    {
        desc := D3D11_TEXTURE2D_DESC.{
            MipLevels = 1,
            ArraySize = 1,
            Format    = .DXGI_FORMAT_R8G8B8A8_UNORM,
            Usage     = .D3D11_USAGE_DEFAULT,
            BindFlags = .D3D11_BIND_SHADER_RESOURCE,
        };
        desc.Width  = cast(u32)width;
        desc.Height = cast(u32)height;
        desc.SampleDesc.Count = 1;

        subResource: D3D11_SUBRESOURCE_DATA;
        subResource.pSysMem = pixels;
        subResource.SysMemPitch = desc.Width * 4;
        subResource.SysMemSlicePitch = 0;
        ID3D11Device_CreateTexture2D(state.device, *desc, *subResource, *state.font_texture);

        srvDesc: D3D11_SHADER_RESOURCE_VIEW_DESC ;
        srvDesc.Format = .DXGI_FORMAT_R8G8B8A8_UNORM;
        srvDesc.ViewDimension = .TEXTURE2D;
        srvDesc.Texture2D.MipLevels = desc.MipLevels;
        srvDesc.Texture2D.MostDetailedMip = 0;
        ID3D11Device_CreateShaderResourceView(state.device, state.font_texture, *srvDesc, *state.font_srv);
    }
    io.Fonts.TexID = xx state.font_srv;
    {
        desc := D3D11_SAMPLER_DESC.{
            Filter = .D3D11_FILTER_MIN_MAG_MIP_LINEAR,
            AddressU = .D3D11_TEXTURE_ADDRESS_WRAP,
            AddressV = .D3D11_TEXTURE_ADDRESS_WRAP,
            AddressW = .D3D11_TEXTURE_ADDRESS_WRAP,
            ComparisonFunc = .D3D11_COMPARISON_ALWAYS,
        };
        ID3D11Device_CreateSamplerState(state.device, *desc, *state.font_sampler);
    }
    return true;
}

ImGui_ImplDX11_CreateDeviceObjects ::(state: *ImGuiState) -> bool
{
    ps : string;
    vs : string;
    hr : HRESULT;
    ps, hr = compile_shader_d3d(imgui_ps_source, "main", "ps_4_0", false);
    if FAILED(ID3D11Device_CreatePixelShader(state.device, ps.data, cast(u64) ps.count, null, *state.imgui_ps)) {
        return false;
    }
    vs, hr = compile_shader_d3d(imgui_vs_source, "main", "vs_4_0", false);
    if FAILED(ID3D11Device_CreateVertexShader(state.device, vs.data, cast(u64) vs.count, null, *state.imgui_vs)) {
        return false;
    }

    // Create the input layout
    local_layout := D3D11_INPUT_ELEMENT_DESC.[
        .{ "POSITION", 0, .R32G32_FLOAT,   0, 0, .VERTEX_DATA, 0 },
        .{ "TEXCOORD", 0, .R32G32_FLOAT,   0, 8,  .VERTEX_DATA, 0 },
        .{ "COLOR",    0, .R8G8B8A8_UNORM, 0, 16, .VERTEX_DATA, 0 },
    ];
    if FAILED(ID3D11Device_CreateInputLayout(state.device, local_layout.data, local_layout.count, vs.data, cast(u64) vs.count, *state.input_layout)) {
        return false;
    }

    // Create the constant buffer
    {
        desc := D3D11_BUFFER_DESC.{
            Usage          = .DYNAMIC,
            BindFlags      = .CONSTANT_BUFFER,
            CPUAccessFlags = .WRITE,
            ByteWidth      = size_of(ImguiCBuffer),
        };
        ID3D11Device_CreateBuffer(state.device, *desc, null, *state.cbuffer);
    }
    
    // Create the blending setup
    {
        desc: D3D11_BLEND_DESC;
        desc.AlphaToCoverageEnable                 = cast(BOOL)false;
        desc.RenderTarget[0].BlendEnable           = cast(BOOL)true;
        desc.RenderTarget[0].SrcBlend              = .SRC_ALPHA;
        desc.RenderTarget[0].DestBlend             = .INV_SRC_ALPHA;
        desc.RenderTarget[0].BlendOp               = .ADD;
        desc.RenderTarget[0].SrcBlendAlpha         = .INV_SRC_ALPHA;
        desc.RenderTarget[0].DestBlendAlpha        = .ZERO;
        desc.RenderTarget[0].BlendOpAlpha          = .ADD;
        desc.RenderTarget[0].RenderTargetWriteMask = xx D3D11_COLOR_WRITE_ENABLE.ALL;
        ID3D11Device_CreateBlendState(state.device, *desc, *state.blend_state);
    }

    // Create the rasterizer state
    {
        desc: D3D11_RASTERIZER_DESC;
        desc.FillMode = .SOLID;
        desc.CullMode = .NONE;
        desc.ScissorEnable = BOOL.TRUE;
        desc.DepthClipEnable = BOOL.TRUE;
        ID3D11Device_CreateRasterizerState(state.device, *desc, *state.rasterizer_state);
    }

    // Create depth-stencil State
    {
        desc: D3D11_DEPTH_STENCIL_DESC;
        desc.DepthEnable    = BOOL.FALSE;
        desc.DepthWriteMask = .ALL;
        desc.DepthFunc      = .ALWAYS;
        desc.StencilEnable  = BOOL.FALSE;
        desc.FrontFace.StencilFailOp      = .KEEP;
        desc.FrontFace.StencilDepthFailOp = .KEEP;
        desc.FrontFace.StencilPassOp      = .KEEP;
        desc.FrontFace.StencilFunc        = .ALWAYS;
        desc.BackFace = desc.FrontFace;
        ID3D11Device_CreateDepthStencilState(state.device, *desc, *state.depth_state);
    }

    ImGui_ImplDX11_CreateFontsTexture(state);    
    return true;
}

ImGui_ImplDX11_InvalidateDeviceObjects :: (state: *ImGuiState) {
    ImGui.GetIO().Fonts.TexID = null;
}

ImGui_ImplSdl_Init :: (state: *ImGuiState, window: *SDL_Window) -> bool {
    io := ImGui.GetIO();
    io.KeyMap[ImGui.Key.Tab] = xx SDLK_TAB;
    io.KeyMap[ImGui.Key.LeftArrow] = xx SDL_SCANCODE_LEFT;
    io.KeyMap[ImGui.Key.RightArrow] = xx SDL_SCANCODE_RIGHT;
    io.KeyMap[ImGui.Key.UpArrow] = xx SDL_SCANCODE_UP;
    io.KeyMap[ImGui.Key.DownArrow] = xx SDL_SCANCODE_DOWN;
    io.KeyMap[ImGui.Key.PageUp] = xx SDL_SCANCODE_PAGEUP;
    io.KeyMap[ImGui.Key.PageDown] = xx SDL_SCANCODE_PAGEDOWN;
    io.KeyMap[ImGui.Key.Home] = xx SDL_SCANCODE_HOME;
    io.KeyMap[ImGui.Key.End] = xx SDL_SCANCODE_END;
    io.KeyMap[ImGui.Key.Delete] = xx SDLK_DELETE;
    io.KeyMap[ImGui.Key.Backspace] = xx SDLK_BACKSPACE;
    io.KeyMap[ImGui.Key.Space] = xx SDLK_SPACE;
    io.KeyMap[ImGui.Key.Enter] = xx SDLK_RETURN;
    io.KeyMap[ImGui.Key.Escape] = xx SDLK_ESCAPE;
    io.KeyMap[ImGui.Key.A] = xx SDLK_a;
    io.KeyMap[ImGui.Key.W] = xx SDLK_w;
    io.KeyMap[ImGui.Key.S] = xx SDLK_s;
    io.KeyMap[ImGui.Key.D] = xx SDLK_d;
    io.KeyMap[ImGui.Key.Q] = xx SDLK_q;
    io.KeyMap[ImGui.Key.E] = xx SDLK_e;
    io.KeyMap[ImGui.Key.C] = xx SDLK_c;
    io.KeyMap[ImGui.Key.V] = xx SDLK_v;
    io.KeyMap[ImGui.Key.X] = xx SDLK_x;
    io.KeyMap[ImGui.Key.Y] = xx SDLK_y;
    io.KeyMap[ImGui.Key.Z] = xx SDLK_z;

    io.SetClipboardTextFn = ImGui_ImplSdl_SetClipboardText;
    io.GetClipboardTextFn = ImGui_ImplSdl_GetClipboardText;
    io.ClipboardUserData = null;

    #if OS == .WINDOWS {
        wmInfo: SDL_SysWMinfo ;
        SDL_VERSION(*wmInfo.version);
        SDL_GetWindowWMInfo(window, *wmInfo);
        io.ImeWindowHandle = wmInfo.info.win.window;
    }

    cursor_arrow = SDL_CreateSystemCursor(.ARROW);
    cursor_text_input = SDL_CreateSystemCursor(.IBEAM);
    cursor_move = SDL_CreateSystemCursor(.HAND);
    cursor_resize_ns = SDL_CreateSystemCursor(.SIZENS);
    cursor_resize_ew = SDL_CreateSystemCursor(.SIZEWE);
    cursor_resize_nesw = SDL_CreateSystemCursor(.SIZENESW);
    cursor_resize_nwse = SDL_CreateSystemCursor(.SIZENWSE);

    return true;
}

ImGui_ImplSdl_Shutdown :: (state: *ImGuiState) {
    ImGui_ImplDX11_InvalidateDeviceObjects(state);
}

ImGui_ImplSdl_NewFrame :: (state: *ImGuiState, window: *SDL_Window) {
    if !state.font_texture ImGui_ImplDX11_CreateDeviceObjects(state);

    io := ImGui.GetIO();

    // Setup display size (every frame to accommodate for window resizing)
    w, h: s32;
    display_w, display_h: s32;
    SDL_GetWindowSize(window, *w, *h);
    SDL_GL_GetDrawableSize(window, *display_w, *display_h);
    io.DisplaySize = .{xx w, xx h};
    io.DisplayFramebufferScale = .{cast(float) display_w / w, cast(float) display_h / h};

    // Setup time step
    frequency := SDL_GetPerformanceFrequency();
    current_time := SDL_GetPerformanceCounter();
    if state.time > 0 {
        io.DeltaTime = cast(float)((cast(float64)(current_time - state.time)) / frequency);
    } else {
        io.DeltaTime = (1.0/60.0);
    }
    state.time = current_time;

    // Setup inputs
    // (we already got mouse wheel, keyboard keys & characters from SDL_PollEvent())
    mx, my: s32;
    mouseMask := SDL_GetMouseState(*mx, *my);
    if (SDL_GetWindowFlags(window) & SDL_WINDOW_MOUSE_FOCUS) {
        io.MousePos = .{xx mx, xx my};   // Mouse position, in pixels (set to -1,-1 if no mouse / on another screen, etc.)
    } else {
        io.MousePos = .{-FLOAT32_MAX,-FLOAT32_MAX};
    }

    io.MouseDown[0] = (state.mouse_pressed[0] || (mouseMask & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0);      // If a mouse press event came, always pass it as "mouse held this frame", so we don't miss click-release events that are shorter than 1 frame.
    io.MouseDown[1] = (state.mouse_pressed[1] || (mouseMask & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0);
    io.MouseDown[2] = (state.mouse_pressed[2] || (mouseMask & SDL_BUTTON(SDL_BUTTON_MIDDLE)) != 0);
    state.mouse_pressed[0], state.mouse_pressed[1], state.mouse_pressed[2] = false, false, false;

    io.MouseWheel = state.mouse_wheel;
    state.mouse_wheel = 0.0;

    // Hide OS mouse cursor if ImGui is drawing it
    SDL_ShowCursor(xx (io.MouseDrawCursor == false));
}
