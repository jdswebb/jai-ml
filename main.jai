#import "Basic";
#import "d3d11"()(INCLUDE_DEBUG_BINDINGS = false);
#import "Math";
#import "Input";
#import "Random";
#import "Windows";
#import "Window_Creation";
#import "Windows_Utf8";
#import "dxgi";
#load "camera.jai";
#load "ml_render.jai";

DEBUG :: 0;
driver_type := D3D_DRIVER_TYPE.NULL;
feature_level := D3D_FEATURE_LEVEL._11_0;

g_window: Window_Type;
g_device             : *ID3D11Device;
g_ctx                : *ID3D11DeviceContext;
g_swap_chain         : *IDXGISwapChain;
g_rtv                : *ID3D11RenderTargetView;
g_ps                 : *ID3D11PixelShader;
g_vs                 : *ID3D11VertexShader;
g_il                 : *ID3D11InputLayout;
g_ps_instanced       : *ID3D11PixelShader;
g_vs_instanced       : *ID3D11VertexShader;
g_il_instanced       : *ID3D11InputLayout;
g_vb_cube      : *ID3D11Buffer;
g_vb_neuron_inst     : *ID3D11Buffer;
g_vb_wire_inst       : *ID3D11Buffer;
g_cbuffer            : *ID3D11Buffer;
g_rs                 : *ID3D11RasterizerState;
g_bs                 : *ID3D11BlendState;
g_camera             : Camera;

g_shape_cube_positions :: Vector3.[
    .{ -0.5,  0.5,  0.5 },
    .{  0.5,  0.5,  0.5 },
    .{ -0.5, -0.5,  0.5 },
    .{  0.5, -0.5,  0.5 },
    .{ -0.5,  0.5, -0.5 },
    .{  0.5,  0.5, -0.5 },
    .{ -0.5, -0.5, -0.5 },
    .{  0.5, -0.5, -0.5 }
];

g_shape_cube :: Vector3.[
    g_shape_cube_positions[0],g_shape_cube_positions[1],g_shape_cube_positions[2],g_shape_cube_positions[1],g_shape_cube_positions[3],g_shape_cube_positions[2],
    g_shape_cube_positions[4],g_shape_cube_positions[6],g_shape_cube_positions[5],g_shape_cube_positions[5],g_shape_cube_positions[6],g_shape_cube_positions[7],
    g_shape_cube_positions[0],g_shape_cube_positions[2],g_shape_cube_positions[4],g_shape_cube_positions[4],g_shape_cube_positions[2],g_shape_cube_positions[6],
    g_shape_cube_positions[1],g_shape_cube_positions[5],g_shape_cube_positions[3],g_shape_cube_positions[5],g_shape_cube_positions[7],g_shape_cube_positions[3],
    g_shape_cube_positions[0],g_shape_cube_positions[4],g_shape_cube_positions[1],g_shape_cube_positions[4],g_shape_cube_positions[5],g_shape_cube_positions[1],
    g_shape_cube_positions[2],g_shape_cube_positions[3],g_shape_cube_positions[6],g_shape_cube_positions[6],g_shape_cube_positions[3],g_shape_cube_positions[7]    
];

CenterWindow :: (hwnd: Window_Type)
{
    rcWindow: RECT;
    GetWindowRect(hwnd, *rcWindow);
    windowWidth := rcWindow.right - rcWindow.left;
    windowHeight := rcWindow.bottom - rcWindow.top;
    screenWidth := GetSystemMetrics(SM_CXSCREEN);
    screenHeight := GetSystemMetrics(SM_CYSCREEN);
    posX := (screenWidth  - windowWidth)  / 2;
    posY := (screenHeight - windowHeight) / 2;
    SetWindowPos(hwnd, null, posX, posY, 0, 0, SWP_NOZORDER | SWP_NOSIZE);
}

#scope_file

init_window :: () -> bool
{
    g_window = create_window(1650, 1024, "ML");
    CenterWindow(g_window);
    return true;
}

init_device :: () -> bool {
    hr : HRESULT = S_OK;

    flags : D3D11_CREATE_DEVICE_FLAG;
    #if DEBUG flags |= .DEBUG;

    driver_types   := D3D_DRIVER_TYPE.[.HARDWARE, .WARP, .REFERENCE];
    feature_levels := D3D_FEATURE_LEVEL.[._11_1, ._11_0, ._10_1, ._10_0];

    for 0..driver_types.count-1 {
        driver_type = driver_types[it];
        hr = D3D11CreateDevice(null, driver_type, null, flags, feature_levels.data, feature_levels.count, D3D11_SDK_VERSION, *g_device, *feature_level, *g_ctx);

        if hr == E_INVALIDARG {
            hr = D3D11CreateDevice(null, driver_type, null, flags, feature_levels.data + 1, feature_levels.count - 1, D3D11_SDK_VERSION, *g_device, *feature_level, *g_ctx);
        }

        if SUCCEEDED(hr) break;
    }
    if FAILED(hr) {
        print("D3D11CreateDevice failed.\n");
        return false;
    }

    assert(g_device != null);
    print("Created D3D11 device % with feature level %\n", g_device, feature_level);

    dxgi_factory : *IDXGIFactory1;
    {
        dxgi_device : *IDXGIDevice;
        hr = IUnknown_QueryInterface(g_device, *uid(IDXGIDevice_UUID), xx *dxgi_device);
        if SUCCEEDED(hr) {
            adapter : *IDXGIAdapter;
            hr = IDXGIDevice_GetAdapter(dxgi_device, *adapter);
            if SUCCEEDED(hr) {
                hr = IDXGIObject_GetParent(adapter, *uid(IDXGIFactory1_UUID), xx *dxgi_factory);
                IUnknown_Release(adapter);
            }
            IUnknown_Release(dxgi_device);
        }
        else {
            print("QueryInterface FAILED\n");
        }
    }
    if FAILED(hr) {
        print("GetAdapter failed.\n");
        return false;
    }

    print("Create swap chain\n");

    rc : RECT;
    GetClientRect(g_window, *rc);
    width := rc.right - rc.left;
    height := rc.bottom - rc.top;

    sd : DXGI_SWAP_CHAIN_DESC;
    sd.BufferCount = 1;
    sd.BufferDesc.Width = xx width;
    sd.BufferDesc.Height = xx height;
    sd.BufferDesc.Format = .R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = .RENDER_TARGET_OUTPUT;
    sd.OutputWindow = g_window;
    sd.SampleDesc.Count = 8;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = BOOL.TRUE;

    hr = IDXGIFactory_CreateSwapChain(dxgi_factory, g_device, *sd, *g_swap_chain);
    if FAILED(hr) {
        log_error("CreateSwapChain failed: %", hr);
    }

    IDXGIFactory_MakeWindowAssociation(dxgi_factory, g_window, cast (u32) DXGI_MWA.NO_ALT_ENTER);
    IUnknown_Release(dxgi_factory);

    if FAILED(hr) return false;
    print("Create a render target view\n");

    back_buffer : *ID3D11Texture2D;
    hr = IDXGISwapChain_GetBuffer(g_swap_chain, 0, *uid(ID3D11Texture2D_UUID), xx *back_buffer);
    if FAILED(hr) {
        log_error("GetBuffer failed: %", hr);
        return false;
    }

    hr = ID3D11Device_CreateRenderTargetView(g_device, back_buffer, null, *g_rtv);
    IUnknown_Release(back_buffer);
    if FAILED(hr) {
        log_error("CreateRenderTargetView failed: %", hr);
        return false;
    }

    ID3D11DeviceContext_OMSetRenderTargets(g_ctx, 1, *g_rtv, null);
    print("Setup the viewport\n");

    vp : D3D11_VIEWPORT;
    vp.Width = xx width;
    vp.Height = xx height;
    vp.MinDepth = 0.0;
    vp.MaxDepth = 1.0;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    ID3D11DeviceContext_RSSetViewports(g_ctx, 1, *vp);
    print("Compile the vertex shader\n");

    bs_desc : D3D11_BLEND_DESC;
    bs_desc.RenderTarget[0].BlendEnable = .TRUE;
    bs_desc.RenderTarget[0].SrcBlend = .SRC_ALPHA;
    bs_desc.RenderTarget[0].DestBlend = .INV_SRC_ALPHA;
    bs_desc.RenderTarget[0].SrcBlendAlpha = .ONE;
    bs_desc.RenderTarget[0].DestBlendAlpha = .ONE;
    bs_desc.RenderTarget[0].BlendOp = .ADD;
    bs_desc.RenderTarget[0].BlendOpAlpha = .ADD;
    bs_desc.RenderTarget[0].RenderTargetWriteMask = xx D3D11_COLOR_WRITE_ENABLE.ALL;
    hr = ID3D11Device_CreateBlendState(g_device, *bs_desc, *g_bs);
    if FAILED(hr) {
        log_error("ID3D11Device_CreateBlendState failed: %", hr);
        free(g_rs);
        return false;
    }

    rs_desc : D3D11_RASTERIZER_DESC;
    rs_desc.CullMode = .NONE;
    rs_desc.FillMode = .SOLID;
    hr = ID3D11Device_CreateRasterizerState(g_device, *rs_desc, *g_rs);
    if FAILED(hr) {
        log_error("CreateRasterizerState failed: %", hr);
        free(g_rs);
        return false;
    }

    {
        layout : [3]D3D11_INPUT_ELEMENT_DESC;
        layout[0].SemanticName = "POSITION";
        layout[0].SemanticIndex = 0;
        layout[0].Format = .R32G32B32_FLOAT;
        layout[0].InputSlot = 0;
        layout[0].AlignedByteOffset = 0;
        layout[0].InputSlotClass = .VERTEX_DATA;
        layout[0].InstanceDataStepRate = 0;
        layout[1].SemanticName = "POSITION";
        layout[1].SemanticIndex = 1;
        layout[1].Format = .R32G32B32_FLOAT;
        layout[1].InputSlot = 1;
        layout[1].AlignedByteOffset = 0;
        layout[1].InputSlotClass = .INSTANCE_DATA;
        layout[1].InstanceDataStepRate = 1;
        layout[2].SemanticName = "COLOR";
        layout[2].SemanticIndex = 1;
        layout[2].Format = .R32G32B32_FLOAT;
        layout[2].InputSlot = 1;
        layout[2].AlignedByteOffset = 12;
        layout[2].InputSlotClass = .INSTANCE_DATA;
        layout[2].InstanceDataStepRate = 1;
        
        compile_shader(neuron_shader_source, "vs_instanced", "ps_instanced",
            *g_vs_instanced, *g_ps_instanced, layout, *g_il_instanced);
    }
    {
        layout : [2]D3D11_INPUT_ELEMENT_DESC;
        layout[0].SemanticName = "POSITION";
        layout[0].SemanticIndex = 0;
        layout[0].Format = .R32G32B32_FLOAT;
        layout[0].InputSlot = 0;
        layout[0].AlignedByteOffset = 0;
        layout[0].InputSlotClass = .VERTEX_DATA;
        layout[0].InstanceDataStepRate = 0;
        layout[1].SemanticName = "COLOR";
        layout[1].SemanticIndex = 0;
        layout[1].Format = .R32G32B32_FLOAT;
        layout[1].InputSlot = 0;
        layout[1].AlignedByteOffset = 12;
        layout[1].InputSlotClass = .VERTEX_DATA;
        layout[1].InstanceDataStepRate = 0;
        
        compile_shader(neuron_shader_source, "vs", "ps",
            *g_vs, *g_ps, layout, *g_il);
    }

    bd : D3D11_BUFFER_DESC;
    bd.Usage = .DEFAULT;
    bd.ByteWidth = size_of(Vector3) * 36;
    bd.BindFlags = .VERTEX_BUFFER;
    bd.CPUAccessFlags = 0;

    InitData : D3D11_SUBRESOURCE_DATA;
    InitData.pSysMem = g_shape_cube.data;

    hr = ID3D11Device_CreateBuffer(g_device, *bd, *InitData, *g_vb_cube);
    if FAILED(hr) {
        log_error("CreateBuffer failed: %", hr);
        return false;
    }

    bd.ByteWidth = size_of(Vector4) * 4;
    hr = ID3D11Device_CreateBuffer(g_device, *bd, null, *g_cbuffer);
    bd.BindFlags = .CONSTANT_BUFFER;
    if FAILED(hr) {
        log_error("CreateBuffer CBuffer failed: %", hr);
        return false;
    }
    
    bd.ByteWidth = size_of(InstanceVertex) * 30000;
    bd.Usage = .DYNAMIC;
    bd.CPUAccessFlags = .WRITE;
    bd.BindFlags = .VERTEX_BUFFER;
    hr = ID3D11Device_CreateBuffer(g_device, *bd, null, *g_vb_neuron_inst);
    hr = ID3D11Device_CreateBuffer(g_device, *bd, null, *g_vb_wire_inst);
    if FAILED(hr) {
        log_error("CreateBuffer IB failed: %", hr);
        return false;
    }
    return true;
}

cleanup_device :: () {
    if g_ctx ID3D11DeviceContext_ClearState(g_ctx);

    if g_vb_cube IUnknown_Release(g_vb_cube);
    if g_il_instanced IUnknown_Release(g_il_instanced);
    if g_vs_instanced IUnknown_Release(g_vs_instanced);
    if g_ps_instanced IUnknown_Release(g_ps_instanced);
    if g_rtv IUnknown_Release(g_rtv);
    if g_swap_chain IUnknown_Release(g_swap_chain);
    if g_ctx IUnknown_Release(g_ctx);
    if g_device IUnknown_Release(g_device);
    if g_rs IUnknown_Release(g_rs);
    if g_bs IUnknown_Release(g_bs);
}

#scope_export
main :: () {
    if !init_window() {
        print ("init_window failed!\n");
        return;
    }

    if !init_device() {
        print ("init_device failed!\n");
        cleanup_device();
        return;
    }

    g_camera.position = .{-40, 14 * 1.5, 30};
    g_camera.euler_angles.x = PI/2;

    init_ml();

    msg : MSG;
    while msg.message != WM_QUIT {
        update_window_events();

        if PeekMessageW(*msg, null, 0, 0, PM_REMOVE) {
            TranslateMessage( *msg );
            DispatchMessageW( *msg );
        }
        else {
            update();
            render();
        }
    }

    cleanup_device();
}

#scope_file
update :: () {    
    update_camera(*g_camera, 1.0 / 60.0);
}

update_neuron_buffers :: () {
    mapped: D3D11_MAPPED_SUBRESOURCE;
    hr := ID3D11DeviceContext_Map(g_ctx, g_vb_neuron_inst, 0, .WRITE_DISCARD, 0, *mapped);
    if FAILED(hr) {
        log_error("Map failed: %", hr);
    }

    ib_data : *InstanceVertex = mapped.pData;
    for _, i : neuron_vertex_instances {
        ib_data[i].offset = neuron_vertex_instances[i].offset;
    }
    ID3D11DeviceContext_Unmap(g_ctx, g_vb_neuron_inst, 0);    
}

update_wire_buffers :: () {
    mapped: D3D11_MAPPED_SUBRESOURCE;
    hr := ID3D11DeviceContext_Map(g_ctx, g_vb_wire_inst, 0, .WRITE_DISCARD, 0, *mapped);
    if FAILED(hr) {
        log_error("Map failed: %", hr);
    }

    ib_data : *InstanceVertex = mapped.pData;
    for _, i : wire_vertex_data {
        ib_data[i].offset = wire_vertex_data[i].offset;
    }
    ID3D11DeviceContext_Unmap(g_ctx, g_vb_wire_inst, 0);
}

render :: () {    
    color: [4] float = .[0,0,0.3,1];
    ID3D11DeviceContext_ClearRenderTargetView(g_ctx, g_rtv, *color);

    update_neuron_buffers();
    update_wire_buffers();
    ID3D11DeviceContext_UpdateSubresource(g_ctx, g_cbuffer, 0, null, *g_camera.view_projection_matrix, 0, 0);

    ID3D11DeviceContext_RSSetState(g_ctx, g_rs);
    ID3D11DeviceContext_OMSetBlendState(g_ctx, g_bs, null, 0xffffffff);
    cbuffers: [1] *ID3D11Buffer;
    cbuffers[0] = g_cbuffer;
    ID3D11DeviceContext_VSSetConstantBuffers(g_ctx, 0, 1, *cbuffers[0]);

    // Draw neurons
    ID3D11DeviceContext_VSSetShader(g_ctx, g_vs_instanced, null, 0);
    ID3D11DeviceContext_PSSetShader(g_ctx, g_ps_instanced, null, 0);
    stride: u32 = size_of(Vector3);
    offset: u32 = 0;
    ID3D11DeviceContext_IASetInputLayout(g_ctx, g_il_instanced);
    ID3D11DeviceContext_IASetVertexBuffers(g_ctx, 0, 1, *g_vb_cube, *stride, *offset);
    stride = size_of(InstanceVertex);
    ID3D11DeviceContext_IASetVertexBuffers(g_ctx, 1, 1, *g_vb_neuron_inst, *stride, *offset);
    ID3D11DeviceContext_IASetPrimitiveTopology(g_ctx, D3D11_PRIMITIVE_TOPOLOGY.TRIANGLELIST);
    ID3D11DeviceContext_DrawInstanced(g_ctx, 36, xx neuron_vertex_instances.count, 0, 0);

    // Draw wires
    ID3D11DeviceContext_VSSetShader(g_ctx, g_vs, null, 0);
    ID3D11DeviceContext_PSSetShader(g_ctx, g_ps, null, 0);
    ID3D11DeviceContext_IASetInputLayout(g_ctx, g_il);
    ID3D11DeviceContext_IASetVertexBuffers(g_ctx, 0, 1, *g_vb_wire_inst, *stride, *offset);
    ID3D11DeviceContext_IASetPrimitiveTopology(g_ctx, D3D11_PRIMITIVE_TOPOLOGY.LINELIST);
    ID3D11DeviceContext_Draw(g_ctx, xx wire_vertex_data.count, 0);

    hr := IDXGISwapChain_Present(g_swap_chain, 1, 0);
    if FAILED(hr) {
        log_error("Present failed: %", hr);
    }
}

