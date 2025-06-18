// SPDX-FileCopyrightText: James Webb
// SPDX-License-Identifier: MIT
// Full details are in the LICENSE file at the repository root.

#import "d3d_compiler";
#import "File";
#import "String";
#import "Sort";
#import "Basic";
#import "Simp";
#load "machine_learning.jai";
#load "shaders.jai";

InstanceVertex :: struct
{
    offset : Vector3;
    colour : Vector3;
}

SHAPE_CUBE_VERTICES :: Vector3.[
    .{ -0.5,  0.5,  0.5 },
    .{  0.5,  0.5,  0.5 },
    .{ -0.5, -0.5,  0.5 },
    .{  0.5, -0.5,  0.5 },
    .{ -0.5,  0.5, -0.5 },
    .{  0.5,  0.5, -0.5 },
    .{ -0.5, -0.5, -0.5 },
    .{  0.5, -0.5, -0.5 }
];

SHAPE_CUBE :: Vector3.[
    SHAPE_CUBE_VERTICES[0], SHAPE_CUBE_VERTICES[1], SHAPE_CUBE_VERTICES[2],
    SHAPE_CUBE_VERTICES[1], SHAPE_CUBE_VERTICES[3], SHAPE_CUBE_VERTICES[2],
    SHAPE_CUBE_VERTICES[4], SHAPE_CUBE_VERTICES[6], SHAPE_CUBE_VERTICES[5],
    SHAPE_CUBE_VERTICES[5], SHAPE_CUBE_VERTICES[6], SHAPE_CUBE_VERTICES[7],
    SHAPE_CUBE_VERTICES[0], SHAPE_CUBE_VERTICES[2], SHAPE_CUBE_VERTICES[4],
    SHAPE_CUBE_VERTICES[4], SHAPE_CUBE_VERTICES[2], SHAPE_CUBE_VERTICES[6],
    SHAPE_CUBE_VERTICES[1], SHAPE_CUBE_VERTICES[5], SHAPE_CUBE_VERTICES[3],
    SHAPE_CUBE_VERTICES[5], SHAPE_CUBE_VERTICES[7], SHAPE_CUBE_VERTICES[3],
    SHAPE_CUBE_VERTICES[0], SHAPE_CUBE_VERTICES[4], SHAPE_CUBE_VERTICES[1],
    SHAPE_CUBE_VERTICES[4], SHAPE_CUBE_VERTICES[5], SHAPE_CUBE_VERTICES[1],
    SHAPE_CUBE_VERTICES[2], SHAPE_CUBE_VERTICES[3], SHAPE_CUBE_VERTICES[6],
    SHAPE_CUBE_VERTICES[6], SHAPE_CUBE_VERTICES[3], SHAPE_CUBE_VERTICES[7]    
];

ShaderConstants :: struct
{
    mvp              : Matrix4;
    bias_time_unused : Vector4;
}

RenderState :: struct
{
    window_hwnd      : *void;
    window_width     : s32;
    window_height    : s32;

    device           : *ID3D11Device;
    ctx              : *ID3D11DeviceContext;
    swap_chain       : *IDXGISwapChain;
    sampler          : *ID3D11SamplerState;
    render_texture   : *ID3D11Texture2D;
    rtv              : *ID3D11RenderTargetView;
    srv              : *ID3D11ShaderResourceView;
    swap_rtv         : *ID3D11RenderTargetView;
    ps_fullscreen    : *ID3D11PixelShader;
    vs_fullscreen    : *ID3D11VertexShader;
    ps               : *ID3D11PixelShader;
    vs               : *ID3D11VertexShader;
    il               : *ID3D11InputLayout;
    ps_instanced     : *ID3D11PixelShader;
    vs_instanced     : *ID3D11VertexShader;
    il_instanced     : *ID3D11InputLayout;
    vb_cube          : *ID3D11Buffer;
    vb_neuron_inst   : *ID3D11Buffer;
    vb_wire_inst     : *ID3D11Buffer;
    cbuffer          : *ID3D11Buffer;
    rs               : *ID3D11RasterizerState;
    bs               : *ID3D11BlendState;
    
    gradient_textures         : [20]*ID3D11Texture2D;
    gradient_texture_srvs     : [20]*ID3D11ShaderResourceView;
    active_gradient           : s32 = 0;
    activation_draw_threshold : float = 0.02;

    buffers_dirty : bool = true;   
    wire_vertices : s64 = 0;

    frame_idx               : u64;
    output_neuron_positions : [..] Vector3;
    neuron_vertex_instances : [..] InstanceVertex;
    wire_vertex_data        : [..] InstanceVertex;
    layer_brightness_scale  : [8]float = float.[0.15, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0];

    camera           : Camera;
    res_scaling      : float = 2.0;
    bias             : float = 0.5;
    driver_type      := D3D_DRIVER_TYPE.NULL;
    feature_level    := D3D_FEATURE_LEVEL._11_0;
}

project_screen :: (pos: Vector3, vp: Matrix4, w: float, h: float) -> Vector3
{
    v4 : Vector4 = .{ pos.x, pos.y, pos.z, 1.0 };
    v4_clip : Vector4 = vp * v4;    
    if v4_clip.w == 0.0 {
        return .{ 0, 0, -1 };
    }
    ndc : Vector2 = .{ v4_clip.x / v4_clip.w, v4_clip.y / v4_clip.w };
    screen_x := (ndc.x + 1.0) * 0.5 * w;
    screen_y := (1.0 - ndc.y) * 0.5 * h;
    return .{ screen_x, screen_y, v4_clip.z };
}

layout_square_layer :: (neuron_vertex_instances: [] InstanceVertex, nn: *NeuralNet, max_width: float, 
    layer_idx: u32, offset: u32, z: float, spacing: float) -> u32
{
    dim: u32 = xx sqrt(xx nn.layers[layer_idx].size);
    is_emnist: bool = (nn.layers[nn.layers.count - 1].size) > 10;
    assert((dim * dim) == nn.layers[layer_idx].size);
    padding := 0.5 * (max_width - ((dim - 1) * spacing));
    if (is_emnist) {
        for i : 0..(dim-1) {
            for j : 0..(dim-1) {
                // rotate 90, flip
                neuron_vertex_instances[offset + (i * dim) + j].offset.x = xx padding + (i * spacing);
                neuron_vertex_instances[offset + (i * dim) + j].offset.y = xx padding + (((dim-1) - j) * spacing);
                neuron_vertex_instances[offset + (i * dim) + j].offset.z = z;
            }
        }
    }
    else {
        for i : 0..(dim-1) {
            for j : 0..(dim-1) {
                // vertical flip
                neuron_vertex_instances[offset + (i * dim) + j].offset.x = xx padding + (j * spacing);
                neuron_vertex_instances[offset + (i * dim) + j].offset.y = xx padding + (((dim-1) - i) * spacing);
                neuron_vertex_instances[offset + (i * dim) + j].offset.z = z;
            }
        }
    }

    return nn.layers[layer_idx].size;
}

layout_output_layer :: (neuron_vertex_instances: [] InstanceVertex, nn: *NeuralNet, max_width: float, layer_idx: u32, offset: u32,
    z: float, spacing: float, store: []Vector3) -> u32
{
    is_emnist: bool = (nn.layers[nn.layers.count - 1].size) > 10;
    padding_y := 0.5 * (max_width - spacing);
        if (is_emnist) {
            // create a line for lower case, upper case and numbers
            for i : 0..(nn.layers[layer_idx].size-1) {
                j := 0;
                padding_x := 0.5 * (max_width - ((10) * spacing));
                spacing_x := (i * spacing);
                if i > 35 {
                    j = 2;
                    padding_x = 0.5 * (max_width - ((11) * spacing));
                    spacing_x = ((i - 36) * spacing);
                }
                else if i > 9 {
                    j = 1;
                    padding_x = 0.5 * (max_width - ((26) * spacing));
                    spacing_x = ((i - 10) * spacing);
                }
                neuron_vertex_instances[offset + i].offset.x = xx padding_x + spacing_x;
                neuron_vertex_instances[offset + i].offset.y = xx padding_y + (j * spacing);
                neuron_vertex_instances[offset + i].offset.z = z;
                store[i] = neuron_vertex_instances[offset + i].offset;
            }
        }
        else {
            for i : 0..(nn.layers[layer_idx].size-1) {
                padding_x := 0.5 * (max_width - ((nn.layers[layer_idx].size - 1) * spacing));
                neuron_vertex_instances[offset + i].offset.x = xx padding_x + (i * spacing);
                neuron_vertex_instances[offset + i].offset.y = xx padding_y;
                neuron_vertex_instances[offset + i].offset.z = z;
                store[i] = neuron_vertex_instances[offset + i].offset;
            }
        }
    
    return nn.layers[layer_idx].size;
}

init_instance_positions :: (nn: *NeuralNet, render_state: *RenderState)
{
    neuron_count: u32 = 0;
    for i:0..nn.layers.count-1 {
        neuron_count += nn.layers[i].size;
    }
    array_resize(*render_state.neuron_vertex_instances, neuron_count);
    print("Total neurons: %\n", neuron_count);

    wire_vertex_count: u32 = 0;
    for i:0..nn.layers.count-2 {
        wire_vertex_count += nn.layers[i].size * nn.layers[i + 1].size;
    }
    print("Total wires: %\n", wire_vertex_count);
    wire_vertex_count *= 2;
    array_resize(*render_state.wire_vertex_data, wire_vertex_count);
    array_resize(*render_state.output_neuron_positions, nn.layers[nn.layers.count - 1].size);

    layer_spacings :: float.[1, 3, 3, 3, 3, 3];
    layer_z_offsets :: float.[40, 20, 20, 20, 20, 20, 20, 20];
    max_width := (sqrt(xx nn.layers[0].size) - 1) * 1.5 * layer_spacings[0];
    offset: u32 = 0;
    offset += layout_square_layer(render_state.neuron_vertex_instances, nn, max_width, 0, offset, 0, layer_spacings[0]);
    z := layer_z_offsets[0];
    for i:1..nn.layers.count-2 {
        offset += layout_square_layer(render_state.neuron_vertex_instances, nn, max_width, xx i, offset, z, layer_spacings[i]);
        z += layer_z_offsets[i];
    } 
    offset += layout_output_layer(render_state.neuron_vertex_instances, nn, max_width, xx (nn.layers.count - 1), offset, z, 2, render_state.output_neuron_positions);
}

depth_sort :: (cam_pos: Vector3, array: [] $T) -> [] T
{
    compare_vertex :: (cam_pos: Vector3, a: InstanceVertex, b: InstanceVertex) -> int
    {
        la := length_squared(cam_pos - a.offset);
        lb := length_squared(cam_pos - b.offset);
        if (la < lb) {
            return 1;
        }
        if (la > lb) {
            return -1;
        }
        return 0;
    };

    quicksort_helper :: (cam_pos: Vector3, a: *T, n: s64, f: (Vector3, T, T) -> int) {
        if n < 2 return;

        p := a[n / 2];
        i := 0;
        j := n-1;
        while true {
            while f(cam_pos, a[i], p) < 0  {i += 1;}
            while f(cam_pos, p, a[j]) < 0  {j -= 1;}

            if i >= j break;

            t := a[i];
            a[i] = a[j];
            a[j] = t;

            i += 1;
            j -= 1;
        }
        
        quicksort_helper(cam_pos, a, i, f);
        quicksort_helper(cam_pos, a + i, n - i, f);
    }

    quicksort_helper(cam_pos, array.data, array.count, compare_vertex);
    return array;    
}

update_neuron_buffers :: (render_state: *RenderState, cam_pos: Vector3, nn: *NeuralNet, instance: *NeuralNetInstance)
{
    mapped: D3D11_MAPPED_SUBRESOURCE;
    hr := ID3D11DeviceContext_Map(render_state.ctx, render_state.vb_neuron_inst, 0, .WRITE_DISCARD, 0, *mapped);
    if FAILED(hr) {
        log_error("Map failed: %", hr);
    }

    ib_data : *InstanceVertex = mapped.pData;
    for _, i : render_state.neuron_vertex_instances {
        ib_data[i].offset = render_state.neuron_vertex_instances[i].offset;
    }
    idx := 0;
    for layer_idx : 0..nn.layers.count-1 {
        state := instance.layer_states[layer_idx];
        for i : 0..(nn.layers[layer_idx].size-1) {
            activ_val := state.activation[i];
            ib_data[idx].colour.x = (activ_val - state.min_activation) / (state.max_activation - state.min_activation);            
            idx += 1;
        }
    }
    
    ib_data_arr : []InstanceVertex = ---;
    ib_data_arr.data = ib_data;
    ib_data_arr.count = render_state.neuron_vertex_instances.count;
    // @TODO - this is pretty slow and significantly limits the render speed
    //depth_sort(cam_pos, ib_data_arr);

    ID3D11DeviceContext_Unmap(render_state.ctx, render_state.vb_neuron_inst, 0);    
}

update_wire_buffers :: (render_state: *RenderState, nn: *NeuralNet, instance: *NeuralNetInstance) -> s64
{
    mapped: D3D11_MAPPED_SUBRESOURCE;
    hr := ID3D11DeviceContext_Map(render_state.ctx, render_state.vb_wire_inst, 0, .WRITE_DISCARD, 0, *mapped);
    if FAILED(hr) {
        log_error("Map failed: %", hr);
    }

    // dim large layers to improve visibility
    ib_data : *InstanceVertex = mapped.pData;
    dst_offset := 0;
    src_offset := 0;
    threshold := render_state.activation_draw_threshold;
    for layer_idx:0..nn.layers.count-2 {
        curr_state := *instance.layer_states[layer_idx];
        next_state := *instance.layer_states[layer_idx + 1];
        for i : 0..(nn.layers[layer_idx].size - 1) {
            for j : 0..(nn.layers[layer_idx + 1].size-1) {
                curr_activ := (curr_state.activation[i] - curr_state.min_activation) / (curr_state.max_activation - curr_state.min_activation);
                next_activ := (next_state.activation[j] - next_state.min_activation) / (next_state.max_activation - next_state.min_activation);
                // multiply since we don't want say 0-valued nodes connected to non-zero nodes to show their wires
                activ_val := (next_activ * curr_activ);
                if (activ_val > threshold) {
                    render_state.wire_vertex_data[dst_offset].colour.x = activ_val;
                    render_state.wire_vertex_data[dst_offset].colour.y = render_state.layer_brightness_scale[layer_idx];
                    render_state.wire_vertex_data[dst_offset].offset = render_state.neuron_vertex_instances[src_offset + i].offset;
                    render_state.wire_vertex_data[dst_offset + 1].colour.x = activ_val;
                    render_state.wire_vertex_data[dst_offset + 1].colour.y = render_state.layer_brightness_scale[layer_idx];
                    render_state.wire_vertex_data[dst_offset + 1].offset = render_state.neuron_vertex_instances[src_offset + nn.layers[layer_idx].size + j].offset;
                    dst_offset += 2;
                }
            }
        }
        src_offset += nn.layers[layer_idx].size;
    }    
    memcpy(ib_data, render_state.wire_vertex_data.data, dst_offset * size_of(InstanceVertex));
    ID3D11DeviceContext_Unmap(render_state.ctx, render_state.vb_wire_inst, 0);
    return dst_offset;
}

compile_shader_d3d :: (source: string, entry_point: string, shader_model: string, row_major: bool = true) -> string, HRESULT
{
    flags := D3DCOMPILE.ENABLE_STRICTNESS;
    if (row_major) {
        flags |= .PACK_MATRIX_ROW_MAJOR;
    }  
    #if DEBUG {
        flags |= .DEBUG;
        flags |= .SKIP_OPTIMIZATION;
    }

    defines: [] string;
    bytecode, errors, hr := D3DCompile(source, "source", defines, null, entry_point, shader_model, xx flags, 0);
    defer free(errors);

    if FAILED(hr) {
        free(bytecode);
        if errors print ("Error msg: %\n", errors);
        return "", hr;
    }
    return bytecode, hr;
}

compile_shader :: (device : *ID3D11Device, source: string, vs_entry: string, ps_entry: string,
    vertex_shader: **ID3D11VertexShader, pixel_shader: **ID3D11PixelShader,
    il_desc: []D3D11_INPUT_ELEMENT_DESC, il: **ID3D11InputLayout)
{
    ps : string;
    hr : HRESULT;
    ps, hr = compile_shader_d3d(source, ps_entry, "ps_4_0");
    if FAILED(hr) {
        log_error("Compile PS failed: %", hr);
    }

    hr = ID3D11Device_CreatePixelShader(device, ps.data, cast(u64) ps.count, null, pixel_shader);
    if FAILED(hr) {
        log_error("CreatePixelShader failed: %", hr);
    }

    vs : string;
    vs, hr = compile_shader_d3d(source, vs_entry, "vs_4_0");
    if FAILED(hr) {
        log_error("Compile VS failed: %", hr);
    }
    assert(vs.data != null);

    hr = ID3D11Device_CreateVertexShader(device, vs.data, cast(u64) vs.count, null, 
        vertex_shader);
    if FAILED(hr) {
        log_error("CreateVertexShader failed: %", hr);
    }
    
    if (il != null) {
        hr = ID3D11Device_CreateInputLayout(device, il_desc.data, cast(u32) il_desc.count,
            vs.data, cast(u64) vs.count, il);
        if FAILED(hr) {
            log_error("CreateInputLayout failed: %", hr);
        }
    }

    free(vs);
    free(ps);
}

init_device :: (render_state: *RenderState) -> bool
{
    hr : HRESULT = S_OK;

    flags : D3D11_CREATE_DEVICE_FLAG;
    #if DEBUG flags |= .DEBUG;

    driver_types   := D3D_DRIVER_TYPE.[.HARDWARE, .WARP, .REFERENCE];
    feature_levels := D3D_FEATURE_LEVEL.[._11_1, ._11_0, ._10_1, ._10_0];

    for 0..driver_types.count-1 {
        render_state.driver_type = driver_types[it];
        hr = D3D11CreateDevice(null, render_state.driver_type, null, flags, feature_levels.data, 
            feature_levels.count, D3D11_SDK_VERSION, *render_state.device, *render_state.feature_level, *render_state.ctx);

        if hr == E_INVALIDARG {
            hr = D3D11CreateDevice(null, render_state.driver_type, null, flags, feature_levels.data + 1, 
            feature_levels.count - 1, D3D11_SDK_VERSION, *render_state.device, *render_state.feature_level, *render_state.ctx);
        }

        if SUCCEEDED(hr) break;
    }
    if FAILED(hr) {
        print("D3D11CreateDevice failed.\n");
        return false;
    }

    assert(render_state.device != null);
    print("Created D3D11 device % with feature level %\n", render_state.device, render_state.feature_level);

    dxgi_factory : *IDXGIFactory1;
    {
        dxgi_device : *IDXGIDevice;
        hr = IUnknown_QueryInterface(render_state.device, *uid(IDXGIDevice_UUID), xx *dxgi_device);
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
    
    samples: u32 = 1;
    while samples <= D3D11_MAX_MULTISAMPLE_SAMPLE_COUNT {
        quality: u32 = 0;
        if (SUCCEEDED(ID3D11Device_CheckMultisampleQualityLevels(render_state.device, DXGI_FORMAT.R8G8B8A8_UNORM, samples, *quality)) && quality > 0) {
            log("Supports % samples with % quality levels\n", samples, quality);
        }
        samples *= 2;
    }

    td : D3D11_TEXTURE2D_DESC;
    td.ArraySize = 1;
    td.BindFlags = .RENDER_TARGET | .SHADER_RESOURCE;
    td.CPUAccessFlags = 0;
    td.Format = .R8G8B8A8_UNORM;
    td.MipLevels = 1;
    td.Usage = .DEFAULT;
    td.MiscFlags = 0;
    td.Width = xx (render_state.window_width * render_state.res_scaling);
    td.Height = xx (render_state.window_height * render_state.res_scaling);
    td.SampleDesc.Count = 8;
    td.SampleDesc.Quality = 0;
    ID3D11Device_CreateTexture2D(render_state.device, *td, null, *render_state.render_texture);

    sd : DXGI_SWAP_CHAIN_DESC;
    sd.BufferCount = 1;
    sd.BufferDesc.Width = xx render_state.window_width;
    sd.BufferDesc.Height = xx render_state.window_height;
    sd.BufferDesc.Format = .R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = .RENDER_TARGET_OUTPUT;
    sd.OutputWindow = render_state.window_hwnd;
    sd.SampleDesc.Count = 8;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = BOOL.TRUE;

    hr = IDXGIFactory_CreateSwapChain(dxgi_factory, render_state.device, *sd, *render_state.swap_chain);
    if FAILED(hr) {
        log_error("CreateSwapChain failed: %", hr);
    }

    IDXGIFactory_MakeWindowAssociation(dxgi_factory, render_state.window_hwnd, cast (u32) DXGI_MWA.NO_ALT_ENTER);
    IUnknown_Release(dxgi_factory);

    if FAILED(hr) return false;
    print("Create a render target view\n");

    back_buffer : *ID3D11Texture2D;
    hr = IDXGISwapChain_GetBuffer(render_state.swap_chain, 0, *uid(ID3D11Texture2D_UUID), xx *back_buffer);
    if FAILED(hr) {
        log_error("GetBuffer failed: %", hr);
        return false;
    }

    hr = ID3D11Device_CreateRenderTargetView(render_state.device, back_buffer, null, *render_state.swap_rtv);
    IUnknown_Release(back_buffer);
    if FAILED(hr) {
        log_error("CreateRenderTargetView failed: %", hr);
        return false;
    }

    hr = ID3D11Device_CreateRenderTargetView(render_state.device, render_state.render_texture, null, *render_state.rtv);
    if FAILED(hr) {
        log_error("CreateRenderTargetView failed: %", hr);
        return false;
    }

    hr = ID3D11Device_CreateShaderResourceView(render_state.device, render_state.render_texture, null, *render_state.srv);
    if FAILED(hr) {
        log_error("CreateShaderResourceView failed: %", hr);
        return false;
    }

    sampler_desc : D3D11_SAMPLER_DESC;
    sampler_desc.Filter         = .MIN_MAG_MIP_LINEAR;
    sampler_desc.AddressU       = .CLAMP;
    sampler_desc.AddressV       = .CLAMP;
    sampler_desc.AddressW       = .CLAMP;
    sampler_desc.ComparisonFunc = .ALWAYS;
    sampler_desc.MinLOD         = 0;
    sampler_desc.MaxLOD         = FLOAT32_MAX;
    ID3D11Device_CreateSamplerState(render_state.device, *sampler_desc, *render_state.sampler);

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
    hr = ID3D11Device_CreateBlendState(render_state.device, *bs_desc, *render_state.bs);
    if FAILED(hr) {
        log_error("ID3D11Device_CreateBlendState failed: %", hr);
        free(render_state.rs);
        return false;
    }

    rs_desc : D3D11_RASTERIZER_DESC;
    rs_desc.CullMode = .NONE;
    rs_desc.FillMode = .SOLID;
    hr = ID3D11Device_CreateRasterizerState(render_state.device, *rs_desc, *render_state.rs);
    if FAILED(hr) {
        log_error("CreateRasterizerState failed: %", hr);
        free(render_state.rs);
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
        
        compile_shader(render_state.device, ML_SHADER_SOURCE, "vs_instanced", "ps_instanced",
            *render_state.vs_instanced, *render_state.ps_instanced, layout, *render_state.il_instanced);
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
        
        compile_shader(render_state.device, ML_SHADER_SOURCE, "vs", "ps",
            *render_state.vs, *render_state.ps, layout, *render_state.il);
    }

    compile_shader(render_state.device, ML_SHADER_SOURCE, "vs_fullscreen", "ps_fullscreen",
        *render_state.vs_fullscreen, *render_state.ps_fullscreen, .{}, null);

    bd : D3D11_BUFFER_DESC;
    bd.Usage = .DEFAULT;
    bd.ByteWidth = size_of(Vector3) * 36;
    bd.BindFlags = .VERTEX_BUFFER;
    bd.CPUAccessFlags = 0;

    InitData : D3D11_SUBRESOURCE_DATA;
    InitData.pSysMem = SHAPE_CUBE.data;

    hr = ID3D11Device_CreateBuffer(render_state.device, *bd, *InitData, *render_state.vb_cube);
    if FAILED(hr) {
        log_error("CreateBuffer failed: %", hr);
        return false;
    }

    bd.ByteWidth = size_of(ShaderConstants);
    hr = ID3D11Device_CreateBuffer(render_state.device, *bd, null, *render_state.cbuffer);
    bd.BindFlags = .CONSTANT_BUFFER;
    if FAILED(hr) {
        log_error("CreateBuffer ShaderConstants failed: %", hr);
        return false;
    }
    
    MAX_INSTANCES :: 1000000;
    bd.ByteWidth = size_of(InstanceVertex) * MAX_INSTANCES;
    bd.Usage = .DYNAMIC;
    bd.CPUAccessFlags = .WRITE;
    bd.BindFlags = .VERTEX_BUFFER;
    hr = ID3D11Device_CreateBuffer(render_state.device, *bd, null, *render_state.vb_neuron_inst);
    hr = ID3D11Device_CreateBuffer(render_state.device, *bd, null, *render_state.vb_wire_inst);
    if FAILED(hr) {
        log_error("CreateBuffer IB failed: %", hr);
        return false;
    }
    return load_gradient_textures(render_state);
}
    
load_gradient_textures :: (render_state: *RenderState) -> bool
{
    hr : HRESULT = S_OK;
    load_texture :: (render_state: *RenderState, file: string, idx: s64) -> HRESULT
    {
        hr : HRESULT = S_OK;
        bitmap: Bitmap;
        if !bitmap_load(*bitmap, file) {
            return E_FAIL;
        }
        defer deinit(*bitmap);
        assert(bitmap.format == .RGB8);

        arr : [..]u8;
        array_resize(*arr, cast(s64)(bitmap.width * bitmap.height * 4));
        i := 0;
        dst := 0;
        while (i < bitmap.data.count ) {
            arr[dst] = bitmap.data[i];
            arr[dst + 1] = bitmap.data[i + 1];
            arr[dst + 2] = bitmap.data[i + 2];
            i += 3;
            dst += 4;
        }
        defer array_free(arr);
        
        td : D3D11_TEXTURE2D_DESC;
        td.ArraySize = 1;
        td.BindFlags = .SHADER_RESOURCE;
        td.CPUAccessFlags = 0;
        td.Format = .R8G8B8A8_UNORM;
        td.MipLevels = 1;
        td.Usage = .DEFAULT;
        td.MiscFlags = 0;
        td.Width = xx bitmap.width;
        td.Height = xx bitmap.height;
        td.SampleDesc.Count = 1;
        td.SampleDesc.Quality = 0;
        srd : D3D11_SUBRESOURCE_DATA;
        srd.pSysMem = arr.data;
        srd.SysMemPitch = xx (4 * bitmap.width);
        srd.SysMemSlicePitch = xx (4 * bitmap.width * bitmap.height);
        hr = ID3D11Device_CreateTexture2D(render_state.device, *td, *srd, *render_state.gradient_textures[idx]);
        if FAILED(hr) {
            log_error("CreateTexture2D failed: %", hr);
            return hr;
        }

        hr = ID3D11Device_CreateShaderResourceView(render_state.device, render_state.gradient_textures[idx], 
            null, *render_state.gradient_texture_srvs[idx]);
        if FAILED(hr) {
            log_error("CreateShaderResourceView failed: %", hr);
            return hr;
        }
        return S_OK;
    }
    hr |= load_texture(render_state, "themes/red_blue.png", 0);
    hr |= load_texture(render_state, "themes/house_of_cards.png", 1);
    hr |= load_texture(render_state, "themes/purple.png", 2);
    hr |= load_texture(render_state, "themes/ygb.png", 3);
    hr |= load_texture(render_state, "themes/neo.png", 4);
    hr |= load_texture(render_state, "themes/blue_white.png", 5);
    hr |= load_texture(render_state, "themes/white.png", 6);
    if FAILED(hr) {
        log_error("Failed to load all gradient textures: %", hr);
        return false;
    }
    return true;
}

cleanup_device :: (render_state: *RenderState)
{
    if render_state.ctx ID3D11DeviceContext_ClearState(render_state.ctx);
    if render_state.vb_cube IUnknown_Release(render_state.vb_cube);
    if render_state.il_instanced IUnknown_Release(render_state.il_instanced);
    if render_state.vs_instanced IUnknown_Release(render_state.vs_instanced);
    if render_state.ps_instanced IUnknown_Release(render_state.ps_instanced);
    if render_state.swap_rtv IUnknown_Release(render_state.swap_rtv);
    if render_state.swap_chain IUnknown_Release(render_state.swap_chain);
    if render_state.ctx IUnknown_Release(render_state.ctx);
    if render_state.device IUnknown_Release(render_state.device);
    if render_state.rs IUnknown_Release(render_state.rs);
    if render_state.bs IUnknown_Release(render_state.bs);
}
