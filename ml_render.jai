#load "ml.jai";
#import "d3d_compiler";

InstanceVertex :: struct {
    offset : Vector3;
    colour : Vector3;
}

neuron_spacing :: 1.5;
neuron_vertex_instances: [..] InstanceVertex;
wire_vertex_data: [..] InstanceVertex;

layout_square_layer :: (max_width: float, layer_idx: u32, offset: u32, z: float) -> u32
{
    dim: u32 = xx sqrt(xx layer_sz[layer_idx]);
    assert((dim * dim) == layer_sz[layer_idx]);
    padding := 0.5 * (max_width - ((dim - 1) * neuron_spacing));
    for i : 0..(dim-1) {
        for j : 0..(dim-1) {
            neuron_vertex_instances[offset + (i * dim) + j].offset.x = xx padding + (i * neuron_spacing);
            neuron_vertex_instances[offset + (i * dim) + j].offset.y = xx padding + (j * neuron_spacing);
            neuron_vertex_instances[offset + (i * dim) + j].offset.z = z;
        }
    }
    return layer_sz[layer_idx];
}

layout_linear_layer :: (max_width: float, layer_idx: u32, offset: u32, z: float) -> u32
{
    padding_x := 0.5 * (max_width - ((layer_sz[layer_idx] - 1) * neuron_spacing));
    padding_y := 0.5 * (max_width - neuron_spacing);
    for i : 0..(layer_sz[layer_idx]-1) {
        neuron_vertex_instances[offset + i].offset.x = xx padding_x + (i * neuron_spacing);
        neuron_vertex_instances[offset + i].offset.y = xx padding_y;
        neuron_vertex_instances[offset + i].offset.z = z;
    }
    return layer_sz[layer_idx];
}

init_instance_positions :: ()
{
    neuron_count: u32 = 0;
    for i:0..layer_sz.count-1 {
        neuron_count += layer_sz[i];
    }
    array_resize(*neuron_vertex_instances, neuron_count);
    print("Total neurons: %\n", neuron_count);

    wire_vertex_count: u32 = 0;
    for i:0..layer_sz.count-2 {
        wire_vertex_count += layer_sz[i] * layer_sz[i + 1];
    }
    print("Total wires: %\n", wire_vertex_count);
    wire_vertex_count *= 2;
    array_resize(*wire_vertex_data, wire_vertex_count);

    max_width :: (image_sz - 1) * neuron_spacing;
    offset: u32 = 0;
    offset += layout_square_layer(max_width, 0, offset, 0);
    offset += layout_square_layer(max_width, 1, offset, 25);
    offset += layout_square_layer(max_width, 2, offset, 40);
    offset += layout_linear_layer(max_width, 3, offset, 60);

    dst_offset := 0;
    src_offset := 0;
    for n:0..layer_sz.count-2 {
        for i : 0..(layer_sz[n] - 1) {
            for j : 0..(layer_sz[n + 1]-1) {
                wire_vertex_data[dst_offset].offset = neuron_vertex_instances[src_offset + i].offset;
                wire_vertex_data[dst_offset + 1].offset = neuron_vertex_instances[src_offset + layer_sz[n] + j].offset;
                dst_offset += 2;
            }
        }
        src_offset += layer_sz[n];
    }
}

init_ml :: ()
{
    init_weights();
    init_instance_positions();
}

neuron_shader_source :: #string END
struct VsInput
{
    float3 pos  : POSITION;
    float3 color   : COLOR;
};

struct VsInputInstanced
{
    float4 pos : POSITION;
    // per instance
    float3 offset  : POSITION1;
    float3 color   : COLOR1;
};

struct PsInput
{
    float4 pos : SV_POSITION;
    float3 pos_o : POSITION;
};

cbuffer cb : register(b0)
{
    matrix g_viewProj;
}

PsInput vs(VsInput input)
{
    PsInput output;
    output.pos_o = input.pos.xyz;
    output.pos = mul(g_viewProj, float4(input.pos.xyz, 1));    
    return output;
}

float4 ps(PsInput input) : SV_Target
{
    return float4(float3(1,1,1), 0.1); 
}

PsInput vs_instanced(VsInputInstanced input)
{
    PsInput output;
    output.pos_o = input.pos.xyz;
    float3 wpos = input.pos.xyz + float3(input.offset.x, input.offset.y, input.offset.z);
    output.pos = mul(g_viewProj, float4(wpos, 1));    
    return output;
}

float4 ps_instanced(PsInput input) : SV_Target
{
    float3 abs_pos = abs(input.pos_o.xyz);
    float edge_width = 0.025;
    float3 mask = smoothstep(0.5 - edge_width, 0.5, abs_pos);
    float alpha = max(max(mask.x * mask.y, mask.y * mask.z), mask.z * mask.x);
    alpha = alpha > 0.5 ? 1.0 : 0.5;
    return float4(float3(1,1,1), alpha); 
}
END

compile_shader_d3d :: (source: string, entry_point: string, shader_model: string) -> string, HRESULT {
    flags := D3DCOMPILE.ENABLE_STRICTNESS;
    flags |= .PACK_MATRIX_ROW_MAJOR;
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

compile_shader :: (source: string, vs_entry: string, ps_entry: string,
    vertex_shader: **ID3D11VertexShader, pixel_shader: **ID3D11PixelShader,
    il_desc: []D3D11_INPUT_ELEMENT_DESC, il: **ID3D11InputLayout)
{
    ps : string;
    hr : HRESULT;
    ps, hr = compile_shader_d3d(source, ps_entry, "ps_4_0");
    if FAILED(hr) {
        log_error("Compile PS failed: %", hr);
    }

    hr = ID3D11Device_CreatePixelShader(g_device, ps.data, cast(u64) ps.count, null, pixel_shader);
    if FAILED(hr) {
        log_error("CreatePixelShader failed: %", hr);
    }

    vs : string;
    vs, hr = compile_shader_d3d(source, vs_entry, "vs_4_0");
    if FAILED(hr) {
        log_error("Compile VS failed: %", hr);
    }
    assert(vs.data != null);

    hr = ID3D11Device_CreateVertexShader(g_device, vs.data, cast(u64) vs.count, null, 
        vertex_shader);
    if FAILED(hr) {
        log_error("CreateVertexShader failed: %", hr);
    }
    
    hr = ID3D11Device_CreateInputLayout(g_device, il_desc.data, cast(u32) il_desc.count,
        vs.data, cast(u64) vs.count, il);
    if FAILED(hr) {
        log_error("CreateInputLayout failed: %", hr);
    }

    free(vs);
    free(ps);
}