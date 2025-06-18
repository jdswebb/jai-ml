// SPDX-FileCopyrightText: James Webb
// SPDX-License-Identifier: MIT
// Full details are in the LICENSE file at the repository root.

ML_SHADER_SOURCE :: #string END
struct VsInput
{
    float3 pos  : POSITION;
    float3 col   : COLOR;
};

struct VsInputInstanced
{
    float4 pos : POSITION;
    // per instance
    float3 offset  : POSITION1;
    float3 col   : COLOR1;
};

struct PsInput
{
    float4 pos : SV_POSITION;
    float3 pos_o : POSITION;
    float3 col : COLOR;
};

cbuffer cb : register(b0)
{
    matrix g_viewProj;
    float g_bias;
    float g_time;
    float2 g_unused;
}

Texture2D g_renderTarget : register(t0);
Texture2D g_gradient : register(t1);
SamplerState g_linearSampler : register(s0);

PsInput vs(VsInput input)
{
    PsInput output;
    output.pos_o = input.pos.xyz;
    output.pos = mul(g_viewProj, float4(input.pos.xyz, 1));
    output.col = input.col;
    return output;
}

float apply_bias(float x, float b)
{
    if (b <= 0.0) return 0.0;
    if (b >= 1.0) return 1.0;
    return x / ((1.0 / b - 2.0) * (1.0 - x) + 1.0);
}

float4 ps(PsInput input) : SV_Target
{
    float bias = g_bias;    
    float x = saturate(input.col.x);
    float remapped = apply_bias(x, bias);
    float4 color = g_gradient.Sample(g_linearSampler, float2(remapped, 0.5));
    float time_shade = 0.5f + 0.25f * (sin((input.pos_o.z - (g_time * 5)) / 2) + 1);
    return float4(color.xyz, time_shade * input.col.x * input.col.y); 
}

PsInput vs_instanced(VsInputInstanced input)
{
    PsInput output;
    output.pos_o = input.pos.xyz;
    float3 wpos = input.pos.xyz + float3(input.offset.x, input.offset.y, input.offset.z);
    output.pos = mul(g_viewProj, float4(wpos, 1));
    output.col = input.col;
    return output;
}

float4 ps_instanced(PsInput input) : SV_Target
{
    float bias = g_bias;
    float x = saturate(input.col.x);
    float remapped = apply_bias(x, bias);
    float3 abs_pos = abs(input.pos_o.xyz);
    float edge_width = 0.025;
    float3 mask = smoothstep(0.5 - edge_width, 0.5, abs_pos);
    float alpha = max(max(mask.x * mask.y, mask.y * mask.z), mask.z * mask.x);
    float final_alpha = alpha > 0.5 ? 0.2 + (0.75 * input.col.x) : (0.75 * input.col.x);
    float4 color = g_gradient.Sample(g_linearSampler, float2(remapped, 0.5));
    return float4(alpha > 0.5 ? float3(1,1,1) : color.xyz, final_alpha); 
}

struct PsFullScreen
{
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD0;
};

PsFullScreen vs_fullscreen(uint vertexID : SV_VertexID)
{
    PsFullScreen output;
    float2 pos = float2((vertexID << 1) & 2, vertexID & 2);
    output.pos = float4(pos * 2.0f - 1.0f, 0.0f, 1.0f);
    output.uv = float2(pos.x, 1.0f - pos.y);
    return output;
}

float4 ps_fullscreen(PsFullScreen input) : SV_Target
{
    float2 texel_size = 1.0 / float2(1650*2, 1024*2);
    float2 uv = input.uv;

    float2 offset[9] = {
        float2(-1, -1), float2(0, -1), float2(1, -1),
        float2(-1,  0), float2(0,  0), float2(1,  0),
        float2(-1,  1), float2(0,  1), float2(1,  1)
    };

    float weight[9] = {
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
    };

    float4 color = float4(0, 0, 0, 0);
    float total_weight = 0;

    [unroll]
    for (int i = 0; i < 9; ++i) {
        float2 sample_uv = uv + offset[i] * texel_size;
        color += g_renderTarget.Sample(g_linearSampler, sample_uv) * weight[i];
        total_weight += weight[i];
    }

    return color / total_weight;
}

END