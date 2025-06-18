Layer :: struct
{
    weights: [..][..]float;
    biases: [..]float;
}

image_sz :: 28;
layer_sz :: u32.[image_sz * image_sz, 16, 16, 10];
NeuralNet :: struct
{
    layers: [layer_sz.count]Layer;
}

sample_normal :: (mean: float, stddev: float) -> float
{
    u1 := random_get_within_range(0.0000001, 1.0);
    u2 := random_get_within_range(0.0, 1.0);
    z0 := sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    return z0 * stddev + mean;
}

init_weights :: ()
{
    nn : NeuralNet;    
    for n:1..layer_sz.count-1 {
        array_resize(*nn.layers[n].weights, layer_sz[n]);
        array_resize(*nn.layers[n].biases, layer_sz[n]);
        for i:0..layer_sz[n]-1 {
            array_resize(*nn.layers[n].weights[i], layer_sz[n - 1]);
        }
    }
    for n:1..layer_sz.count-1 {
        for i:0..layer_sz[n]-1 {
            for j:0..layer_sz[n-1]-1 {
                nn.layers[n].weights[i][j] = sample_normal(0.0, 2.0 / (layer_sz[n] + layer_sz[n - 1]));
            }
        }
    }
}