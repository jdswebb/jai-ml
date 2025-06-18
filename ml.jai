layer_0_dim :: 28;
layer_0_size :: layer_0_dim * layer_0_dim;
layer_1_dim :: 4;
layer_1_size :: layer_1_dim * layer_1_dim;
layer_2_dim :: 4;
layer_2_size :: layer_2_dim * layer_2_dim;
layer_3_size :: 10;

sample_normal :: (mean: float, stddev: float) -> float {
    u1 := random_get_within_range(0.0000001, 1.0);
    u2 := random_get_within_range(0.0, 1.0);
    z0 := sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    return z0 * stddev + mean;
}

layer_0_weights: [layer_0_size * layer_1_size]float;
layer_1_weights: [layer_1_size * layer_2_size]float;
layer_2_weights: [layer_2_size * layer_3_size]float;

init_weights :: ()
{
    for i : 0 .. layer_0_weights.count-1 {
        layer_0_weights[i] = sample_normal(0.0, 2.0 / (layer_0_size + layer_1_size));
    }
    for i : 0 .. layer_1_weights.count-1 {
        layer_1_weights[i] = sample_normal(0.0, 2.0 / (layer_1_size + layer_2_size));
    }
    for i : 0 .. layer_2_weights.count-1 {
        layer_2_weights[i] = sample_normal(0.0, 2.0 / (layer_2_size + layer_3_size));
    }
}