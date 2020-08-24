#include <iostream>
#include <random>

// Seed and Random
std::default_random_engine gen;
std::uniform_real_distribution<double> randn(0.0, 1.0);

class dense
{
private:
    // 2 dimenisional Array
    double *weights;
    double *biases;

public:
    int in_features;
    int out_features;

    dense(int, int);

    void get_weights();
    void get_biases();
    void get_info();

    int get_in_features();
    int get_out_features();

    double *forward(double *);
};

class model
{
private:
    dense **layers;
    int n_layers=3;

public:
    model();
    double *forward(double *);
};

model::model()
{
    layers = new dense*[n_layers]; // 3 Layer Neural Network

    dense l1(1, 5);
    dense l2(5, 10);
    dense l3(10, 3);

    layers[0] = &l1;
    layers[1] = &l2;
    layers[2] = &l3;
}

int dense::get_in_features()
{
    return in_features;
}

int dense::get_out_features()
{
    return out_features;
}

double *model::forward(double *x)
{
    double *y = layers[0]->forward(x);
    y = layers[1]->forward(y);
    y = layers[2]->forward(y);

    for (int i = 0; i < layers[2]->out_features; i++)
        std::cout << y[i] << "\n";

    return y;
}

dense::dense(int in_f, int out_f)
{
    in_features = in_f;
    out_features = out_f;

    weights = new double[in_features * out_features];
    biases = new double[out_features];

    // Weight Initialization
    for (int in = 0; in < in_f; in++)
        for (int out = 0; out < out_f; out++)
            weights[in * in_features + out] = randn(gen) * 0.1;

    // Biases Initialization
    for (int out = 0; out < out_f; out++)
        biases[out] = randn(gen) * 0.1;
}

double *dense::forward(double *x)
{
    // Equal to numpy.dot()
    // https://www.mathsisfun.com/algebra/matrix-multiplying.html
    double *y = new double[out_features];

    for (int i = 0; i < in_features; i++)
        for (int j = 0; j < out_features; j++)
            y[j] += (x[i] * weights[i * in_features + j]) + biases[j];

    return y;
}

void dense::get_weights()
{
    for (int y = 0; y < in_features; y++)
    {
        std::cout << "[";
        for (int x = 0; x < out_features; x++)
            std::cout << weights[y * in_features + x] << ",";
        std::cout << "],"
                  << "\n";
    }
}

void dense::get_biases()
{
    std::cout << "[";
    for (int y = 0; y < out_features; y++)
        std::cout << biases[y] << ",";

    std::cout << "],"
              << "\n";
}

void dense::get_info()
{
    std::cout << "in_features: " << in_features << "\n";
    std::cout << "out_features: " << out_features << "\n";
    std::cout << "Weights: "
              << "\n";
    get_weights();
    std::cout << "Biases: "
              << "\n";
    get_biases();
}

double *random_input(int size)
{
    double *x = new double[size];

    for (int i = 0; i < size; i++)
        x[i] = randn(gen);

    return x;
}

int main()
{
    int in_features = 3;
    int out_features = 5;

    double *x = random_input(in_features);
    double *out = 0;

    model test;
    test.forward(x);
    return 0;
}
