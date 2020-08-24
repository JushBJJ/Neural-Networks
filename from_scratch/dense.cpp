#include <iostream>
#include <random>
#include <chrono>

// Seed and Random
unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine gen(seed);
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

    void init(int, int);

    void get_weights();
    void get_biases();
    void get_info();

    int get_in_features();
    int get_out_features();

    double *forward(double *);
};

void dense::init(int in_f, int out_f)
{
    this->in_features = in_f;
    this->out_features = out_f;

    this->weights = new double[in_f * out_f];
    this->biases = new double[out_f];

    // Weight Initialization
    for (int in = 0; in < in_f; in++)
        for (int out = 0; out < out_f; out++)
            this->weights[in * in_f + out] = randn(gen) * 0.1;

    // Biases Initialization
    for (int out = 0; out < out_f; out++)
        this->biases[out] = randn(gen) * 0.1;
}

double *dense::forward(double *x)
{
    // Equal to numpy.dot()
    // https://www.mathsisfun.com/algebra/matrix-multiplying.html
    double *out = new double[this->out_features];

    for (int j = 0; j < this->out_features; j++)
        for (int i = 0; i < this->in_features; i++)
            out[j] += (x[i] * this->weights[j * this->in_features + i]) + this->biases[j];

    return out;
}

void dense::get_weights()
{
    for (int y = 0; y < this->in_features; y++)
    {
        std::cout << "[";
        for (int x = 0; x < this->out_features; x++)
            std::cout << weights[y * this->in_features + x] << ",";
        std::cout << "],"
                  << "\n";
    }
    std::cout<<"\n";
}

void dense::get_biases()
{
    std::cout << "[";
    for (int y = 0; y < this->out_features; y++)
        std::cout << biases[y] << ",";

    std::cout << "],"
              << "\n\n";
}

void dense::get_info()
{
    std::cout << "in_features: " << this->in_features << "\n";
    std::cout << "out_features: " << this->out_features << "\n";
    std::cout << "Weights: "
              << "\n";

    get_weights();

    std::cout << "Biases: "
              << "\n";

    get_biases();
}

class model
{
private:
    dense layers[3];

public:
    int n_layers = 3;

    model();
    double *forward(double *);
    void show_info();
};

int dense::get_in_features()
{
    return this->in_features;
}

int dense::get_out_features()
{
    return this->out_features;
}

model::model()
{
    dense ly1;
    dense ly2;
    dense ly3;

    ly1.init(5, 10);
    ly2.init(10, 3);
    ly3.init(3, 2);

    this->layers[0] = ly1;
    this->layers[1] = ly2;
    this->layers[2] = ly3;
}

void model::show_info()
{
    for (int i = 0; i < sizeof(this->layers) / sizeof(this->layers[0]); i++)
    {
        std::cout << "Layer " << i << "\n";
        this->layers[i].get_info();

        std::cout << "\n";
    }
}

double *model::forward(double *x)
{
    double *y = x;

    // Forward all layers
    y = this->layers[0].forward(x);
    y = this->layers[1].forward(y);
    y = this->layers[2].forward(y);

    // Print all values
    std::cout<<"Output: \n";

    std::cout<<"[";
    for (int i = 0; i < this->layers[2].out_features; i++)
        std::cout << y[i]<<",";
    
    std::cout<<"],\n";
    return y;
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

    std::cout << "Model Info: "
              << "\n";

    test.show_info();
    test.forward(x);
    return 0;
}
