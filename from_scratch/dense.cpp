#include <iostream>
#include <random>

// Seed and Random
std::default_random_engine gen;
std::uniform_real_distribution<double> randn(0.0, 1.0);

class net
{
        private:
                int in_features;
                int out_features;

                // 2 dimenisional Array (10,10)
                double weights[10][10];
                double biases[10];
        public:
                net(int, int);

                void get_weights();
                void get_biases();
                void get_info();

                double forward(double*, int);
};

net::net(int in_f, int out_f)
{
        in_features=in_f;
        out_features=out_f;

        // Weight Initialization
        for(int in=0; in<in_f; in++)
        {
                for(int out=0; out<out_f; out++)
                        weights[in][out]=randn(gen);
        }

        // Biases Initialization
        for (int out=0; out<out_f; out++)
                biases[out]=randn(gen);
}

double net::forward(double *x, int size)
{
        // Equal to numpy.dot()
        // https://www.mathsisfun.com/algebra/matrix-multiplying.html
        double y=0;

        for(int i=0; i<in_features; i++)
        {
                for(int j=0; j<out_features; j++)
                {
                        y+=*(x+i)*weights[i][j];
                }
        }

        return y;
}

void net::get_weights()
{
        for(int y=0; y<in_features; y++)
        {
                std::cout<<"[";
                for(int x=0; x<out_features; x++)
                        std::cout<<weights[y][x]<<",";
                std::cout<<"],"<<"\n";
        }
}

void net::get_biases()
{
        std::cout<<"[";
        for(int y=0; y<out_features; y++)
                std::cout<<biases[y]<<",";

        std::cout<<"],"<<"\n";
}

void net::get_info()
{
        std::cout<<"in_features: "<<in_features<<"\n";
        std::cout<<"out_features: "<<out_features<<"\n";
        std::cout<<"Weights: "<<"\n";
        get_weights();
        std::cout<<"Biases: "<<"\n";
        get_biases();
}

double *random_input(int size){
        double *x=new double[size];

        for(int i=0; i<size; i++){
                x[i]=randn(gen);
        }

        return x;
}

int main(){
        int in_features=3;
        int out_features=1;
        
        double *x=random_input(in_features);
        double out=0;

        net net1(in_features, out_features);
        net1.get_info();

        std::cout<<"Forward: "<<"\n";

        out=net1.forward(x, in_features);
        std::cout<<out<<"\n";
        return 0;
}
