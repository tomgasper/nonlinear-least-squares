#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen>
#include "SurfaceMinExample.h"
#include "CurveMinExample.h"
#include "utilities.h"
#include "GaussNewton.h"

// Example model function
std::vector<double> model_fnc(std::vector<double>& X, std::vector<double>& C)
{
    double o = (C[0] * X[0]) / (C[1] + X[0]);

    std::vector<double> v{ o };
    return v;
};

int main()
{
    std::vector<double> X = { 0.038, 0.194, 0.425, 0.626, 1.253, 2.500, 3.740 };
    std::vector<double> Y = { 0.050, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317 };
    std::vector<double> B = { 0.9, 0.5 };

    std::vector<std::vector<double>> X_v,Y_v;
    for (int i = 0; i < X.size(); i++)
    {
        X_v.push_back(std::vector<double>{X[i]});
        Y_v.push_back(std::vector<double>{Y[i]});
    }
    
    double n = 5;
    std::vector<double> p, x, y;

    // GaussNewton Algorithm
    
    //Example: Optimizing parameters for the model function: R = x*B_1/x+B_2, using X, Y data points 
    std::vector<double> optim_params;
    GaussNewton fnc(model_fnc, X_v, Y_v, B);

    fnc.Optimize(5, optim_params);

    std::cout << "Optimized Parameters: " << std::endl;
    std::cout << "---------------------" << std::endl;
    for (int i = 0; i < optim_params.size(); i++)
    {
        std::cout << "B_" << i << ": " << optim_params[i] << std::endl;
    }

    for (double i = 0; i < n; i += 0.1)
    {
        x.push_back(i);
        y.push_back((optim_params[0] * i) / (optim_params[1] + i));
    }

    matplotlibcpp::plot(x, y, "m");
    matplotlibcpp::scatter(X, Y, 10.0);

    // Newtons Method
    if (PyArray_API == NULL)
    {
        Py_Initialize();
        import_array();
    }

    // Example: looking for local extrema of: f(x,y)=4+x^3+y^3-3xy
    std::vector<double> x_x{ 54 }, x_y{ 32 };
    std::vector<double> X_0 = { x_x[0], x_y[0] };

    SurfaceMinExample f(X_0);
    f.NewtonsMethod();

    // Example: looking for local extrema of: f(x) = x^2 - 10 + 6x*cos(x)
    double x_0 = 2;

    CurveMinExample f_1(x_0);
    f_1.NewtonsMethod();

    matplotlibcpp::show();
}