#pragma once
#include <vector>
#include "../matplotlib-cpp/matplotlibcpp.h"

// Goal:
// We want to minimize some convex function f(x) -> min_f(x), x in R^n
// So in other words, we want to find arguments for the provided function such that they will give us the smallest(local) function value
// If the function is not convex then the algorithm will "go" towards local extrema

// First let's approximate f(x) w/ second order Taylor polynomial around point x_k and also insert a certain amount of displacement(t)
// f(x_k + t) =(approx) f(x_k) + f'(x_k)*((x_k + t) - x_k) + 1/2f''(x_k)*((x_k + t) - x_k)^2
// f(x_k + t) =(approx) f(x_k) + f'(x_k)*t + 1/2f''(x_k)*t^2

// Now differentiate this approximation w.r.t to t
// This will tell us the local minimum if f'' > 0 (concave up)
// 
// d/dt(f(x_k) + f'(x_k)*t + 1/2f''(x_k)*t^2) = 0
// f'(x_k) + f''(x_k)t = 0
// t = - f'(x_k)/f''(x_k)

// Next, iteration is defined as: x_k+1 = f(x_k+t) = x_k - f'(x_k)/f''(x_k) for single variable case
// Or more generally x_k+1 = x_k - (grad^2)^-1f(x_k)*(grad)f(x_k) <-- ((grad^2)^-1)f(x) is the inverse of the Hessian matrix

// Sample function
// f = x^2 - 10 + 6x*cos(x)
// f' = 2*x + 6*cos(x)-6*x*sin(x)
// f'' = 2 - 6*sin(x) - 6*sin(x)- 6*x*cos(x)
template <typename T>
class CurveMinExample {
public:
    CurveMinExample(T x_0)
    {
        update(x_0);
        matplotlibcpp::pause(2.5);
    }
    void NewtonsMethod(int k = 5)
    {
        T x_new;
        for (int i = 1; i < k; i++)
        {
            _t = -_f_x / _f_xx;
            x_new = _x + _t;

            update(x_new);

            matplotlibcpp::clf();
            drawFncGraph(3);
            drawPnt(_x, _f);
            matplotlibcpp::pause(1);
            drawTaylorApproxGraph(1);
            matplotlibcpp::pause(1.5);

            std::cout << "Newtons Method: " << i << std::endl;
            std::cout << "x_" << i << ": " << _x << std::endl;
            std::cout << "t: " << _t << std::endl;
            std::cout << "f_dd: " << _f_xx << std::endl;
            std::cout << "---------------------" << std::endl;
        }
    }
private:
    T _f, _f_x, _f_xx;
    T _x, _t;
    void update(T x_i)
    {
        _x = x_i;

        _f = pow(_x, 2) - 10 + 6 * _x * cos(_x);
        _f_x = 2 * _x + 6 * cos(_x) - 6 * _x * sin(_x);
        _f_xx = 2 - 6 * sin(_x) - 6 * sin(_x) - 6 * _x * cos(_x);
    }
    void drawTaylorApproxGraph(double r=5)
    {
        T f_x_p, f_xx_p;
        std::vector<T> _x_p, _y_p;

        for (double i = -r; i < r; i += 0.1)
        {
            _x_p.push_back(_x + i);
            f_x_p = 2 * (_x + i) + 6 * cos(_x + i) - 6 * (_x + i) * sin(_x + i);
            f_xx_p = 2 - 6 * sin(_x + i) - 6 * sin(_x + i) - 6 * (_x + i) * cos(_x + i);

            _y_p.push_back(_f + f_x_p * (i)+1 / 2 * f_xx_p * pow(i, 2));
        }
        matplotlibcpp::named_plot("2nd degree Taylor approx of f(x)", _x_p, _y_p, "g--");
    }
    void drawFncGraph(double r = 5)
    {
        // Draw main function graph
        std::vector<T> f_x, f_y;
        for (double i = -r; i < r; i += 0.1)
        {
            T y = pow(i, 2) - 10 + 6 * i * cos(i);
            f_x.push_back(i);
            f_y.push_back(y);
        }
        matplotlibcpp::plot(f_x, f_y, "m");
    }
    void drawPnt(T x, T f)
    {
        std::vector<T> X, F;
        X.push_back(x);
        F.push_back(f);

        matplotlibcpp::scatter(X, F, 50);
    }
};