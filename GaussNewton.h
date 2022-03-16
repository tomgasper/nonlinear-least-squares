#pragma once

#include <Eigen>
#include "utilities.h"

template<typename T>
class GaussNewton
{
public:
    GaussNewton(std::vector<double> (&fnc)(std::vector<T>&, std::vector<T>&), std::vector<std::vector<T>>& X,std::vector<std::vector<T>>& Y, std::vector<T> C) : _fnc(fnc)
    {
        // Gauss-Newton algorithm iteratively finds the value of the variables that minimizes the sum of squares

        // m - number of functions
        // functions -> r_i = r_1, ..., r_m
        // n variables B = (B_1, ..., B_n)
        // with m >= n

        // basically just calculate the jacobian...

        // r functions are:
        // r_i(B) = y_i - f(x_i, B)
        // where (x_i, y_i) data points
        // f - model function
        // can be thought as residual functions.. 

        // Then B(curr iter + 1) = B(curr iter) + (J_f^T*J_f)^-1 * J_f^T*r(B(curr iter))
        // (J_f^T*J_f)^-1 * J_f^T -> left pseudoinverse of J_f

        _X = X;
        _Y = Y;
        _B = C;
        _J = Eigen::MatrixX<T>(X.size() * X[0].size(), C.size());
        _R = Eigen::MatrixX<T>(X.size() * X[0].size(), 1);
    }
    void Optimize(int n_iter, std::vector<T>& B_out)
    {
        CalcJacobianResiduals(_J, _fnc, _X, _B);
        for (int i = 0; i < n_iter; i++)
        {
            iterate(_fnc);
        }

        B_out = _B;
    }
private:
    Eigen::MatrixX<T> _R, _J;
    std::vector<T> _B;
    std::vector<std::vector<T>> _X, _Y;
    int _n_iter;
    std::vector<double>(&_fnc)(std::vector<T>&, std::vector<T>&);

    void iterate(std::vector<double>(&fnc)(std::vector<T>&, std::vector<T>&))
    {
        Eigen::MatrixX<double> J_pseudo;

        // We are solving pseudoinverse of J because
        // J acts the same as matrix A in: A^T*A*x(approx) = A^Tb (normal equation)
        // or x(approx) = ((A^T*A*x)^-1*A^T)*b
        // 
        // When we approximate residuals vector with Taylor expansion
        // r(B) =(approx) r(B(s)) + J_r(B(s))*delta , with delta = B-B(s)
        // we want to find delta such that r(B) is as small as possible -> min||r(B(s)) + J_r(B(s))*delta||^2
        // 
        // Note that:
        // In normal equations the projection error is: ||e||^2 = || b - A*x(approx) |
        // and solution x(approx) that minimizes it is equal to solving (pseudoinverse of A)*b
        // So minimizing ||r(B(s)) + J_r(B(s))*delta|| can be though of as minimizing the projection error in normal equation
        // And for that it's enough to solve (pseudoinverse of A)*b
        // This way we will find the change in parameters that will minimize the sum of squares of residuals(for current iteration)

        J_pseudo = (_J.transpose() * _J).inverse() * _J.transpose();

        for (int j = 0; j < _Y.size(); j++)
        {
            std::vector<T> v = fnc(_X[j], _B);
            for (int i = 0; i < v.size(); i++)
            {
                _R(j*v.size()+i, 0) = _Y[j][i] - v[i];
            }   
        }

        // computing delta for current iteration
        Eigen::MatrixX<T> _B_J = J_pseudo * _R;

        for (int l = 0; l < _B.size(); l++)
        {
            _B[l] = _B[l] - _B_J(l, 0);
        }

        std::cout << "Iter: " << _n_iter << std::endl;
        for (int i =0; i < _B.size(); i++)
        {
            std::cout << "B_" << i << ": " << _B[i] << std::endl;
        }
        std::cout << "---------------------" << std::endl;

        _n_iter++;
    }
};