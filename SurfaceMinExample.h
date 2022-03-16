#pragma once
#include <Eigen>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

// Generalization of the 2D method,
// same as CurveMin but in higher dimensions so using Hessian and gradient vector instead of single var derivatives
template <typename T>
class SurfaceMinExample
{
public:
    SurfaceMinExample(std::vector<T>& X_0) : X(X_0)
    {
        if (X.size() < 2) throw std::runtime_error("Input vector must be dim >= 2");
        H = Eigen::MatrixX<T>(X.size(), X.size());
        f_g = Eigen::MatrixX<T>(X.size(), 1);
        t = Eigen::MatrixX<T>(X.size(), 1);

        update(X_0);
    }
    void NewtonsMethod(int k=10)
    {
        for (int i = 0; i < k; i++)
        {
            if ( H.determinant() == 0 || H.rows() > H.cols() ) throw std::runtime_error("Incorrect Hessian Matrix, can't be inverted!");
            if ( f_g.rows() != H.cols() ) throw std::runtime_error("Incorrect Gradient Matrix");
            t = H.inverse() * f_g;

            if (t.rows() != X.size()) throw std::runtime_error("Displacement vector must have the same dimension as the position vector");
            for (int j = 0; j < t.rows(); j++)
            {
                X[j] = X[j] - t(j, 0);
            }
            update(X);

            std::cout << "Iteration #: " << i << std::endl;
            std::cout << "X: " << "[" << X[0] << "," << X[1] << "]" << std::endl;
            std::cout << "F(X): " << f << std::endl;
            std::cout << "Current step: " << '\n' << t << std::endl;
            std::cout << "---------------------" << std::endl;
        }
    }
    std::vector<T>& GetX()
    {
        return X;
    }
    Eigen::MatrixX<T>& GetH()
    {
        return H;
    }
    Eigen::MatrixX<T>& GetF_g()
    {
        return f_g;
    }
private:
    T f, f_x, f_y, f_xx, f_xy, f_yy, f_yx;
    std::vector<T>& X;
    Eigen::MatrixX<T> H, f_g,t;

    void update(std::vector<T>& X_i)
    {
        if (!setX(X_i)) throw std::runtime_error("Incorrect vector");

        f = 4 + pow(X_i[0], 3) + pow(X_i[1], 3) - 3 * X_i[0] * X_i[1];
        f_x = 3 * pow(X_i[0], 2) - 3 * X_i[1];
        f_y = 3 * pow(X_i[1], 2) - 3 * X_i[0];
        f_xx = 6 * X_i[0];
        f_xy = -6;
        f_yy = 6 * X_i[1];
        f_yx = -3;

        H(0, 0) = f_xx;
        H(1, 0) = f_yx;

        H(0, 1) = f_xy;
        H(1, 1) = f_yy;

        f_g(0, 0) = f_x;
        f_g(1, 0) = f_y;
    }
    bool setX(std::vector<T>& X_in)
    {
        if (X_in.size() != X.size())
        {
            throw std::runtime_error("Input vector must have the same dimension as the constructor vector");
            return false;
        }
        else { X = X_in; return true; }
    }
};