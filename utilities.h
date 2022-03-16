#pragma once
#include <vector>
#include <Eigen>
#include <iostream>
#include "../matplotlib-cpp/matplotlibcpp.h"


template<typename T>
void CalcJacobianResiduals(Eigen::MatrixX<T>& J_out, std::vector<double> (&fnc)(std::vector<T>&, std::vector<T>&), std::vector<std::vector<T>>& X, std::vector<T> C)
{
	// h represents just a tiny nudge in some direction..
	T h = sqrt(DBL_EPSILON);

	Eigen::MatrixX<T> J(X.size()*X[0].size(), C.size());

	for (int i = 0; i < C.size(); i++)
	{
		std::vector<T> C_plus = C;
		std::vector<T> C_minus = C;

		C_plus[i] = C_plus[i] + h;
		C_minus[i] = C_minus[i] - h;

		for (int j = 0; j < X.size(); j++)
		{
			// now taking derivatives of each function
			// note that these functions only differ in regards to prediction-observed input data
			// 
			// r_i(B)  = y_i - f(x_i,B)
			// r_i'(B) = -'f(x_i,B)

			// Typical numerical derivative (fnc(x + h) - fnc(x - h)) / 2 * h;
			// But x in higher dimensions

			std::vector<T> d_1 = (fnc(X[j], C_plus));
			std::vector<T> d_2 = (fnc(X[j], C_minus));

			for (int l = 0; l < d_1.size(); l++)
			{
				// d_l is a partial derivative of a give functio r_j w.r.t to component C_
				// Note there's minus sign because in reality we are taking derivative of r_i = Y_j - model_fnc(X_j, C)
				// but Y_i just a constant term so we're skipping it
				T d_l = -(d_1[l] - d_2[l]) / (2 * h);
				J(X[0].size() * j + l, i) = d_l;
			}
		}
	}
	J_out = J;
}