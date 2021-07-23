//
// Created by lgj on 2021/7/21.
//

#ifndef LUCAS_KANADE_LUCAS_KANADA_H
#define LUCAS_KANADE_LUCAS_KANADA_H


#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>


class LucasKanada {
public:
    void align(const cv::Mat& target_mat, const cv::Mat& source_mat, cv::Mat& tf);

private:
    cv::Point2f lucas_kanada_least_squares(const cv::Mat& target, const cv::Mat& source, const cv::Point2f& loc);

    cv::Mat compute_derivatives(const cv::Mat& src, const std::string& type);

    cv::Mat compute_jacobian(const cv::Point2f& velocity, const cv::Point2f& loc, const cv::Mat& x_gradient, const cv::Mat& y_gradient);

    cv::Mat compute_b(const cv::Point2f& velocity, const cv::Point2f& loc, const cv::Mat& source, const cv::Mat& target);

    float bilinear_interp(const cv::Mat& img, const cv::Point2f& pt);

    cv::Point2f solve_normal_equation(const cv::Mat& jacobian, const cv::Mat& b);

private:
    cv::Mat source_mat_x_gradient_;
    cv::Mat source_mat_y_gradient_;
};


#endif //LUCAS_KANADE_LUCAS_KANADA_H
