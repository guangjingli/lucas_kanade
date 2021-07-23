//
// Created by lgj on 2021/7/21.
//

#ifndef LUCAS_KANADE_LUCAS_KANADA_H
#define LUCAS_KANADE_LUCAS_KANADA_H


#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>


namespace LucasKanda {
    struct Velocity {
        float theta = 0.0F;
        float x = 0.0F;
        float y = 0.0F;

        Velocity add(const Velocity &v) const {
            return { theta + v.theta, x + v.x, y + v.y };

        }

        Velocity neg() const {
            return { -theta, -x, -y };
        }

        cv::Point2f apply(const cv::Point2d &p1) const {
            cv::Point2d p2;
            const float sint = sin(theta);
            const float cost = cos(theta);

            p2.x = cost * p1.x - sint * p1.y + x;
            p2.y = sint * p1.x + cost * p1.y + y;
            return p2;
        }
    };

    class LucasKanada {
    public:
        void align(const cv::Mat &target_mat, const cv::Mat &source_mat, cv::Mat &tf);

    private:
        Velocity lucas_kanada_least_squares(const cv::Mat &target, const cv::Mat &source, const cv::Point2f &loc);

        cv::Mat compute_derivatives(const cv::Mat &src, const std::string &type);

        cv::Mat compute_jacobian(const Velocity &velocity, const cv::Point2f &loc, const cv::Mat &x_gradient,
                                 const cv::Mat &y_gradient);

        cv::Mat
        compute_b(const Velocity &velocity, const cv::Point2f &loc, const cv::Mat &source, const cv::Mat &target);

        float bilinear_interp(const cv::Mat &img, const cv::Point2f &pt);

        Velocity solve_normal_equation(const cv::Mat &jacobian, const cv::Mat &b);

    private:
        cv::Mat source_mat_x_gradient_;
        cv::Mat source_mat_y_gradient_;
    };

}
#endif //LUCAS_KANADE_LUCAS_KANADA_H
