//
// Created by lgj on 2021/7/21.
//

#include "lucas_kanada.h"


static constexpr float half_window_size_ = 100.0F;


/**
 * @brief
 *
 * @param
 *
 * */
void LucasKanada::align(const cv::Mat &target_mat, const cv::Mat &source_mat, cv::Mat &tf) {

    source_mat_x_gradient_.release();
    source_mat_y_gradient_.release();

    cv::Point2f location = cv::Point2f(source_mat.cols/2.0f, source_mat.rows/2.0f);
    std::cout << location.x << ", " << location.y << std::endl;

    cv::Point2f tf_ = lucas_kanada_least_squares(target_mat, source_mat, location);

    std::cout << tf_.x << ", " << tf_.y << std::endl;

}

cv::Point2f LucasKanada::lucas_kanada_least_squares(const cv::Mat &target, const cv::Mat &source, const cv::Point2f &loc) {
    source_mat_x_gradient_ = compute_derivatives(source, "x");
    source_mat_y_gradient_ = compute_derivatives(source, "y");
    cv::Mat source_mat_gardient = abs(source_mat_x_gradient_) + abs(source_mat_y_gradient_);
    cv::imshow("source_mat_gardient", source_mat_gardient);

    cv::waitKey(0);

     constexpr size_t MAX_ITERS = 20;

     cv::Point2f loc_cur_iter = loc;
     cv::Point2f optimization_vars;
     optimization_vars.x = 0;
     optimization_vars.y = 0;

     for(size_t iter = 0; iter < MAX_ITERS; ++iter){
         const cv::Mat J = compute_jacobian(optimization_vars, loc_cur_iter, source_mat_x_gradient_, source_mat_y_gradient_);

         const cv::Mat b = compute_b(optimization_vars, loc_cur_iter, source, target);

         cv::Point2f delta = solve_normal_equation(J, b);

         optimization_vars += delta;
         // Minus because of dx,dy is modeled as last_x + dx = cur_x;   source + dx = target
         loc_cur_iter -= delta;

         std::cout << "MAX_ITERS_" <<  iter << "\tdelta=(" << delta.x << "," << delta.y << ")=" << delta.dot(delta) << "\toptimization_vars=("
                   << optimization_vars.x << "," << optimization_vars.y << ") loc_iter: (" << loc_cur_iter.x << ", " << loc_cur_iter.y << ") "  << std::endl;

         if(delta.dot(delta) < 1e-4){
             break;
         }
     }
     return optimization_vars;
}


cv::Mat LucasKanada::compute_derivatives(const cv::Mat& src, const std::string& type){
    cv::Mat result;

    if(type == "x" || type == "X"){
        cv::Mat x_kernel = (cv::Mat_<double>(3, 3) << 0, 0, 0,
                -0.5, 0, 0.5,
                0, 0, 0);

        cv::filter2D(src, result, CV_32F, x_kernel);
    }
    else if(type == "y" || type == "Y"){
        cv::Mat y_kernel = (cv::Mat_<double>(3, 3) << 0, -0.5, 0,
                0, 0, 0,
                0, 0.5, 0);
        cv::filter2D(src, result, CV_32F, y_kernel);

    }
    assert(result.cols == src.cols);
    assert(result.rows == src.rows);

    return result;
}

cv::Mat LucasKanada::compute_jacobian(const cv::Point2f &velocity,
                                      const cv::Point2f &loc,
                                      const cv::Mat &x_gradient,
                                      const cv::Mat &y_gradient) {

    const int patch_size = (half_window_size_ * 2 + 1) * (half_window_size_ * 2 + 1);
    cv::Mat jacobian(patch_size, 2, CV_32F);

    size_t count = 0;
    // The 2 for loops can be generalized by a kernal function.
    for (float y = loc.y - half_window_size_; y <= loc.y + half_window_size_ + 1e-6; y += 1.) {
        for (float x = loc.x - half_window_size_; x <= loc.x + half_window_size_ + 1e-6; x += 1.) {
            const float last_x = x + velocity.x;
            const float last_y = y + velocity.y;
            // dx
            jacobian.at<float>(count, 0) = -bilinear_interp(x_gradient, { last_x, last_y });
            // dy
            jacobian.at<float>(count, 1) = -bilinear_interp(y_gradient, { last_x, last_y });
            ++count;
        }
    }
    assert(static_cast<int>(count) == patch_size);
    return jacobian;
}


cv::Mat LucasKanada::compute_b(const cv::Point2f &velocity, const cv::Point2f &loc, const cv::Mat &source,
                               const cv::Mat &target) {
    /// residual function的一行是A的一个像素差， 和移动过的B的一个像素的差
    const int patch_size = (half_window_size_ * 2 + 1) * (half_window_size_ * 2 + 1);
    cv::Mat b (patch_size, 1, CV_32F);

    size_t count = 0;
    for(float y = loc.y - half_window_size_; y <= loc.y + half_window_size_ + 1e-6; y+=1.0F){
        for(float x = loc.x - half_window_size_; x <= loc.x + half_window_size_ + 1e-6; x+=1.0F){
            const float last_x = x + velocity.x;
            const float last_y = y + velocity.y;

            b.at<float>(count, 0) = bilinear_interp(target, {x, y}) - bilinear_interp(source, {last_x, last_y});
            ++count;
        }
    }
    assert(static_cast<int>(count) == patch_size);
    return b;
}

float LucasKanada::bilinear_interp(const cv::Mat &img, const cv::Point2f &pt) {
    assert(!img.empty());
    assert(img.channels() == 1);

    // ceil
    const int x = static_cast<int>(pt.x);
    const int y = static_cast<int>(pt.y);

    const int x0 = cv::borderInterpolate(x, img.cols, cv::BORDER_DEFAULT);
    const int x1 = cv::borderInterpolate(x + 1, img.cols, cv::BORDER_DEFAULT);
    const int y0 = cv::borderInterpolate(y, img.rows, cv::BORDER_DEFAULT);
    const int y1 = cv::borderInterpolate(y + 1, img.rows, cv::BORDER_DEFAULT);

    const float dx = pt.x - static_cast<float>(x);
    const float dy = pt.y - static_cast<float>(y);

    assert(dx >= 0 && dy >= 0 && "dx dy less than 0");


    const float v1 = img.at<float>(y0, x0);
    const float v2 = img.at<float>(y0, x1);
    const float v3 = img.at<float>(y1, x0);
    const float v4 = img.at<float>(y1, x1);

    const float val = (v1 * (1.0F - dx) + v2 * dx) * (1.0F - dy)
                      + (v3 * (1.0F - dx) + v4 * dx) * dy;

    return val;
}


cv::Point2f LucasKanada::solve_normal_equation(const cv::Mat &jacobian, const cv::Mat &b) {

    cv::Mat_<float> delta;
    // J^T J delta = -J^T b

    cv::solve(jacobian, -b, delta, cv::DECOMP_CHOLESKY | cv::DECOMP_NORMAL);

    return { delta.at<float>(0, 0), delta.at<float>(1, 0) };
}
