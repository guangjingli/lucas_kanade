//
// Created by lgj on 2021/7/21.
//

#include "lucas_kanada.h"

const std::string image_path = "/home/lgj/workspace/read_ws/lucas_kanade/data/chessboard.png";


cv::Mat transformImage(const cv::Mat& src, const float& angle, const float& delta_x, const float& delta_y){
    cv::Mat dst;

    cv::Size src_sz = src.size();
    cv::Size dst_sz (src_sz.width, src_sz.height);
    int len = std::max(src.cols, src.rows);

    cv::Point2f center(len / 2., len / 2.);
    std::cout << "transform center : " << center.x << ", " << center.y << std::endl;
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);

    cv::warpAffine(src, dst, rot_mat, dst_sz, cv::INTER_LINEAR, cv::BORDER_DEFAULT);

    //定义平移矩阵
    cv::Mat t_mat =cv::Mat::zeros(2, 3, CV_32FC1);

    t_mat.at<float>(0, 0) = 1;
    t_mat.at<float>(0, 2) = delta_x; //水平平移量
    t_mat.at<float>(1, 1) = 1;
    t_mat.at<float>(1, 2) = delta_y; //竖直平移量

    cv::warpAffine(dst, dst, t_mat, dst_sz,cv::INTER_LINEAR, cv::BORDER_DEFAULT);

    return dst;
}

int main(){
    cv::Mat image_source = cv::imread(image_path);

    if(image_source.data == nullptr){
        std::cout << image_path << "不存在。";
        return 0;
    }

    cv::Mat image_target = transformImage(image_source, 0.0F, 20, 5);

    cv::Mat im_target_gray, im_source_gray;
    cv::cvtColor(image_source, im_source_gray, CV_BGR2GRAY);
    cv::cvtColor(image_target, im_target_gray, CV_BGR2GRAY);

    cv::imshow("image_source", im_source_gray);
    cv::imshow("image_target", im_target_gray);
    cv::waitKey(0);

    im_target_gray.convertTo(im_target_gray, CV_32F, 1 / 255.0);
    im_source_gray.convertTo(im_source_gray, CV_32F, 1 / 255.0);

    cv::Mat result;
    LucasKanada lucas_kanada;

    lucas_kanada.align(im_target_gray, im_source_gray, result);


    return 0;

}
