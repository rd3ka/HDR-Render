#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#pragma once

class imageHandler {

public:
    imageHandler();

    const void imgRead(std::string path, std::vector<cv::Mat> &imageList,
                       std::vector<float> &exposureTimeList);
    const cv::Mat imgConvert(cv::Mat img);
    const void imgWrite(cv::Mat img, std::string filename = "out.jpeg");
    const void imgShow(std::vector <cv::Mat> imageList, std::vector <float> exposureList);
};
