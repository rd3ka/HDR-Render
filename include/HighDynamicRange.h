#include "imageHandler.h"
#include <future>
#include <opencv2/opencv.hpp>
#include <vector>

#define _MAX_TONEMAP 3

class HighDynamicRange {

public:
    std::vector<cv::Mat> imageList;
    std::vector<float> exposureTimeList;
    HighDynamicRange(std::string);
    HighDynamicRange(std::vector<cv::Mat> imageList,
                     std::vector<float> exposureTimeList);
    const void alignImages();
    const cv::Mat caliberateCRF();
    const cv::Mat mergeFrameHDR(cv::Mat CRFresp);
    const cv::Mat toneMap(cv::Mat hdr, int mode);
    const cv::Mat exposureFusion(std::vector<cv::Mat> im);
    const std::vector<cv::Mat> computeTonemap(cv::Mat HDR);
};
