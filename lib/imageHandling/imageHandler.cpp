#include "imageHandler.h"
#include "easyEXIF.h"

imageHandler::imageHandler() {}

const void imageHandler::imgRead(std::string path,
                                 std::vector<cv::Mat> &imageList,
                                 std::vector<float> &exposureTimeList) {
    std::vector<std::string> *filelist = new std::vector<std::string>;
    for (const auto &entry : std::filesystem::directory_iterator(path))
        filelist->emplace_back(entry.path());

    for (const std::string file : *filelist) {
        imageList.emplace_back(cv::imread(file));
        exposureTimeList.emplace_back(easyEXIF(file).getExposureTime());
    }
    return;
}

const cv::Mat imageHandler::imgConvert(cv::Mat img) {
    img.convertTo(img, CV_8U, 255);
    return img;
}

const void imageHandler::imgWrite(cv::Mat img, std::string filename) {
    std::vector<int> *compression_params =
        new std::vector<int> {cv::IMWRITE_JPEG_QUALITY, 100};
    cv::imwrite(filename, img, *compression_params);
    delete compression_params;
}

const void imageHandler::imgShow(std::vector<cv::Mat> imageList,
                                 std::vector<float> exposureTimeList) {
    for (int i = 0; i < static_cast<int>(imageList.size()); i++) {
        cv::namedWindow("Display Image", cv::WINDOW_NORMAL);
        cv::resizeWindow("Display Image", 600, 400);
        cv::imshow("Display Image", imageList[i]);
        printf("ExposureTime : %f s\n", exposureTimeList.at(i));
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    return;
}
