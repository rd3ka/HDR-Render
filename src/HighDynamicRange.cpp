#include "HighDynamicRange.h"

HighDynamicRange::HighDynamicRange(std::string path) {
    imageHandler().imgRead(path, this->imageList, this->exposureTimeList);
}

HighDynamicRange::HighDynamicRange(std::vector<cv::Mat> imageList,
                                   std::vector<float> exposureTimeList) {
    this->imageList = imageList;
    this->exposureTimeList = exposureTimeList;
}

const void HighDynamicRange::alignImages() {
    cv::Ptr<cv::AlignMTB> aMTB = cv::createAlignMTB();
    aMTB->process(imageList, imageList);
    return;
}

const cv::Mat HighDynamicRange::caliberateCRF() {
    cv::Mat response;
    cv::Ptr<cv::CalibrateDebevec> cD = cv::createCalibrateDebevec();
    cD->process(imageList, response, exposureTimeList);
    return response;
}

const cv::Mat HighDynamicRange::mergeFrameHDR(cv::Mat CRFresp) {
    cv::Mat hiDepthRange;
    cv::Ptr<cv::MergeDebevec> mD = cv::createMergeDebevec();
    mD->process(imageList, hiDepthRange, exposureTimeList, CRFresp);
    return hiDepthRange;
}

const cv::Mat HighDynamicRange::toneMap(cv::Mat hdr, int mode) {
    cv::Mat tm;
    switch (mode) {
    case 1: {
        cv::Ptr<cv::TonemapReinhard> tR = cv::createTonemapReinhard(1.5, 0.7, 0, 0);
        tR->process(hdr, tm);
        break;
    }

    case 2: {
        cv::Ptr<cv::TonemapDrago> tD = cv::createTonemapDrago(1.53, 0.75);
        tD->process(hdr, tm);
        break;
    }

    case 3: {
        cv::Ptr<cv::TonemapMantiuk> tM = cv::createTonemapMantiuk(2.75, 0.85, 1.5);
        tM->process(hdr, tm);
        break;
    }
    default: {
        toneMap(hdr, 2);
    }
    }
    return imageHandler().imgConvert(tm);
}

const cv::Mat HighDynamicRange::exposureFusion(std::vector<cv::Mat> im) {
    cv::Mat exposureFusedImage;
    cv::Ptr<cv::MergeMertens> mM = cv::createMergeMertens();
    mM->process(im, exposureFusedImage);
    return imageHandler().imgConvert(exposureFusedImage);
}

const std::vector<cv::Mat> HighDynamicRange::computeTonemap(cv::Mat HDR) {
    std::vector<cv::Mat> ppline;
    for (int i = 1; i <= _MAX_TONEMAP; i++)
        ppline.emplace_back(toneMap(HDR, i));

    return ppline;
}
