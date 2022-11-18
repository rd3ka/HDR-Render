#include "lib/imageHandling/easyEXIF.h"
#include "lib/imageHandling/imageHandler.h"
#include <future>
#include <opencv2/core/ocl.hpp>
using namespace std;

#define _MAX_TONEMAP 3

static const void systemInformation() {
    vector<cv::ocl::PlatformInfo> pf;
    cv::ocl::getPlatfomsInfo(pf);

    cv::ocl::setUseOpenCL(true);
    for (size_t i = 0; i < pf.size(); i++) {
        const cv::ocl::PlatformInfo *platform = &pf[i];
        std::cout << "Platform Name: " << platform->name().c_str() << std::endl;
        cv::ocl::Device current_device;
        for (int j = 0; j < platform->deviceNumber(); j++) {
            platform->getDevice(current_device, j);
            std::cout << "Device Name: " << current_device.name().c_str()
                      << std::endl;
            std::cout << "Device Number: " << platform->deviceNumber() << std::endl;
            std::cout << "Device Type: " << current_device.type() << std::endl;
            std::cout << "Device Driver Version "
                      << current_device.driverVersion().c_str() << std::endl;
        }
    }
}

class HighDynamicRange {
private:
    vector<cv::Mat> imageList;
    vector<float> exposureTimeList;

    const void alignImages() {
        printf("Aligning images using Median Threshold Boundary Algorithm..\n");
        cv::Ptr<cv::AlignMTB> aMTB = cv::createAlignMTB();
        aMTB->process(imageList, imageList);
        return;
    }

    const cv::Mat caliberateCRF() {
        cv::Mat response;
        cv::Ptr<cv::CalibrateDebevec> cD = cv::createCalibrateDebevec();
        printf("Caliberating Camera Response Function to be Linear...\n");
        cD->process(imageList, response, exposureTimeList);
        return response;
    }

    const cv::Mat mergeFrameHDR(cv::Mat CRFresp) {
        cv::Mat hiDepthRange;
        cv::Ptr<cv::MergeDebevec> mD = cv::createMergeDebevec();
        printf("Processing High Dynamic Range rendering...\n");
        mD->process(this->imageList, hiDepthRange, this->exposureTimeList, CRFresp);
        return hiDepthRange;
    }

    static const cv::Mat toneMap(cv::Mat hdr, int mode) {
        cv::Mat tm;
        switch (mode) {

        case 1: {
            printf("Tonemapping the HDR Image using Reinhard's Algorithm...\n");
            cv::Ptr<cv::TonemapReinhard> tR =
                cv::createTonemapReinhard(1.5, 0.7, 0, 0);
            tR->process(hdr, tm);
            break;
        }

        case 2: {
            printf("Tonemapping the HDR Image using Drago's Algorithm...\n");
            cv::Ptr<cv::TonemapDrago> tD = cv::createTonemapDrago(1.55, 0.80);
            tD->process(hdr, tm);
            break;
        }

        case 3: {
            printf("Tonemapping the HDR Image using Mantiuk's Algorithm...\n");
            cv::Ptr<cv::TonemapMantiuk> tM = cv::createTonemapMantiuk(2.30, 0.9, 1.5);
            tM->process(hdr, tm);
            break;
        }

        default: {
            toneMap(hdr, 2);
        }
        }
        return imageHandler().imgConvert(tm);
    }

    const cv::Mat exposureFusion(std::vector<cv::Mat> im) {
        cv::Mat exposureFusedImage;
        cv::Ptr<cv::MergeMertens> mM = cv::createMergeMertens();
        mM->process(im, exposureFusedImage);
        return imageHandler().imgConvert(exposureFusedImage);
    }

    const vector<cv::Mat> computeTonemap(cv::Mat HDR) {
        vector<cv::Mat> ppline;

        vector<future<const cv::Mat>> futureObject;
        for (int i = 1; i <= _MAX_TONEMAP; i++)
            futureObject.emplace_back(async(this->toneMap, HDR, i));

        for (auto &obj : futureObject)
            ppline.emplace_back(obj.get());

        return ppline;
    }

public:
    HighDynamicRange(vector<cv::Mat> imageList, vector<float> exposureTimeList) {
        this->imageList = imageList;
        this->exposureTimeList = exposureTimeList;
    }

    HighDynamicRange(std::string path) {
        imageHandler().imgRead(path, this->imageList, this->exposureTimeList);
    }

    const void process() {
        cv::Mat finalImage;

        vector<cv::Mat> ppline;
        cv::Mat HighDynamicRange = mergeFrameHDR(caliberateCRF());

        for (auto e : computeTonemap(HighDynamicRange))
            ppline.emplace_back(e);
        ppline.emplace_back(exposureFusion(this->imageList));

        finalImage = exposureFusion(ppline);
        imageHandler().imgWrite(finalImage);
    }
};

int main(int argc, char **argv) {
    systemInformation();
    HighDynamicRange HDR = HighDynamicRange(argv[1]);
    HDR.process();
    return 0;
}
