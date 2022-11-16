#include "lib/EasyExif/exif.h"
#include <algorithm>
#include <filesystem>
#include <future>
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;

#define _MAX_TONEMAP 3

static vector<cv::Mat> imageList;
static vector<float> exposureTimeList;
static future <const cv::Mat> futureExpFuse;

class easyEXIF {
private:
    easyexif::EXIFInfo exif;
    int code;

public:
    easyEXIF(string path) {
        FILE *filePointer = fopen(path.c_str(), "rb");
        if (!filePointer) {
            printf("Cannot open file! Abort\n");
            return;
        }

        fseek(filePointer, 0, SEEK_END);
        unsigned long fsize = ftell(filePointer);
        rewind(filePointer);
        unsigned char *buffer = new unsigned char[fsize];
        if (fread(buffer, 1, fsize, filePointer) != fsize) {
            printf("Cannot read file! Abort\n");
            delete[] buffer;
            return;
        }
        fclose(filePointer);
        this->code = this->exif.parseFrom(buffer, fsize);
        delete[] buffer;
    }

    const float getExposureTime() {
        return this->exif.ExposureTime;
    }
    const pair<int, int> getResolution() {
        return make_pair(this->exif.ImageWidth, this->exif.ImageHeight);
    }
    ~easyEXIF() {}
};

static const cv::Mat convert(cv::Mat img) {
    img.convertTo(img, CV_8U, 255.0);
    return img;
}

static const void save(cv::Mat img, string filename = "out.jpeg") {
    vector<int> *compression_params =
        new vector<int> {cv::IMWRITE_JPEG_QUALITY, 100};
    cv::imwrite(filename, img, *compression_params);
    delete compression_params;
}

static const void imgRead(string path) {
    std::cout << "Reading Image & Image Metadata" << std::endl;
    vector<string> *filelist = new vector<string>;
    for (const auto &entry : filesystem::directory_iterator(path))
        filelist->emplace_back(entry.path());

    sort(filelist->begin(), filelist->end());

    for (const string file : *filelist) {
        imageList.emplace_back(cv::imread(file));
        exposureTimeList.emplace_back(easyEXIF(file).getExposureTime());
    }
    return;
}

static const void alignImages() {
    printf("Aligning images using Median Threshold Boundary Algorithm..\n");
    cv::Ptr<cv::AlignMTB> aMTB = cv::createAlignMTB();
    aMTB->process(imageList, imageList);
    return;
}

static const cv::Mat caliberateCRF() {
    cv::Mat response;
    cv::Ptr<cv::CalibrateDebevec> cD = cv::createCalibrateDebevec();
    printf("Caliberating Camera Response Function to be Linear...\n");
    cD->process(imageList, response, exposureTimeList);
    return response;
}

static const cv::Mat mergeFrameHDR( cv::Mat CRFresp ) {
    cv::Mat hiDepthRange;
    cv::Ptr<cv::MergeDebevec> mD = cv::createMergeDebevec();
    printf("Processing High Dynamic Range rendering...\n");
    mD->process(imageList, hiDepthRange, exposureTimeList, CRFresp);
    return hiDepthRange;
}

static const cv::Mat toneMap(cv::Mat hdr, int mode) {
    cv::Mat tm;
    switch (mode) {

    case 1: {
        printf("Tonemapping the HDR Image using Reinhard's Algorithm...\n");
        cv::Ptr<cv::TonemapReinhard> tR = cv::createTonemapReinhard(1.5, 0.7, 0, 0);
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
    return convert(tm);
}

static const cv::Mat exposureFusion(vector <cv::Mat> im = imageList) {
    cv::Mat exposureFusedImage;
    cv::Ptr<cv::MergeMertens> mM = cv::createMergeMertens();
    mM->process(imageList, exposureFusedImage);
    return convert(exposureFusedImage);
}

static vector<cv::Mat> computeTonemap(cv::Mat HDR) {
    vector<cv::Mat> ppline;

    vector<future<const cv::Mat>> futureObject;
    for (int i = 1; i <= _MAX_TONEMAP; i++)
        futureObject.emplace_back(async(toneMap, HDR, i));
    
    for (auto &obj : futureObject)
        ppline.emplace_back(obj.get());

    return ppline;
}

static const void process() {
    futureExpFuse = async(exposureFusion,imageList);
    cv::Mat finalImage;
    vector<cv::Mat> ppline;
    cv::Mat HighDynamicRange = mergeFrameHDR(caliberateCRF());
    vector<cv::Mat> tonemappedImageList = computeTonemap(HighDynamicRange);
    ppline.insert(ppline.end(), tonemappedImageList.begin(),
                  tonemappedImageList.end());
    ppline.emplace_back(futureExpFuse.get()); 
    finalImage = exposureFusion(ppline);
    save(finalImage);
}

static const void debugFunc() {
    cv::Mat eF = exposureFusion(imageList);
    cv::Mat HighDynamicRange = mergeFrameHDR(caliberateCRF());

    save(eF, "eF.jpeg");
    save(HighDynamicRange, "hdr.jpeg");
}

static const void showImageList() {

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

static const void sysInfo() {
    vector<cv::ocl::PlatformInfo> pf;
    cv::ocl::getPlatfomsInfo(pf);

    cv::ocl::setUseOpenCL(true);
    for (size_t i = 0; i < pf.size(); i++) {
        const cv::ocl::PlatformInfo *platform = &pf[i];
        printf("Platform Name: %s\n", platform->name().c_str());
        cv::ocl::Device current_device;
        for (int j = 0; j < platform->deviceNumber(); j++) {
            platform->getDevice(current_device, j);
            printf("Device Name: %s\n", current_device.name().c_str());
            printf("Device Number: %d\n", platform->deviceNumber());
            printf("Device Type: %d\n", current_device.type());
            printf("Device Driver Version %s\n",
                   current_device.driverVersion().c_str());
        }
    }
}

int main(int argc, char **argv) {
    sysInfo();
    imgRead(argv[1]);
    alignImages();
    process();
    return 0;
}
