#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <filesystem>
#include <vector>
#include <queue>
#include <future>
#include "lib/EasyExif/exif.h"
using namespace std;
vector <cv::Mat>* imglist;
vector <float>* eTL;

static mutex m, p;

static const float easyEXIFshow(string path) {
    FILE *filePointer = fopen(path.c_str(), "rb");
    if (!filePointer) { 
        printf("Cannot open file! Abort\n"); 
        return -1; 
    }
    
    fseek(filePointer, 0, SEEK_END);
    unsigned long fsize = ftell(filePointer);
    rewind(filePointer);
    unsigned char *buffer = new unsigned char[fsize];
    if (fread(buffer, 1, fsize, filePointer) != fsize) {
        printf("Cannot read file! Abort\n");
        delete []buffer;
        return -2;
    }
    fclose(filePointer);

    easyexif::EXIFInfo exif;
    int code = exif.parseFrom(buffer, fsize);
    delete []buffer; 

    if (code) { 
        printf("Error parsing exif info! Abort\n"); 
        return -3;
    }
    return exif.ExposureTime;
}

static const void imgRead(string path) {
    printf("Reading image and information..\n");
    imglist = new vector <cv::Mat>;

    eTL = new vector <float>;
    for(const auto & entry : filesystem::directory_iterator(path)) {
        imglist->emplace_back(cv::imread(entry.path()));
        eTL->emplace_back(easyEXIFshow(entry.path()));
    }
    return;
}

static const void alignImages() {
    printf("Aligning images using Median Threshold Boundary Algorithm..\n");
    cv::Ptr<cv::AlignMTB> aMTB = cv::createAlignMTB();
    aMTB->process(*imglist, *imglist);
    return;
}

static const void convert(cv::Mat& img, int mode = 2) {
    vector <int> compression_params;
    compression_params.emplace_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.emplace_back(100);

    img.convertTo(img, CV_8UC3, 255.0);

    if (mode == 2)
        cv::imwrite("out.jpeg", img, compression_params);
    else
        printf("image converted!\n");
}

static const cv::Mat caliberateCRF() {
    cv::Mat response;
    cv::Ptr <cv::CalibrateDebevec> cD = cv::createCalibrateDebevec();
    printf("Caliberating Camera Response Function to be Linear...\n");
    cD->process(*imglist, response, *eTL); 
    return response;
}

static const cv::Mat mergeFrameHDR(cv::Mat cCRF) {
    cv::Mat hiDepthRange;
    cv::Ptr <cv::MergeDebevec> mD = cv::createMergeDebevec();
    printf("Processing High Dynamic Range rendering...\n");
    mD->process(*imglist, hiDepthRange, *eTL, cCRF);
    return hiDepthRange;
}

static const cv::Mat toneMap(int mode = 2) {
    cv::Mat tm;
    switch(mode) {

        case 1 : {
            printf("Tonemapping the HDR Image using Reinhard's Algorithm...\n");
            cv::Ptr <cv::TonemapReinhard> tR = cv::createTonemapReinhard();
            tR->process(mergeFrameHDR(caliberateCRF()), tm);
            break;
        }
        
        case 2 : {
            printf("Tonemapping the HDR Image using Drago's Algorithm...\n");
            cv::Ptr <cv::TonemapDrago> tD = cv::createTonemapDrago();
            tD->process(mergeFrameHDR(caliberateCRF()), tm);
            break;
        }

        case 3 : {
            printf("Tonemapping the HDR Image using Mantiuk's Algorithm...\n");
            cv::Ptr <cv::TonemapMantiuk> tM = cv::createTonemapMantiuk();
            tM->process(mergeFrameHDR(caliberateCRF()), tm);
            break;
        }
        default: { toneMap(2); }
    }
    convert(tm,1);
    return tm;
}

static const cv::Mat exposureFusion() {
    cv::Mat eFI;
    cv::Ptr<cv::MergeMertens> mM = cv::createMergeMertens();
    mM->process(*imglist, eFI);
    delete eTL;
    return eFI;
}

static const void multiCompute() {
    /* function should not break? */
    vector <cv::Mat>* ppline = new vector <cv::Mat>; 
    cv::Mat hstack;

    vector <future <const cv::Mat>> fo;
    for(int i = 0; i < 3; i++)
        fo.emplace_back(async(toneMap, i + 1));

    for(auto& f : fo)  
        ppline->emplace_back(f.get());

    cv::Ptr <cv::MergeMertens> merge = cv::createMergeMertens();
    merge->process(*ppline, hstack);
    convert(hstack);
}

static const void showImgList() {
    for(int i = 0; i < static_cast <int> (imglist->size()); i++) {
        cv::namedWindow("Display Image", cv::WINDOW_NORMAL);
        cv::resizeWindow("Display Image", 600, 400);
        cv::imshow("Display Image", imglist[i]);
        printf("ExposureTime : %f s\n",eTL->at(i));
        cv::waitKey(0);
    }
    return;
}

static const void sysInfo() {
    vector <cv::ocl::PlatformInfo> pf;
    cv::ocl::getPlatfomsInfo(pf);

    cv::ocl::setUseOpenCL(true);
    for(size_t i = 0; i < pf.size(); i++) {
        const cv::ocl::PlatformInfo* platform = &pf[i];
        printf("Platform Name: %s\n", platform->name().c_str());
        cv::ocl::Device current_device;
        for (int j = 0; j < platform->deviceNumber(); j++) {
            platform->getDevice(current_device, j);
            printf("Device Name: %s\n",current_device.name().c_str());
            printf("Device Number: %d\n", platform->deviceNumber());
            printf("Device Type: %d\n", current_device.type());
            printf("Device Driver Version %s\n",current_device.driverVersion().c_str());
        }
    }
}

int main(int argc, char** argv) {
    sysInfo();
    imgRead(argv[1]);
    alignImages();
    multiCompute();
    return 0;
}
