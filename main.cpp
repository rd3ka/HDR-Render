#include "HighDynamicRange.h"
#include <chrono>
#include <opencv2/core/ocl.hpp>
#include <utility>

typedef std::chrono::high_resolution_clock::time_point timeVar;

#define duration(a) std::chrono::duration_cast<std::chrono::seconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()

static const void systemInformation() {
    std::vector<cv::ocl::PlatformInfo> pf;
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
            std::cout << "Device Driver Version: "
                      << current_device.driverVersion().c_str() << std::endl;
        }
    }
}

const static void debug(std::string path) {
    HighDynamicRange h(path);

    timeVar start = timeNow();
    h.alignImages();
    timeVar end = timeNow();
    std::cout << "Aligning Images takes => " << duration(end - start) << " seconds"
              << std::endl;

    start = timeNow();
    cv::Mat CRFresp = h.caliberateCRF();
    end = timeNow();
    std::cout << "Caliberating Response Function takes => "
              << duration(end - start) << " seconds" << std::endl;

    start = timeNow();
    cv::Mat hdr = h.mergeFrameHDR(CRFresp);
    end = timeNow();
    std::cout << "Mergering Images Frame takes => " << duration(end - start)
              << " seconds" << std::endl;

    start = timeNow();
    cv::Mat ldr = h.toneMap(hdr, 1);
    end = timeNow();
    std::cout << "Tonemapping takes => " << duration(end - start) << " seconds"
              << std::endl;

    imageHandler().imgWrite(ldr, "ldr.jpeg");

    start = timeNow();
    imageHandler().imgWrite(h.exposureFusion(h.imageList), "expoFuse.jpeg");
    end = timeNow();
    std::cout << "Exposure Fusion takes => " << duration(end - start) << " seconds"
              << std::endl;
}

int main(int argc, char **argv) {
    systemInformation();
    debug(argv[1]);
    return 0;
}
