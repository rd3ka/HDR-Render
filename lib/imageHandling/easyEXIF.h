#include "exif.h"
#include <string>
#include <utility>

#pragma once
class easyEXIF {
private:
    easyexif::EXIFInfo exif;
    int code;

public:
    easyEXIF(std::string path); 
    const float getExposureTime();
    const std::pair<int, int> getResolution();
    easyEXIF(); 
};

