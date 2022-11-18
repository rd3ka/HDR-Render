#include "easyEXIF.h"

easyEXIF::easyEXIF(std::string path) {
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

const float easyEXIF::getExposureTime() {
    return this->exif.ExposureTime;
}

const std::pair <int,int> easyEXIF::getResolution() {
    return std::make_pair(this->exif.ImageWidth, this->exif.ImageHeight);
}
