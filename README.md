# HDR-Render    

As the name suggests, this project aims to render
High Dynamic Range Image using exposure bracketing. The program implements a high level abstraction of pre-existing algorithms to achieve this. Initial prototype of this program was written in ```Python``` but due to the computation heavy nature of the processing pipeline, most of the code written now is in ```C++``` and some parts in ```C``` which increases the performance in computation heavy workloads. 

## Dependencies & Installation

* [```OpenCV ```](https://opencv.org/) - FOSS Image processing and computer vision framework
* [```CMake```](https://cmake.org/) - FOSS and cross platform software to build, test, automate and package code  
* [```easyexif```](https://github.com/mayanklahiri/easyexif) - Tiny ISO-compliant C++ EXIF parsing library, third-party dependency free.

### For Linux: 
It is advised to compile OpenCV from source. The steps to 
produce the same are:

### 1. Install Build Dependencies:

``` 
 Debian/Ubuntu : sudo apt-get -y install gcc g++ cmake
 Arch          : sudo pacman -S gcc g++ cmake
 Fedora/RedHat : sudo dnf install gcc g++ cmake
 ```

Additionally, to get **GTK** support for GUI features in Ubuntu/Debain based distros, follow the bellow commands
```
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev

sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
```
Note: Most of the build dependencies come pre-shipped with most modern distributions

### 2. Cloning the OpenCV Repository:
* Go to [OpenCV releases](https://opencv.org/releases/)
* Select the most recent stable version and go to it's Github repo
* Clone the repository by using ```git clone```

### 3. Generating Build Files & Building OpenCV:
* After cloning the repository, create a folder, name it  _build_. Change Directory to the _build_ folder. From the build folder execute
``` cmake ../opencv ``` which will trigger build script generation, these scripts are required for the building process.
* Once build script generation is complete. From the _build_ folder itself, execute ```make -j4```. This will trigger the compilation and building processing of all the headers and modules of opencv. This step will take some time and will depend upon your processor.

### 4. Installing OpenCV:
* After the completion of the build, all that is left to do is to install. To install simply execute ``` sudo make install ```. If everything goes well then opencv files will be installed to path ```usr/local/include/opencv4```

### 5. Cloning this repo
* To clone this repository, run: 
> ```git clone https://github.com/rd3ka/HDR-Render.git ```
>

### 6. Generate Build Files
* Change Directory into the folder of this repository. 
> Run ```cmake -S . -B build```, keeping in account that you already have created a folder named _build_. This will generate build scripts.
* Execute ```cmake --build build``` to build the program
### 7. Run
* To execute, Run
> ```./build/main path/to/folder/having/pictures```