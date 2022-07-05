#include <iostream>
#include "yolov4.h"

int main(int argc, char* argv[]) {
    YOLOv4 yolo;
    cv::Mat input_image = cv::imread("../assets/images/bike1.png");

    bool initiated = yolo.init("../assets/models/yolov4/yolov4.onnx");
    if (!initiated) {
        std::cout << "Error" << std::endl;
        return -1;
    }

    yolo.preprocessImage(input_image, 416, 416);
    yolo.runModel();
    yolo.postprocess();

    std::cout << "Done" << std::endl;
    return 0;
}