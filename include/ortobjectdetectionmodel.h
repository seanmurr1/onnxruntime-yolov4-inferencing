#ifndef ORTOBJECTDETECTIONMODEL_H
#define ORTOBJECTDETECTIONMODEL_H

#include <opencv2/opencv.hpp>

class OrtObjectDetectionModel {
    public:
        virtual void preprocessImage(cv::Mat img, int target_height, int target_width) = 0;
        virtual void runModel() = 0;
        virtual void postprocess() = 0;
};

#endif
