#ifndef YOLOV4_H
#define YOLOV4_H

#include "ortobjectdetectionmodel.h"
#include <onnxruntime_cxx_api.h>

typedef struct _BoundingBox {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int class_index;

    _BoundingBox(float xmin, float ymin, float xmax, float ymax, float score, int class_index) : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax), score(score), class_index(class_index) {}

} BoundingBox;

class YOLOv4 : OrtObjectDetectionModel {
    private:
        const int NUM_CLASSES = 80;
        const int INPUT_HEIGHT = 416;
        const int INPUT_WIDTH = 416;
        const int INPUT_CHANNELS = 3;

        int org_img_w;
        int org_img_h;
        cv::Mat org_img;

        float resize_ratio;

        cv::Mat preprocessed_img;

        bool check_init;

        int num_classes_detected;

        size_t input_tensor_size; // SET IN INIT OR PREPROCESS

        size_t num_input_nodes;
        std::vector<const char*> input_node_names;

        size_t num_output_nodes;
        std::vector<const char*> output_node_names;

        float *model_input;

        std::vector<Ort::Value> model_output;
        std::vector<std::string> labels;
        
        Ort::SessionOptions session_options;
        Ort::AllocatorWithDefaultOptions allocator;
        
        std::unique_ptr<Ort::Session> session;
        std::unique_ptr<Ort::Env> env;

        void setOrtEnv();
        void setModelInputOutput();
        void createSession(std::string onnx_model_path); // TODO add optimization level and exeuction provider

        cv::Mat padImage(cv::Mat image, int target_height, int target_width);

        std::vector<BoundingBox*> getBoundingBoxes(std::vector<float> anchors, std::vector<float> strides, std::vector<float> xyscale, float threshold);
        float bbox_iou(BoundingBox *bbox1, BoundingBox *bbox2);
        std::vector<BoundingBox*> nms(std::vector<BoundingBox*> bboxes, float threshold);

        void writeBoundingBoxes(std::vector<BoundingBox*> bboxes, std::vector<std::string> class_names, cv::Mat image);


    public:
        YOLOv4();
        virtual ~YOLOv4();
        bool init(std::string onnx_model_path);
        void preprocessImage(cv::Mat image, int target_height, int target_width);
        void runModel();
        void postprocess();





};

#endif