#include <fstream>
#include "yolov4.h"

YOLOv4::YOLOv4() {}

// TODO: fix memory leaks
YOLOv4::~YOLOv4() {
    if (check_init) {
        for (const char *node_name : input_node_names)
            allocator.Free(const_cast<void*>(reinterpret_cast<const void*>(node_name)));

        for (const char *node_name : output_node_names)
            allocator.Free(const_cast<void*>(reinterpret_cast<const void*>(node_name)));

        for (Ort::Value &layer_output : model_output) {
            layer_output.release();
        }

        session.release();
        env.release();
    }
}

/**
 * @brief Set up ONNX Runtime environment and set session options (e.g. graph optimaztion level).
 */
void YOLOv4::setOrtEnv() {
    env = std::make_unique<Ort::Env>(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test"));

    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

/**
 * @brief Parses ONNX model input/output information.
 */
void YOLOv4::setModelInputOutput() {
    num_input_nodes = session->GetInputCount();
    input_node_names = std::vector<const char*>(num_input_nodes);

    num_output_nodes = session->GetOutputCount();
    output_node_names = std::vector<const char*>(num_output_nodes);
}

/**
 * @brief Creates ONNX Runtime session from existing environment.
 * Uses environment and session options set up in `setOrtEnv`.
 * 
 * @param onnx_model_path path to YOLOv4 ONNX model file to use for inferencing.
 */
void YOLOv4::createSession(std::string onnx_model_path) {
    session = std::make_unique<Ort::Session>(Ort::Session(*env, onnx_model_path.c_str(), session_options));
}

/**
 * @brief Initializes YOLOv4 for object detection. 
 * Creates ONNX Runtime environment, session, parses input/output, etc.
 * 
 * @param onnx_model_path path to YOLOv4 ONNX model file to use for inferencing.
 * @return true if setup was successful.
 * @return false if setup failed.
 */
bool YOLOv4::init(std::string onnx_model_path) {
    setOrtEnv();
    createSession(onnx_model_path);
    setModelInputOutput();
    check_init = true;
    return true;
}

/**
 * @brief Runs YOLOv4 on prepared input image.
 * Internally sets up model output.
 */
void YOLOv4::runModel() {
    std::vector<int64_t> input_node_dims;
    std::vector<int64_t> output_node_dims;

    printf("Number of inputs = %zu\n", num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; i++) {
        char *input_name = session->GetInputName(i, allocator);
        input_node_names[i] = input_name;
        //printf("Input %zu : name=%s\n", i, input_name);

        Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        //printf("Input %zu : type=%d\n", i, type);

        input_node_dims = tensor_info.GetShape();
        //printf("Input %zu : num_dims=%zu\n", i, input_node_dims.size());
    }

    // HOTFIX
    input_node_dims[0] = 1;

    for (size_t i = 0; i < num_output_nodes; i++) {
        char *output_name = session->GetOutputName(i, allocator);
        output_node_names[i] = output_name;
        //printf("Output %zu : name=%s\n", i, output_name);

        Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        //printf("Output %zu : type=%d\n", i, type);

        output_node_dims = tensor_info.GetShape();
        //printf("Output %zu : num_dims=%zu\n", i, output_node_dims.size());
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    ///////////////////////////////////////
    // GET INPUT TENSOR HERE
    input_tensor_size = INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS;
    std::vector<float> input_tensor_values;
    input_tensor_values.assign(preprocessed_img.data, preprocessed_img.data + preprocessed_img.total() * preprocessed_img.channels());
    for (size_t i = 0; i < input_tensor_values.size(); i++) {
        input_tensor_values[i] = input_tensor_values[i] / 255.f;
    }

    int i1 = 300 * 416 *3 + 300*3;
    int i2 = 245 * 416 + 324*3 + 2;
    std::cout << input_tensor_values[i1] << std::endl;    
    std::cout << input_tensor_values[i1+1] << std::endl;
    std::cout << input_tensor_values[i1+2] << std::endl;
    std::cout << input_tensor_values[i2] << std::endl;


    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
    assert(input_tensor.IsTensor());
    ///////////////////////////////////////

    model_output = session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, num_input_nodes, output_node_names.data(), num_output_nodes);
}

/**
 * @brief Pads inputted image to YOLOv4 input specifications.
 * Preserves aspect ratio. Pad with grey (128, 128, 128) pixels.
 * 
 * @param image image to pad.
 * @param target_height input image height for algorithm.
 * @param target_width input image width for algorithm.
 * @return cv::Mat padded image.
 */
cv::Mat YOLOv4::padImage(cv::Mat image, int target_height, int target_width) {
    int h = image.rows;
    int w = image.cols;

    resize_ratio = std::min(target_width / (w * 1.0f), target_height / (h * 1.0f));
    // New dimensions to preserve aspect ratio
    int nw = resize_ratio * w;
    int nh = resize_ratio * h;

    // Padding on either side
    float dw = std::floor((target_width - nw) / 2.0f);
    float dh = std::floor((target_height - nh) / 2.0f);

    cv::Mat out(target_height, target_width, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(nw, nh));
    resized.copyTo(out(cv::Rect(dw, dh, resized.cols, resized.rows)));
    return out;
}

/**
 * @brief Preprocesses input image to comply with input specifications of YOLOv4 algorithm.
 * Internally sets preprocessed image.
 * 
 * @param image image to process in cv::Mat format.
 * @param target_height input height for YOLOv4 algorithm.
 * @param target_width input width for YOLOv4 algorithm.
 */
void YOLOv4::preprocessImage(cv::Mat image, int target_height, int target_width) {
    // Set original image data
    org_img = image;
    org_img_h = image.rows;
    org_img_w = image.cols;
    
    // Pad image
    cv::Mat padded = padImage(image.clone(), target_height, target_width);
    // imread by default gives image with channels in BGR order
    // Swap to RGB order
    cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);
    cv::Mat image_float;
    // Convert data to floats
    padded.convertTo(image_float, CV_32FC3, 255.f);

    // Image is already in HWC format from imread function

    // Set preprocessed image
    preprocessed_img = padded;
}

// Apply sigmoid function to a value; returns a number between 0 and 1
float sigmoid(float value) {
    float k = (float) exp(-1.0f * value);
    return 1.0f / (1.0f + k);
}

/**
 * @brief Parses model output to extract bounding boxes. Filters bounding boxes and converts coordinates
 * to be respective to original image.
 * 
 * @param anchors vector of YOLOv4 anchors value (for each anchor index of each layer).
 * @param strides vector of strides for each output layer.
 * @param xyscale vector of xyscale for each output layer.
 * @param threshold threshold to filter boxes based on confidence/score.
 * @return std::vector<BoundingBox*> filtered bounding boxes from model output (all layers).
 */
std::vector<BoundingBox*> YOLOv4::getBoundingBoxes(std::vector<float> anchors, std::vector<float> strides, std::vector<float> xyscale, float threshold) {
    std::vector<BoundingBox*> bboxes;
    auto dw = (INPUT_WIDTH - resize_ratio * org_img_w) / 2;
    auto dh = (INPUT_HEIGHT - resize_ratio * org_img_h) / 2;

    // Iterate through output layers
    for (size_t layer = 0; layer < num_output_nodes; layer++) {
        float *layer_output = model_output[layer].GetTensorMutableData<float>();
        auto layer_shape = model_output[layer].GetTensorTypeAndShapeInfo().GetShape();
        
        // Layer data
        auto grid_size = layer_shape[1];
        auto anchors_per_cell = layer_shape[3];
        auto features_per_anchor = layer_shape[4];

        // Iterate through grid cells in current layer, and anchors in each grid cell
        for (auto row = 0; row < grid_size; row++) {
            for (auto col = 0; col < grid_size; col++) {
                for (auto anchor = 0; anchor < anchors_per_cell; anchor++) {
                    // Calculate offset for current grid cell and anchor
                    auto offset = (row * grid_size * anchors_per_cell * features_per_anchor) + (col * anchors_per_cell * features_per_anchor) + (anchor * features_per_anchor);
                    // Extract data
                    auto x = layer_output[offset + 0];
                    auto y = layer_output[offset + 1];
                    auto h = layer_output[offset + 3]; 
                    auto w = layer_output[offset + 2];
                    auto conf = layer_output[offset + 4];

                    if (conf < threshold) {
                        //continue;
                    }

                    // Transform coordinates
                    x = ((sigmoid(x) * xyscale[layer]) - 0.5f * (xyscale[layer] - 1.0f) + col) * strides[layer];
                    y = ((sigmoid(y) * xyscale[layer]) - 0.5f * (xyscale[layer] - 1.0f) + row) * strides[layer];
                    h = exp(h) * anchors[(layer * 6) + (anchor * 2) + 1]; 
                    w = exp(w) * anchors[(layer * 6) + (anchor * 2)];   
                    // Convert (x, y, w, h) => (xmin, ymin, xmax, ymax)
                    auto xmin = x - w * 0.5f;
                    auto ymin = y - h * 0.5f;
                    auto xmax = x + w * 0.5f;
                    auto ymax = y + h * 0.5f;
                    // Convert (xmin, ymin, xmax, ymax) => (xmin_org, ymin_org, xmax_org, ymax_org), relative to original image
                    auto xmin_org = 1.0f * (xmin - dw) / resize_ratio;
                    auto ymin_org = 1.0f * (ymin - dh) / resize_ratio;
                    auto xmax_org = 1.0f * (xmax - dw) / resize_ratio;
                    auto ymax_org = 1.0f * (ymax - dh) / resize_ratio;

                    // Disregard clipped boxes
                    if (xmin_org > xmax_org || ymin_org > ymax_org) {
                        continue;
                    }
                    // Disregard boxes with invalid size/area
                    auto area = (xmax_org - xmin_org) * (ymax_org - ymin_org);
                    if (area <= 0 || isnan(area) || !isfinite(area)) {
                        continue;
                    }

                    // Find class with highest probability
                    int max_class = -1;
                    float max_prob;
                    for (int i = 0; i < NUM_CLASSES; i++) {
                        if (max_class == -1 || layer_output[offset + 5 + i] > max_prob) {
                            max_class = i;
                            max_prob = layer_output[offset + 5 + i];
                        }
                    }
                    // Calculate score and compare against threshold
                    float score = conf * max_prob;
                    if (score <= threshold) {
                        continue;
                    }
                    // Create bounding box and add to vector
                    BoundingBox *bbox = new BoundingBox(xmin_org, ymin_org, xmax_org, ymax_org, score, max_class);
                    bboxes.push_back(bbox);
                }
            }
        }
    }
    return bboxes;
}

/**
 * @brief Calculate the intersection over union (IOU) of two bounding boxes.
 * 
 * @param bbox1 first bounding box.
 * @param bbox2 second bounding box.
 * @return float IOU of the two boxes.
 */
float YOLOv4::bbox_iou(BoundingBox *bbox1, BoundingBox *bbox2) {
    float area1 = (bbox1->xmax - bbox1->xmin) * (bbox1->ymax - bbox1->ymin);
    float area2 = (bbox2->xmax - bbox2->xmin) * (bbox2->ymax - bbox2->ymin);

    float left = std::max(bbox1->xmin, bbox2->xmin);
    float right = std::min(bbox1->xmax, bbox2->xmax);
    float top = std::max(bbox1->ymin, bbox2->ymin);
    float bottom = std::min(bbox1->ymax, bbox2->ymax);

    float intersection_area;
    if (left > right || top > bottom) {
        intersection_area = 0;
    } else {
        intersection_area = (right - left) * (bottom - top);
    }
    float union_area = area1 + area2 - intersection_area;
    return intersection_area / union_area;
}

// Compares score of two bounding boxes. Used to sort vectors of bounding boxes.
bool compareBoxScore(BoundingBox* b1, BoundingBox* b2) {
    return b2->score < b1->score;
}

/**
 * @brief Perform non-maximal suppression (nms) on vector of bounding boxes.
 * 
 * @param bboxes vector of bounding boxes to perform nms on.
 * @param threshold IOU threshold for nms.
 * @return std::vector<BoundingBox*> filtered boxes after applying nms.
 */
std::vector<BoundingBox*> YOLOv4::nms(std::vector<BoundingBox*> bboxes, float threshold) {
    // Organize boxes by class 
    std::unordered_map<int, std::vector<BoundingBox*>> class_map;
    for (size_t i = 0; i < bboxes.size(); i++) {
        class_map[bboxes[i]->class_index].push_back(bboxes[i]);        
    }
    num_classes_detected = class_map.size();

    std::vector<BoundingBox*> filtered_boxes;

    // Iterate through each class detected
    for (auto &pair : class_map) {
        std::vector<BoundingBox*> boxes = pair.second;
        // Sort class specific boxes by score in decreasing order 
        std::sort(boxes.begin(), boxes.end(), compareBoxScore);

        while (boxes.size() > 0) {
            // Extract box with highest score
            BoundingBox *accepted_box = boxes[0];
            filtered_boxes.push_back(accepted_box);
            boxes.erase(boxes.begin());

            std::vector<BoundingBox*> safe_boxes;
            // Compare extracted box with all remaining class boxes
            for (size_t i = 0; i < boxes.size(); i++) {
                BoundingBox *test_box = boxes[i];
                if (bbox_iou(accepted_box, test_box) <= threshold) {
                    safe_boxes.push_back(test_box);
                } else {
                    delete test_box;
                }
            }
            // Update class boxes
            boxes = safe_boxes;
        }
    }
    return filtered_boxes;
}

/**
 * @brief Write bounding boxes and class labels/scores to inputted image.
 * 
 * @param bboxes filtered bounding boxes.
 * @param class_names vector of class names.
 * @param image image to write to.
 */
void YOLOv4::writeBoundingBoxes(std::vector<BoundingBox*> bboxes, std::vector<std::string> class_names, cv::Mat image) {
    float font_scale = 0.5f;
    int bbox_thick = (int) (0.6f * (org_img_h + org_img_w) / 600.f);
    std::srand(5);

    std::unordered_map<int, cv::Scalar> class_colors;

    for (size_t i = 0; i < bboxes.size(); i++) {
        // Bounding box information
        BoundingBox *bbox = bboxes[i];
        std::string class_name = class_names[bbox->class_index];
        float score = bbox->score;
        auto c1 = cv::Point(bbox->xmin, bbox->ymin);
        auto c2 = cv::Point(bbox->xmax, bbox->ymax);

        // Get color for class
        cv::Scalar color;
        if (class_colors.find(bbox->class_index) != class_colors.end()) {
            color = class_colors.at(bbox->class_index);
        } else {
            color = cv::Scalar((std::rand()%256), std::rand()%256, std::rand()%256);
            class_colors[bbox->class_index] = color;
        }

        // Place rectangle around bounding box
        cv::Rect rect = cv::Rect(c1, c2);
        cv::rectangle(image, rect, color, bbox_thick);

        std::stringstream msg;
        msg << class_name << ": " << roundf(score * 100) / 100;
        int base_line = 0;
        auto t_size = cv::getTextSize(msg.str(), 0, font_scale, bbox_thick / 2, &base_line);
        // Place rectangle for class label & score message
        cv::rectangle(image, c1, cv::Point(c1.x + t_size.width, c1.y - t_size.height - 3), color, -1);
        // Place message
        cv::putText(image, msg.str(), cv::Point(bbox->xmin, bbox->ymin - 2), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 0, 0), bbox_thick / 2);

        std::cout << "Found " << class_name << " at (" << bbox->xmin << ", " << bbox->ymin << ") with score " << score << std::endl;

        delete bbox;
    }
    std::cout << "Writing image..." << std::endl;
    cv::imwrite("../assets/images/output1.png", image);
}

/**
 * @brief 
 * 
 * @param filename 
 * @return std::vector<std::string> 
 */
std::vector<std::string> fetchClassNames(const std::string filename) {
    std::vector<std::string> class_names(80);
    std::ifstream input(filename);
    std::string line;
    
    for (int i = 0; i < 80; i++) {
        if (!getline(input, line)) {
            // Malformed label file
            return class_names;
        }
        class_names[i] = line;
    }
    return class_names;
}

/**
 * @brief 
 * 
 */
void YOLOv4::postprocess() {
    std::vector<float> anchors{12.f,16.f, 19.f,36.f, 40.f,28.f, 36.f,75.f, 76.f,55.f, 72.f,146.f, 142.f,110.f, 192.f,243.f, 459.f,401.f};
    std::vector<float> strides{8, 16, 32};
    std::vector<float> xyscale{1.2, 1.1, 1.05};
    auto bboxes = getBoundingBoxes(anchors, strides, xyscale, 0.25);
    bboxes = nms(bboxes, 0.213);

    auto class_names = fetchClassNames("../assets/models/yolov4/labels.txt");
    writeBoundingBoxes(bboxes, class_names, org_img);
}