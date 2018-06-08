#ifndef MXNET_PREDICTOR_H
#define MXNET_PREDICTOR_H

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <c_predict_api.h>


class mxnet_predictor{
public:
    mxnet_predictor(std::string json_config, std::string model);
    cv::Mat extract(const cv::Mat &image);
    ~mxnet_predictor();
private:
    void prepare_image(const cv::Mat &image, mx_float* image_data) const;
    std::vector<char> json_data;
    std::vector<char> param_data;
    int dev_type = 1;
    int dev_id = 0;
    mx_uint num_input_nodes = 1;
    const char* input_key[1] = {"data"};
    const char** input_keys = input_key;
    int width = 224;
    int height = 224;
    int channels = 3;
    const char* output_key[1] = {"global_pool"};
    PredictorHandle out;
};
#endif // MXNET_PREDICTOR_H
