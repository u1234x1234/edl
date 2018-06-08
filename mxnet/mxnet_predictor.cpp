#include "mxnet_predictor.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

mxnet_predictor::mxnet_predictor(std::string json_config, std::string params)
{
	std::ifstream json_config_file(json_config, std::ios::binary);
	json_data = vector<char>((std::istreambuf_iterator<char>(json_config_file)),
								   std::istreambuf_iterator<char>());

	std::ifstream params_file(params, std::ios::binary);
	param_data = vector<char>((std::istreambuf_iterator<char>(params_file)),
								   std::istreambuf_iterator<char>());

	const mx_uint input_shape_indptr[2] = { 0, 4 };
        const mx_uint input_shape_data[4] = { 1, static_cast<mx_uint>(channels),
                                                 static_cast<mx_uint>(width),
                                                 static_cast<mx_uint>(height) };

        const char* output_key[1] = {"prob"};
	const char** output_keys = output_key;

        MXPredCreatePartialOut((const char*)json_data.data(),
                               (const char*)param_data.data(),
                               static_cast<size_t>(param_data.size()),
                               dev_type,
                               dev_id,
                               num_input_nodes,
                               input_keys,
                               input_shape_indptr,
                               input_shape_data,
                               1, output_keys,
                               &out);
}

void mxnet_predictor::prepare_image(const cv::Mat &image, mx_float* image_data) const
{
	Mat im;
	resize(image, im, Size(width, height));

	for (int i = 0; i < im.rows; i++)
	{
		for (int j = 0; j < im.cols; j++)
		{
			Vec3f pi = Vec3b(117, 117, 117);

            int idx = (i * 224 + j);
			Vec3b pix_val = im.at<Vec3b>(i, j);

			image_data[idx] = pix_val[2] - pi[2];
            image_data[idx + 224*224] = pix_val[1] - pi[1];
            image_data[idx + 224*224*2] = pix_val[0] - pi[0];
		}
	}
}

cv::Mat data(1, 1000, CV_32FC1);

cv::Mat mxnet_predictor::extract(const cv::Mat &image)
{
	int image_size = width * height * channels;
        std::vector<mx_float> image_data = std::vector<mx_float>(image_size);

        prepare_image(image, image_data.data());
	MXPredSetInput(out, "data", image_data.data(), image_size);
	MXPredForward(out);

	mx_uint output_index = 0;
        mx_uint *shape = 0;
        mx_uint shape_len;

        MXPredGetOutputShape(out, output_index, &shape, &shape_len);
        size_t size = 1000;
        for (mx_uint i = 0; i < shape_len; ++i)
            size *= shape[i];

	MXPredGetOutput(out, output_index, reinterpret_cast<float*>(data.data), size);

	return data;
}

mxnet_predictor::~mxnet_predictor()
{
	MXPredFree(out);
}
