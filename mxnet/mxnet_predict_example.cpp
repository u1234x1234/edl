#include "mxnet_predictor.h"
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    cv::Mat image = cv::imread("index.jpeg");

    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    mxnet_predictor net("squeezenet_v1.1-symbol.json", "squeezenet_v1.1-0000.params");
    cout << "init time:" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - start << "ms" << endl;

    start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    cv::Mat preds = net.extract(image);
    cout << "prediction time:" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - start << "ms" << endl;

    start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    preds = net.extract(image);
    cout << "prediction time:" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - start << "ms" << endl;

    cout << preds.at<float>(0, 0) << endl;
    cout << preds.at<float>(0, 5) << endl;

    return 0;
}
