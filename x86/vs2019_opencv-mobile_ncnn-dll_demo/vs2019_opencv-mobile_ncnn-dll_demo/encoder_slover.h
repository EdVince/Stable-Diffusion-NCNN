#pragma once
#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <stack>
#include <fstream>
#include <map>
#include <math.h>
#include <net.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <time.h>
using namespace std;

class EncodeSlover
{
public:
    EncodeSlover(int h, int w);

    std::vector<ncnn::Mat> encode(cv::Mat& image);

private:
    void generate_param(int height, int width);

    const float _mean_[3] = { 127.5f, 127.5f, 127.5f };
    const float _norm_[3] = { 1.0 / 127.5f, 1.0 / 127.5f, 1.0 / 127.5f };

    ncnn::Net net;

    int h_size, w_size;
};