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
#include "benchmark.h"
#include "cpu.h"
using namespace std;

class DiffusionSlover
{
public:
    int load(AAssetManager* mgr, std::string bin);

    ncnn::Mat sampler(int seed, int step, ncnn::Mat& c, ncnn::Mat& uc);

private:
    ncnn::Mat randn_4_64_64(int seed);
    ncnn::Mat CFGDenoiser_CompVisDenoiser(ncnn::Mat& input, float sigma, ncnn::Mat cond, ncnn::Mat uncond);

private:
    float log_sigmas[1000] = { 0 };

    ncnn::Net net;
};