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

    ncnn::Mat sampler_txt2img(int seed, int step, ncnn::Mat& c, ncnn::Mat& uc);
    ncnn::Mat sampler_img2img(int seed, int step, ncnn::Mat& c, ncnn::Mat& uc, vector<ncnn::Mat>& init);

private:
    ncnn::Mat randn_4_32_32(int seed);
    ncnn::Mat CFGDenoiser_CompVisDenoiser(ncnn::Mat& input, float sigma, ncnn::Mat cond, ncnn::Mat uncond);

private:
    float log_sigmas[1000] = { 0 };
    const float guidance_scale = 7.5;
    const float strength = 0.75;

    const float factor[4] = { 0.18215f, 0.18215f, 0.18215f, 0.18215f };

    ncnn::Net net;
};