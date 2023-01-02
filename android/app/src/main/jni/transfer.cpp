// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "transfer.h"
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

int Transfer::load(AAssetManager* mgr,  bool use_gpu)
{
    transfer_net.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    transfer_net.opt = ncnn::Option();
//    transfer_net.opt.use_bf16_storage = true;
//    transfer_net.opt.use_fp16_arithmetic = false;
//    transfer_net.opt.use_fp16_packed = false;
//    transfer_net.opt.use_fp16_storage = false;
//    transfer_net.opt.use_packing_layout = false;
//    transfer_net.opt.use_sgemm_convolution = false;
//    transfer_net.opt.use_winograd_convolution = false;
    transfer_net.opt.use_vulkan_compute = true;

    transfer_net.opt.num_threads = ncnn::get_big_cpu_count();

    int ret_param = transfer_net.load_param(mgr, "makeup.param");
    int ret_bin = transfer_net.load_model(mgr, "makeup.bin");
    if (ret_param != 0 || ret_bin != 0)
    {
        return -1;
    }
    target_size = 256;

    return 0;
}
int Transfer::transfer(cv::Mat non_makeup_img, cv::Mat makeup_img,
                       cv::Mat non_makeup_parse_img, cv::Mat makeup_parse_img,
                        cv::Mat& makeup_result, bool use_gpu)
{
    const int target_width = target_size;
    const int target_height = target_size;

    int input_width = non_makeup_img.cols;
    int input_height = non_makeup_img.rows;

    cv::resize(non_makeup_img,non_makeup_img,cv::Size(286,286),0,0,cv::INTER_LINEAR);
    cv::resize(makeup_img,makeup_img,cv::Size(286,286),0,0,cv::INTER_LINEAR);
    cv::resize(non_makeup_parse_img,non_makeup_parse_img,cv::Size(286,286),0,0,cv::INTER_NEAREST);
    cv::resize(makeup_parse_img,makeup_parse_img,cv::Size(286,286),0,0,cv::INTER_NEAREST);

    non_makeup_img = non_makeup_img(cv::Rect(15,15,256,256)).clone();
    makeup_img = makeup_img(cv::Rect(15,15,256,256)).clone();
    non_makeup_parse_img = non_makeup_parse_img(cv::Rect(15,15,256,256)).clone();
    makeup_parse_img = makeup_parse_img(cv::Rect(15,15,256,256)).clone();

    ncnn::Mat non_makeup_parse = ncnn::Mat(target_width, target_height, 18);
    ncnn::Mat makeup_parse = ncnn::Mat(target_width, target_height, 18);
    non_makeup_parse.fill(0.f);
    makeup_parse.fill(0.f);

    for (int i = 0; i < 256; i++)
    {
        for (int j = 0; j < 256; j++)
        {
            unsigned char idx = non_makeup_parse_img.at<uchar>(i, j);
            if(idx < 18)
                non_makeup_parse.channel(idx)[i * 256 + j] = 1.0;
        }
    }
    for (int i = 0; i < 256; i++)
    {
        for (int j = 0; j < 256; j++)
        {
            unsigned char idx = makeup_parse_img.at<uchar>(i, j);
            if(idx < 18)
                makeup_parse.channel(makeup_parse_img.at<uchar>(i, j))[i * 256 + j] = 1.0;
        }
    }

    ncnn::Mat non_makeup = ncnn::Mat::from_pixels(non_makeup_img.data, ncnn::Mat::PIXEL_RGB, non_makeup_img.cols, non_makeup_img.rows);
    ncnn::Mat makeup = ncnn::Mat::from_pixels(makeup_img.data, ncnn::Mat::PIXEL_RGB, makeup_img.cols, makeup_img.rows);
    non_makeup.substract_mean_normalize(mean_vals, norm_vals);
    makeup.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Mat out;
    {
        ncnn::Extractor ex = transfer_net.create_extractor();
        ex.set_light_mode(true);
        ex.set_num_threads(ncnn::get_big_cpu_count());
        ex.set_vulkan_compute(true);
        ex.input("non_makeup", non_makeup);
        ex.input("makeup", makeup);
        ex.input("non_makeup_parse", non_makeup_parse);
        ex.input("makeup_parse", makeup_parse);
        ex.extract("out", out);
    }

    const float mean_vals1[3] = { -1.0, -1.0, -1.0 };
    const float norm_vals1[3] = { 127.5, 127.5, 127.5 };
    out.substract_mean_normalize(mean_vals1, norm_vals1);

    cv::Mat makeup_result_resize(cv::Size(256, 256), CV_8UC3);
    out.to_pixels(makeup_result_resize.data, ncnn::Mat::PIXEL_RGB);
    cv::resize(makeup_result_resize,makeup_result,cv::Size(input_width,input_height));

    return 0;
}
