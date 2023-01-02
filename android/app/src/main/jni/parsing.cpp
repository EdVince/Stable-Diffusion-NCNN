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

#include "parsing.h"
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

int Parsing::load(AAssetManager* mgr,  bool use_gpu)
{
    parsing_net.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    parsing_net.opt = ncnn::Option();

#if NCNN_VULKAN
    parsing_net.opt.use_vulkan_compute = use_gpu;
#endif

    parsing_net.opt.num_threads = ncnn::get_big_cpu_count();

    int ret_param = parsing_net.load_param(mgr, "face_parsing.param");
    int ret_bin = parsing_net.load_model(mgr, "face_parsing.bin");
    if (ret_param != 0 || ret_bin != 0)
    {
        return -1;
    }
    target_size = 512;

    return 0;
}
int Parsing::parsing(const cv::Mat& in, cv::Mat& out, bool use_gpu)
{
    const int target_width = target_size;
    const int target_height = target_size;
    const int num_class = 19;

    ncnn::Extractor ex = parsing_net.create_extractor();
    ncnn::Mat ncnn_in = ncnn::Mat::from_pixels_resize(in.data, ncnn::Mat::PIXEL_RGB, in.cols, in.rows, target_width, target_height);

    ncnn_in.substract_mean_normalize(mean_vals, norm_vals);

    ex.set_vulkan_compute(use_gpu);
    ex.set_light_mode(true);
    ex.input("input", ncnn_in);
    ncnn::Mat output;
    ex.extract("output", output);

    out = cv::Mat::zeros(cv::Size(output.w, output.h), CV_8UC1);
    float* output_data = (float*)output.data;

    int out_h = out.rows;
    int out_w = out.cols;
    for (int i = 0; i < out_h; i++)
    {
        for (int j = 0; j < out_w; j++)
        {
            int maxk = 0;
            float tmp = output_data[0 * out_w * out_h + i * out_w + j];
            for (int k = 0; k < num_class; k++)
            {
                if (tmp < output_data[k * out_w * out_h + i * out_w + j])
                {
                    tmp = output_data[k * out_w * out_h + i * out_w + j];
                    maxk = k;
                }
            }
            out.at<uchar>(i,j) = maxk;
        }
    }

    return 0;
}