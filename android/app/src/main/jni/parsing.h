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

#ifndef PARSING_H
#define PARSING_H

#include <opencv2/core/core.hpp>

#include <net.h>

class Parsing
{
public:
    int load(AAssetManager* mgr, bool use_gpu = false);

    int parsing(const cv::Mat& in, cv::Mat& out, bool use_gpu);

private:
    const float mean_vals[3] = { 123.675f, 116.28f,  103.53f };
    const float norm_vals[3] = { 0.017058f, 0.017439f, 0.017361f };
    
    int target_size;

private:
    ncnn::Net parsing_net;

};

#endif // TRANSFER_H
