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

#include "blazeface.h"

#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

static inline float intersection_area(const FaceObject& a, const FaceObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<FaceObject>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

//     #pragma omp parallel sections
    {
//         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
//         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<FaceObject>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<FaceObject>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const FaceObject& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const FaceObject& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

int BlazeFace::generate_anchors(int target_size,int step_size,std::vector<float> min_sizes,std::vector<float> aspect_ratios,float offset,std::vector<float> variances)
{
	int image_w = target_size;
	int image_h = target_size;

	float step_w = step_size;
	float step_h = step_size;
    int feat_size = target_size/step_size;

	int num_min_size = min_sizes.size();
	int num_aspect_ratio = aspect_ratios.size();

	for (int i = 0; i < feat_size; i++)
	{
		float center_x = offset * step_w;
		float center_y = offset * step_h + i * step_h;

		for (int j = 0; j < feat_size; j++)
		{
			float box_w;
			float box_h;

			for (int k = 0; k < num_min_size; k++)
			{
				float min_size = min_sizes[k];
				box_w = min_size;
                box_h = min_size;
                for (int p = 0; p < num_aspect_ratio; p++)
				{
					float ar = aspect_ratios[p];

					box_w = static_cast<float>(min_size * sqrt(ar));
					box_h = static_cast<float>(min_size / sqrt(ar));

					float box0 = (center_x - box_w * 0.5f) / image_w;
					float box1 = (center_y - box_h * 0.5f) / image_h;
					float box2 = (center_x + box_w * 0.5f) / image_w;
					float box3 = (center_y + box_h * 0.5f) / image_h;

					float pb_w = box2 - box0;
					float pb_h = box3 - box1;
					float pb_x = box0 + pb_w * 0.5;
					float pb_y = box1 + pb_h * 0.5;

					anchors.push_back(std::vector<float>{pb_x, pb_y, pb_w, pb_h});
				}
			}
			center_x += step_w;
		}
	}

	return 0;
}
void BlazeFace::generate_proposals(const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, float score_threshold, int num_anchors,int target_size,std::vector<FaceObject> &faceobjects)
{
	const float* score_data = (float*)score_blob.data;
	const float* bbox_data = (float*)bbox_blob.data;
    for (int i = 0; i < num_anchors; i++)
	{
        if (score_data[i * 2 + 0] > score_threshold)
		{
			FaceObject obj;
			float pb_w = anchors[i][2];
			float pb_h = anchors[i][3];
			float pb_x = anchors[i][0];
			float pb_y = anchors[i][1];

			float x_center = pb_x + bbox_data[i * 4 + 0] * pb_w * 0.1;
			float y_center = pb_y + bbox_data[i * 4 + 1] * pb_h * 0.1;
			float w = exp(bbox_data[i * 4 + 2] * 0.2) * pb_w;
			float h = exp(bbox_data[i * 4 + 3] * 0.2) * pb_h;

            float x1 = std::max(std::min((float)(x_center - w / 2.0), 1.f), 0.f) * target_size;
            float y1 = std::max(std::min((float)(y_center - h / 2.0), 1.f), 0.f) * target_size;
            float x2 = std::max(std::min((float)(x_center + w / 2.0), 1.f), 0.f) * target_size;
            float y2 = std::max(std::min((float)(y_center + h / 2.0), 1.f), 0.f) * target_size;
            float prob = std::max(std::min((float)score_data[i * 2 + 0], 1.f), 0.f);

            obj.rect = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
            obj.prob = prob;

            faceobjects.push_back(obj);
        }
    }
}


int BlazeFace::load(AAssetManager* mgr, bool use_gpu)
{
    blazeface.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    blazeface.opt = ncnn::Option();

#if NCNN_VULKAN
    blazeface.opt.use_vulkan_compute = use_gpu;
#endif

    blazeface.opt.num_threads = ncnn::get_big_cpu_count();

    int ret_param = blazeface.load_param(mgr, "blazeface.param");
    int ret_bin = blazeface.load_model(mgr, "blazeface.bin");
    if (ret_param != 0 || ret_bin != 0)
    {
        return -1;
    }
    target_size = 128;

    anchors.clear();
	for (int i = 0; i < steps.size(); i++)
	{
		generate_anchors(target_size, steps[i], min_sizes[i],  aspect_ratios[i], offset, variances);
	}

    return 0;
}

int BlazeFace::detect(const cv::Mat& rgb, std::vector<FaceObject>& faceobjects, float prob_threshold, float nms_threshold)
{
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);

    // pad to target_size rectangle
    int wpad = target_size - w;
    int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = blazeface.create_extractor();

    ex.input("image", in_pad);

	ncnn::Mat scores;
	ncnn::Mat boxes;
	ex.extract("scores", scores);
	ex.extract("boxes", boxes);

    std::vector<FaceObject> faceproposals;
    generate_proposals(scores,boxes,prob_threshold,anchors.size(),target_size,faceproposals);
    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(faceproposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(faceproposals, picked, nms_threshold);

    int face_count = picked.size();

    faceobjects.resize(face_count);
    for (int i = 0; i < face_count; i++)
    {
        faceobjects[i] = faceproposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (faceobjects[i].rect.x - (wpad / 2)) / scale - 50;
        float y0 = (faceobjects[i].rect.y - (hpad / 2)) / scale - 50;
        float x1 = (faceobjects[i].rect.x + faceobjects[i].rect.width - (wpad / 2)) / scale + 50;
        float y1 = (faceobjects[i].rect.y + faceobjects[i].rect.height - (hpad / 2)) / scale + 50;

        x0 = std::max(std::min(x0, (float)width - 1), 0.f);
        y0 = std::max(std::min(y0, (float)height - 1), 0.f);
        x1 = std::max(std::min(x1, (float)width - 1), 0.f);
        y1 = std::max(std::min(y1, (float)height - 1), 0.f);

        faceobjects[i].rect.x = x0;
        faceobjects[i].rect.y = y0;
        faceobjects[i].rect.width = x1 - x0;
        faceobjects[i].rect.height = y1 - y0;

    }

    return 0;
}

int BlazeFace::draw(cv::Mat& rgb, const std::vector<FaceObject>& faceobjects)
{
    for (size_t i = 0; i < faceobjects.size(); i++)
    {
        const FaceObject& obj = faceobjects[i];
        int rx = obj.rect.x + obj.rect.width/2 - obj.rect.height/2 > 0? obj.rect.x + obj.rect.width/2 - obj.rect.height/2 : 0;
        int ry = obj.rect.y - 20;
        int rwidth = rx + obj.rect.height < rgb.cols? obj.rect.height : rgb.cols - rx - 1;
        int rheight = obj.rect.height + ry + 40 < rgb.rows? obj.rect.height+40 : rgb.rows- ry -1;
        cv::rectangle(rgb, cv::Rect(rx,ry,rwidth,rheight), cv::Scalar(0, 255, 0));

        char text[256];
        sprintf(text, "%.1f%%", obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    return 0;
}
