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
#include "getmem.h"

#include "prompt_slover.h"
#include "encoder_slover.h"
#include "diffusion_slover.h"
#include "decoder_slover.h"

using namespace std;

int main()
{
	int height, width, mode, step, seed;
	string init_image;
	string positive_prompt, negative_prompt;

	// default setting
	height = 256;
	width = 256;
	mode = 0;
	step = 15;
	seed = 42;
	init_image = "";
	positive_prompt = "floating hair, portrait, ((loli)), ((one girl)), cute face, hidden hands, asymmetrical bangs, beautiful detailed eyes, eye shadow, hair ornament, ribbons, bowties, buttons, pleated skirt, (((masterpiece))), ((best quality)), colorful";
	negative_prompt = "((part of the head)), ((((mutated hands and fingers)))), deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, Octane renderer, lowres, bad anatomy, bad hands, text";

	// parse the magic.txt
	ifstream magic;
	magic.open("magic.txt");
	if (!magic) {
		cout << "can not find magic.txt, using the default setting" << endl;
	}
	else {
		string content = "";
		int i = 0;
		for (i = 0; i < 8; i++) {
			if (getline(magic, content)) {
				switch (i)
				{
				case 0:height = stoi(content);
				case 1:width = stoi(content);
				case 2:mode = stoi(content);
				case 3:step = stoi(content);
				case 4:seed = stoi(content);
				case 5:init_image = content;
				case 6:positive_prompt = content;
				case 7:negative_prompt = content;
				default:break;
				}
			}
			else {
				break;
			}
		}
		if (i != 8) {
			cout << "magic.txt has wrong format, please fix it" << endl;
			return 0;
		}

	}
	if (seed == 0) {
		seed = (unsigned)time(NULL);
	}
	magic.close();

	// stable diffusion
	cout << "----------------[  init   ]----------------";
	PromptSlover prompt_slover;
	DiffusionSlover diffusion_slover(height, width, mode);
	DecodeSlover decode_slover(height, width);
	EncodeSlover encode_slover(height, width);
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	cout << "----------------[ prompt  ]----------------";
	ncnn::Mat cond = prompt_slover.get_conditioning(positive_prompt);
	ncnn::Mat uncond = prompt_slover.get_conditioning(negative_prompt);
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	vector<ncnn::Mat> init_latents;
	cv::Mat img = cv::imread(init_image);
	if (!img.empty()) {
		cout << "----------------[ encoder ]----------------";
		init_latents = encode_slover.encode(img);
		printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);
	}

	cout << "----------------[diffusion]----------------" << endl;
	ncnn::Mat sample;
	if (!img.empty()) {
		sample = diffusion_slover.sampler_img2img(seed, step, cond, uncond, init_latents);
	}
	else {
		sample = diffusion_slover.sampler_txt2img(seed, step, cond, uncond);
	}
	cout << "----------------[diffusion]----------------";
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	cout << "----------------[ decoder ]----------------";
	ncnn::Mat x_samples_ddim = decode_slover.decode(sample);
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	cout << "----------------[  save   ]----------------" << endl;
	cv::Mat image(height, width, CV_8UC3);
	x_samples_ddim.to_pixels(image.data, ncnn::Mat::PIXEL_RGB2BGR);
	cv::imwrite("result_" + to_string(step) + "_" + to_string(seed) + "_" + to_string(height) + "x" + to_string(width) + ".png", image);

	return 0;
}
