#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <stack>
#include <fstream>
#include <map>
#include <math.h>
#include <net.h>
#include "prompt_slover.h"
#include "decoder_slover.h"
#include "diffusion_slover.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <time.h>
using namespace std;

int main()
{
	int step, seed;
	string positive_prompt, negative_prompt;

	// default setting
	step = 15;
	seed = 42;
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
		for (i = 0; i < 4; i++) {
			if (getline(magic, content)) {
				switch (i)
				{
				case 0:step = stoi(content);
				case 1:seed = stoi(content);
				case 2:positive_prompt = content;
				case 3:negative_prompt = content;
				default:break;
				}
			}
			else {
				break;
			}
		}
		if (i != 4) {
			cout << "magic.txt has wrong format, please fix it" << endl;
			return 0;
		}

	}
	if (seed == 0) {
		seed = (unsigned)time(NULL);
	}
	magic.close();

	// stable diffusion
	cout << "----------------[init]--------------------" << endl;
	PromptSlover prompt_slover;
	DiffusionSlover diffusion_slover;
	DecodeSlover decode_slover;

	cout << "----------------[prompt]------------------" << endl;
	ncnn::Mat cond = prompt_slover.get_conditioning(positive_prompt);
	ncnn::Mat uncond = prompt_slover.get_conditioning(negative_prompt);

	cout << "----------------[diffusion]---------------" << endl;
	ncnn::Mat sample = diffusion_slover.sampler(seed, step, cond, uncond);

	cout << "----------------[decode]------------------" << endl;
	ncnn::Mat x_samples_ddim = decode_slover.decode(sample);

	cout << "----------------[save]--------------------" << endl;
	cv::Mat image(512, 512, CV_8UC3);
	x_samples_ddim.to_pixels(image.data, ncnn::Mat::PIXEL_RGB2BGR);
	cv::imwrite("result_" + to_string(step) + "_" + to_string(seed) + ".png", image);

	cout << "----------------[close]-------------------" << endl;

	return 0;
}
