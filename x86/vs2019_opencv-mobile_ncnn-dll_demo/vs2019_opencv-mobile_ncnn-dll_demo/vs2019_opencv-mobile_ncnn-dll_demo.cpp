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

	//cout << "------------------------------------------" << endl;
	//cout << "[step]:"; cin >> step;
	//cout << "[seed]:"; cin >> seed; if (seed <= 0) seed = (unsigned)time(NULL);
	//getline(cin, positive_prompt);
	//cout << "[positive prompt]:" << endl; getline(cin, positive_prompt);
	//cout << "[negative prompt]:" << endl; getline(cin, negative_prompt);
	//cout << "------------------------------------------" << endl;
	//cout << "[step]:" << step << ", [seed]:" << seed << endl;
	//cout << "[positive prompt]:" << positive_prompt << endl;
	//cout << "[negative prompt]:" << negative_prompt << endl;

	step = 15;
	seed = 42;
	positive_prompt = "floating hair, portrait, ((loli)), ((one girl)), cute face, hidden hands, asymmetrical bangs, beautiful detailed eyes, eye shadow, hair ornament, ribbons, bowties, buttons, pleated skirt, (((masterpiece))), ((best quality)), colorful";
	negative_prompt = "((part of the head)), ((((mutated hands and fingers)))), deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, Octane renderer, lowres, bad anatomy, bad hands, text";

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
