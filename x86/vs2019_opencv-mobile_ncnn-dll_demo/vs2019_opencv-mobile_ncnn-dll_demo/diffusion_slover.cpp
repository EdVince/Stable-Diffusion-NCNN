#include "diffusion_slover.h"

DiffusionSlover::DiffusionSlover(int h, int w, int mode)
{
	net.opt.use_vulkan_compute = false;
	net.opt.lightmode = true;
	if (mode == 0)
	{
		net.opt.use_winograd_convolution = false;
		net.opt.use_sgemm_convolution = false;
	}
	else
	{
		net.opt.use_winograd_convolution = true;
		net.opt.use_sgemm_convolution = true;
	}
	net.opt.use_fp16_packed = true;
	net.opt.use_fp16_storage = true;
	net.opt.use_fp16_arithmetic = true;
	net.opt.use_packing_layout = true;

	if (h == 512 && w == 512)
		net.load_param("assets/UNetModel-512-512-MHA-fp16-opt.param");
	else if (h == 256 && w == 256)
		net.load_param("assets/UNetModel-256-256-MHA-fp16-opt.param");
	else
	{
		generate_param(h, w);
		net.load_param(("assets/tmp-UNetModel-" + std::to_string(h) + "-" + std::to_string(w) + "-MHA-fp16.param").c_str());
	}
	net.load_model("assets/UNetModel-MHA-fp16.bin");

	h_size = h / 8;
	w_size = w / 8;

	ifstream in("assets/log_sigmas.bin", ios::in | ios::binary);
	in.read((char*)&log_sigmas, sizeof log_sigmas);
	in.close();
}

void DiffusionSlover::generate_param(int height, int width)
{
	string line;
	ifstream diffuser_file("assets/UNetModel-base-MHA-fp16.param");
	ofstream diffuser_file_new("assets/tmp-UNetModel-" + std::to_string(height) + "-" + std::to_string(width) + "-MHA-fp16.param");

	int cnt = 0;
	while (getline(diffuser_file, line))
	{
		if (line.substr(0, 7) == "Reshape")
		{
			switch (cnt)
			{
			case 0: line = line.substr(0, line.size() - 4) + to_string(width * height / 8 / 8); break;
			case 1: line = line.substr(0, line.size() - 7) + to_string(width / 8) + " 2=" + std::to_string(height / 8); break;
			case 2: line = line.substr(0, line.size() - 4) + to_string(width * height / 8 / 8); break;
			case 3: line = line.substr(0, line.size() - 7) + to_string(width / 8) + " 2=" + std::to_string(height / 8); break;
			case 4: line = line.substr(0, line.size() - 4) + to_string(width * height / 2 / 2 / 8 / 8); break;
			case 5: line = line.substr(0, line.size() - 7) + to_string(width / 2 / 8) + " 2=" + std::to_string(height / 2 / 8); break;
			case 6: line = line.substr(0, line.size() - 4) + to_string(width * height / 2 / 2 / 8 / 8); break;
			case 7: line = line.substr(0, line.size() - 7) + to_string(width / 2 / 8) + " 2=" + std::to_string(height / 2 / 8); break;
			case 8: line = line.substr(0, line.size() - 3) + to_string(width * height / 4 / 4 / 8 / 8); break;
			case 9: line = line.substr(0, line.size() - 7) + to_string(width / 4 / 8) + " 2=" + std::to_string(height / 4 / 8); break;
			case 10: line = line.substr(0, line.size() - 3) + to_string(width * height / 4 / 4 / 8 / 8); break;
			case 11: line = line.substr(0, line.size() - 7) + to_string(width / 4 / 8) + " 2=" + std::to_string(height / 4 / 8); break;
			case 12: line = line.substr(0, line.size() - 2) + to_string(width * height / 8 / 8 / 8 / 8); break;
			case 13: line = line.substr(0, line.size() - 5) + to_string(width / 8 / 8) + " 2=" + std::to_string(height / 8 / 8); break;
			case 14: line = line.substr(0, line.size() - 3) + to_string(width * height / 4 / 4 / 8 / 8); break;
			case 15: line = line.substr(0, line.size() - 7) + to_string(width / 4 / 8) + " 2=" + std::to_string(height / 4 / 8); break;
			case 16: line = line.substr(0, line.size() - 3) + to_string(width * height / 4 / 4 / 8 / 8); break;
			case 17: line = line.substr(0, line.size() - 7) + to_string(width / 4 / 8) + " 2=" + std::to_string(height / 4 / 8); break;
			case 18: line = line.substr(0, line.size() - 3) + to_string(width * height / 4 / 4 / 8 / 8); break;
			case 19: line = line.substr(0, line.size() - 7) + to_string(width / 4 / 8) + " 2=" + std::to_string(height / 4 / 8); break;
			case 20: line = line.substr(0, line.size() - 4) + to_string(width * height / 2 / 2 / 8 / 8); break;
			case 21: line = line.substr(0, line.size() - 7) + to_string(width / 2 / 8) + " 2=" + std::to_string(height / 2 / 8); break;
			case 22: line = line.substr(0, line.size() - 4) + to_string(width * height / 2 / 2 / 8 / 8); break;
			case 23: line = line.substr(0, line.size() - 7) + to_string(width / 2 / 8) + " 2=" + std::to_string(height / 2 / 8); break;
			case 24: line = line.substr(0, line.size() - 4) + to_string(width * height / 2 / 2 / 8 / 8); break;
			case 25: line = line.substr(0, line.size() - 7) + to_string(width / 2 / 8) + " 2=" + std::to_string(height / 2 / 8); break;
			case 26: line = line.substr(0, line.size() - 4) + to_string(width * height / 8 / 8); break;
			case 27: line = line.substr(0, line.size() - 7) + to_string(width / 8) + " 2=" + std::to_string(height / 8); break;
			case 28: line = line.substr(0, line.size() - 4) + to_string(width * height / 8 / 8); break;
			case 29: line = line.substr(0, line.size() - 7) + to_string(width / 8) + " 2=" + std::to_string(height / 8); break;
			case 30: line = line.substr(0, line.size() - 4) + to_string(width * height / 8 / 8); break;
			case 31: line = line.substr(0, line.size() - 7) + to_string(width / 8) + " 2=" + std::to_string(height / 8); break;
			default: break;
			}

			cnt++;
		}
		diffuser_file_new << line << endl;
	}
	diffuser_file_new.close();
	diffuser_file.close();
}

ncnn::Mat DiffusionSlover::randn_4(int seed)
{
	cv::Mat cv_x(cv::Size(w_size, h_size), CV_32FC4);
	cv::RNG rng(seed);
	rng.fill(cv_x, cv::RNG::NORMAL, 0, 1);
	ncnn::Mat x_mat(w_size, h_size, 4, (void*)cv_x.data);
	return x_mat.clone();
}

ncnn::Mat DiffusionSlover::CFGDenoiser_CompVisDenoiser(ncnn::Mat& input, float sigma, ncnn::Mat cond, ncnn::Mat uncond)
{
	// get_scalings
	float c_out = -1.0 * sigma;
	float c_in = 1.0 / sqrt(sigma * sigma + 1);

	// sigma_to_t
	float log_sigma = log(sigma);
	vector<float> dists(1000);
	for (int i = 0; i < 1000; i++) {
		if (log_sigma - log_sigmas[i] >= 0)
			dists[i] = 1;
		else
			dists[i] = 0;
		if (i == 0) continue;
		dists[i] += dists[i - 1];
	}
	int low_idx = min(int(max_element(dists.begin(), dists.end()) - dists.begin()), 1000 - 2);
	int high_idx = low_idx + 1;
	float low = log_sigmas[low_idx];
	float high = log_sigmas[high_idx];
	float w = (low - log_sigma) / (low - high);
	w = max(0.f, min(1.f, w));
	float t = (1 - w) * low_idx + w * high_idx;

	ncnn::Mat t_mat(1);
	t_mat[0] = t;

	ncnn::Mat c_in_mat(1);
	c_in_mat[0] = c_in;

	ncnn::Mat c_out_mat(1);
	c_out_mat[0] = c_out;

	ncnn::Mat v44;
	ncnn::Mat v83;
	ncnn::Mat v116;
	ncnn::Mat v163;
	ncnn::Mat v251;
	ncnn::Mat v337;
	ncnn::Mat v425;
	ncnn::Mat v511;
	ncnn::Mat v599;
	ncnn::Mat v627;
	ncnn::Mat v711;
	ncnn::Mat v725;
	ncnn::Mat v740;
	ncnn::Mat v755;
	ncnn::Mat v772;
	ncnn::Mat v858;
	ncnn::Mat v944;
	ncnn::Mat v1032;
	ncnn::Mat v1118;
	ncnn::Mat v1204;
	ncnn::Mat v1292;
	ncnn::Mat v1378;
	ncnn::Mat v1464;

	ncnn::Mat denoised_cond;
	{
		ncnn::Extractor ex = net.create_extractor();
		ex.set_light_mode(true);
		ex.input("in0", input);
		ex.input("in1", t_mat);
		ex.input("in2", cond);
		ex.input("c_in", c_in_mat);
		ex.input("c_out", c_out_mat);
		ex.extract("44", v44, 1);
		ex.extract("83", v83, 1);
		ex.extract("116", v116, 1);
		ex.extract("163", v163, 1);
		ex.extract("251", v251, 1);
		ex.extract("337", v337, 1);
		ex.extract("425", v425, 1);
		ex.extract("511", v511, 1);
		ex.extract("599", v599, 1);
		ex.extract("627", v627, 1);
		ex.extract("711", v711, 1);
		ex.extract("725", v725, 1);
		ex.extract("740", v740, 1);
		ex.extract("755", v755, 1);
		ex.extract("772", v772, 1);
		ex.extract("858", v858, 1);
		ex.extract("944", v944, 1);
		ex.extract("1032", v1032, 1);
		ex.extract("1118", v1118, 1);
		ex.extract("1204", v1204, 1);
		ex.extract("1292", v1292, 1);
		ex.extract("1378", v1378, 1);
		ex.extract("1464", v1464, 1);
		ex.extract("outout", denoised_cond);
	}

	ncnn::Mat denoised_uncond;
	{
		ncnn::Extractor ex = net.create_extractor();
		ex.set_light_mode(true);
		ex.input("in0", input);
		ex.input("in1", t_mat);
		ex.input("in2", uncond);
		ex.input("c_in", c_in_mat);
		ex.input("c_out", c_out_mat);
		ex.input("44", v44);
		ex.input("83", v83);
		ex.input("116", v116);
		ex.input("163", v163);
		ex.input("251", v251);
		ex.input("337", v337);
		ex.input("425", v425);
		ex.input("511", v511);
		ex.input("599", v599);
		ex.input("627", v627);
		ex.input("711", v711);
		ex.input("725", v725);
		ex.input("740", v740);
		ex.input("755", v755);
		ex.input("772", v772);
		ex.input("858", v858);
		ex.input("944", v944);
		ex.input("1032", v1032);
		ex.input("1118", v1118);
		ex.input("1204", v1204);
		ex.input("1292", v1292);
		ex.input("1378", v1378);
		ex.input("1464", v1464);
		ex.extract("outout", denoised_uncond);
	}

	for (int c = 0; c < 4; c++) {
		float* u_ptr = denoised_uncond.channel(c);
		float* c_ptr = denoised_cond.channel(c);
		for (int hw = 0; hw < h_size * w_size; hw++) {
			(*u_ptr) = (*u_ptr) + guidance_scale * ((*c_ptr) - (*u_ptr));
			u_ptr++;
			c_ptr++;
		}
	}

	return denoised_uncond;
}

ncnn::Mat DiffusionSlover::sampler_txt2img(int seed, int step, ncnn::Mat& c, ncnn::Mat& uc)
{
	// t_to_sigma
	vector<float> sigma(step);
	float delta = 0.0 - 999.0 / (step - 1);
	for (int i = 0; i < step; i++) {
		float t = 999.0 + i * delta;
		int low_idx = floor(t);
		int high_idx = ceil(t);
		float w = t - low_idx;
		sigma[i] = exp((1 - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx]);
	}
	sigma.push_back(0.f);

	// init
	ncnn::Mat x_mat = randn_4(seed % 1000);
	float _norm_[4] = { sigma[0], sigma[0], sigma[0], sigma[0] };
	x_mat.substract_mean_normalize(0, _norm_);

	// euler ancestral
	{
		for (int i = 0; i < sigma.size() - 1; i++) {
			printf("step:%2d/%d\t", i + 1, sigma.size() - 1);

			double t1 = ncnn::get_current_time();
			ncnn::Mat denoised = CFGDenoiser_CompVisDenoiser(x_mat, sigma[i], c, uc);
			double t2 = ncnn::get_current_time();
			printf("%.2lfms\n", t2 - t1);

			float sigma_up = min(sigma[i + 1], sqrt(sigma[i + 1] * sigma[i + 1] * (sigma[i] * sigma[i] - sigma[i + 1] * sigma[i + 1]) / (sigma[i] * sigma[i])));
			float sigma_down = sqrt(sigma[i + 1] * sigma[i + 1] - sigma_up * sigma_up);

			srand(time(NULL) + i);
			ncnn::Mat randn = randn_4(rand() % 1000);
			for (int c = 0; c < 4; c++) {
				float* x_ptr = x_mat.channel(c);
				float* d_ptr = denoised.channel(c);
				float* r_ptr = randn.channel(c);
				for (int hw = 0; hw < h_size * w_size; hw++) {
					*x_ptr = *x_ptr + ((*x_ptr - *d_ptr) / sigma[i]) * (sigma_down - sigma[i]) + *r_ptr * sigma_up;
					x_ptr++;
					d_ptr++;
					r_ptr++;
				}
			}
		}
	}

	/*
	// DPM++ 2M Karras
	ncnn::Mat old_denoised;
	{
		for (int i = 0; i < sigma.size() - 1; i++) {
			cout << "step:" << i << "\t\t";

			double t1 = ncnn::get_current_time();
			ncnn::Mat denoised = CFGDenoiser_CompVisDenoiser(x_mat, sigma[i], c, uc);
			double t2 = ncnn::get_current_time();
			cout << t2 - t1 << "ms" << endl;

			float sigma_curt = sigma[i];
			float sigma_next = sigma[i + 1];
			float tt = -1.0 * log(sigma_curt);
			float tt_next = -1.0 * log(sigma_next);
			float hh = tt_next - tt;
			if (old_denoised.empty() || sigma_next == 0)
			{
				for (int c = 0; c < 4; c++) {
					float* x_ptr = x_mat.channel(c);
					float* d_ptr = denoised.channel(c);
					for (int hw = 0; hw < size * size; hw++) {
						*x_ptr = (sigma_next / sigma_curt) * *x_ptr - (exp(-hh) - 1) * *d_ptr;
						x_ptr++;
						d_ptr++;
					}
				}
			}
			else
			{
				float hh_last = -1.0 * log(sigma[i - 1]);
				float r = hh_last / hh;
				for (int c = 0; c < 4; c++) {
					float* x_ptr = x_mat.channel(c);
					float* d_ptr = denoised.channel(c);
					float* od_ptr = old_denoised.channel(c);
					for (int hw = 0; hw < size * size; hw++) {
						*x_ptr = (sigma_next / sigma_curt) * *x_ptr - (exp(-hh) - 1) * ((1 + 1 / (2 * r)) * *d_ptr - (1 / (2 * r)) * *od_ptr);
						x_ptr++;
						d_ptr++;
						od_ptr++;
					}
				}
			}
			old_denoised.clone_from(denoised);
		}
	}
	*/

	ncnn::Mat fuck_x;
	fuck_x.clone_from(x_mat);
	return fuck_x;
}

ncnn::Mat DiffusionSlover::sampler_img2img(int seed, int step, ncnn::Mat& c, ncnn::Mat& uc, vector<ncnn::Mat>& init)
{
	// t_to_sigma
	vector<float> sigma(step);
	float delta = 0.0 - 999.0 / (step - 1);
	for (int i = 0; i < step; i++) {
		float t = 999.0 + i * delta;
		int low_idx = floor(t);
		int high_idx = ceil(t);
		float w = t - low_idx;
		sigma[i] = exp((1 - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx]);
	}
	sigma.push_back(0.f);

	// init
	ncnn::Mat x_mat(w_size, h_size, 4);

	// finish the rest of decoder
	{
		ncnn::Mat noise_mat = randn_4(seed % 1000);
		for (int c = 0; c < 4; c++) {
			float* x_ptr = x_mat.channel(c);
			float* noise_ptr = noise_mat.channel(c);
			float* mean_ptr = init[0].channel(c);
			float* std_ptr = init[1].channel(c);
			for (int hw = 0; hw < h_size * w_size; hw++) {
				*x_ptr = *mean_ptr + *std_ptr * *noise_ptr;
				x_ptr++;
				noise_ptr++;
				mean_ptr++;
				std_ptr++;
			}
		}
		x_mat.substract_mean_normalize(0, factor);
	}

	// reset scheduling
	int new_step = step * strength;
	{
		float _sigma_ = sigma[step - new_step];
		ncnn::Mat noise_mat = randn_4(seed % 1000);
		for (int c = 0; c < 4; c++) {
			float* x_ptr = x_mat.channel(c);
			float* noise_ptr = noise_mat.channel(c);
			for (int hw = 0; hw < h_size * w_size; hw++) {
				*x_ptr = *x_ptr + *noise_ptr * _sigma_;
				x_ptr++;
				noise_ptr++;
			}
		}
	}
	vector<float> sub_sigma(sigma.begin() + step - new_step, sigma.end());

	// euler ancestral
	{
		for (int i = 0; i < sub_sigma.size() - 1; i++) {
			printf("step:%2d/%d\t", i+1, sub_sigma.size()-1);

			double t1 = ncnn::get_current_time();
			ncnn::Mat denoised = CFGDenoiser_CompVisDenoiser(x_mat, sub_sigma[i], c, uc);
			double t2 = ncnn::get_current_time();
			printf("%.2lfms\n", t2 - t1);

			float sigma_up = min(sub_sigma[i + 1], sqrt(sub_sigma[i + 1] * sub_sigma[i + 1] * (sub_sigma[i] * sub_sigma[i] - sub_sigma[i + 1] * sub_sigma[i + 1]) / (sub_sigma[i] * sub_sigma[i])));
			float sigma_down = sqrt(sub_sigma[i + 1] * sub_sigma[i + 1] - sigma_up * sigma_up);

			srand(time(NULL) + i);
			ncnn::Mat randn = randn_4(rand() % 1000);
			for (int c = 0; c < 4; c++) {
				float* x_ptr = x_mat.channel(c);
				float* d_ptr = denoised.channel(c);
				float* r_ptr = randn.channel(c);
				for (int hw = 0; hw < h_size * w_size; hw++) {
					*x_ptr = *x_ptr + ((*x_ptr - *d_ptr) / sub_sigma[i]) * (sigma_down - sub_sigma[i]) + *r_ptr * sigma_up;
					x_ptr++;
					d_ptr++;
					r_ptr++;
				}
			}
		}
	}

	ncnn::Mat fuck_x;
	fuck_x.clone_from(x_mat);
	return fuck_x;
}
