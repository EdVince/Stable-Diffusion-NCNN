#include "encoder_slover.h"

int EncodeSlover::load(AAssetManager* mgr)
{
	ncnn::set_cpu_powersave(0);
	ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

	net.opt.lightmode = true;
	net.opt.use_vulkan_compute = false;
	net.opt.use_winograd_convolution = false;
	net.opt.use_sgemm_convolution = false;
	net.opt.use_fp16_packed = true;
	net.opt.use_fp16_storage = true;
	net.opt.use_fp16_arithmetic = true;
	net.opt.use_packing_layout = true;
	net.opt.num_threads = ncnn::get_big_cpu_count();

	net.load_param(mgr,"AutoencoderKL-encoder-256-256-fp16.param");
	net.load_model(mgr,"AutoencoderKL-encoder-fp16.bin");

	return 0;
}

std::vector<ncnn::Mat> EncodeSlover::encode(ncnn::Mat& rgb_image)
{
	std::vector<ncnn::Mat> mean_std(2);
	{
		ncnn::Mat in = rgb_image;
		in.substract_mean_normalize(_mean_, _norm_);

		{
			ncnn::Extractor ex = net.create_extractor();
			ex.set_light_mode(true);
			ex.input("in0", in);
			ex.extract("out0", mean_std[0]);
			ex.extract("out1", mean_std[1]);
		}
	}

	return mean_std;
}