#include "decoder_slover.h"

int DecodeSlover::load(AAssetManager* mgr)
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

	net.load_param(mgr,"AutoencoderKL-fp16.param");
	net.load_model(mgr,"AutoencoderKL-fp16.bin");

	return 0;
}

ncnn::Mat DecodeSlover::decode(ncnn::Mat sample)
{
	ncnn::Mat x_samples_ddim;
	{
		sample.substract_mean_normalize(0, factor);

		{
			ncnn::Extractor ex = net.create_extractor();
			ex.set_light_mode(true);
			ex.input("input.1", sample);
			ex.extract("815", x_samples_ddim);
		}

		x_samples_ddim.substract_mean_normalize(_mean_, _norm_);
	}

	return x_samples_ddim;
}