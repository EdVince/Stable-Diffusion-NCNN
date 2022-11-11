#include "decoder_slover.h"

DecodeSlover::DecodeSlover()
{
	opt.use_packing_layout = true;
	opt.use_fp16_arithmetic = true;
	opt.use_fp16_packed = true;
	opt.use_fp16_storage = true;
	opt.num_threads = 8;
	
	net.opt = opt;
	net.load_param("assets/AutoencoderKL-fp16.param");
	net.load_model("assets/AutoencoderKL-fp16.bin");
}

ncnn::Mat DecodeSlover::decode(ncnn::Mat sample)
{
	ncnn::Mat x_samples_ddim;
	{
		sample.substract_mean_normalize(0, factor);

		{
			ncnn::Extractor ex = net.create_extractor();
			ex.set_light_mode(TRUE);
			ex.input("input.1", sample);
			ex.extract("815", x_samples_ddim);
		}

		x_samples_ddim.substract_mean_normalize(_mean_, _norm_);

	}

	return x_samples_ddim;
}