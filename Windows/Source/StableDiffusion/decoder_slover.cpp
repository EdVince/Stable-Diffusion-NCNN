#include "decoder_slover.h"

DecodeSlover::DecodeSlover(int h, int w)
{
	net.opt.use_vulkan_compute = false;
	net.opt.use_winograd_convolution = false;
	net.opt.use_sgemm_convolution = false;
	net.opt.use_fp16_packed = false;
	net.opt.use_fp16_storage = false;
	net.opt.use_fp16_arithmetic = false;
	net.opt.use_bf16_storage = true;
	net.opt.use_packing_layout = true;

	if (h == 512 && w == 512)
		net.load_param("assets/AutoencoderKL-512-512-fp16-opt.param");
	else if (h == 256 && w == 256)
		net.load_param("assets/AutoencoderKL-256-256-fp16-opt.param");
	else
	{
		generate_param(h, w);
		net.load_param(("assets/tmp-AutoencoderKL-" + std::to_string(h) + "-" + std::to_string(w) + "-fp16.param").c_str());
	}
	net.load_model("assets/AutoencoderKL-fp16.bin");
}

void DecodeSlover::generate_param(int height, int width)
{
    std::string line;
    std::ifstream decoder_file("assets/AutoencoderKL-base-fp16.param");
    std::ofstream decoder_file_new("assets/tmp-AutoencoderKL-" + std::to_string(height) + "-" + std::to_string(width) + "-fp16.param");

	int cnt = 0;
	while (getline(decoder_file, line))
	{
		if (line.substr(0, 7) == "Reshape")
		{
			if (cnt < 3)
				line = line.substr(0, line.size() - 12) + "0=" + std::to_string(width * height / 8 / 8) + " 1=512";
			else
				line = line.substr(0, line.size() - 15) + "0=" + std::to_string(width / 8) + " 1=" + std::to_string(height / 8) + " 2=512";
			cnt++;
		}
		decoder_file_new << line << std::endl;
	}
	decoder_file_new.close();
	decoder_file.close();
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
