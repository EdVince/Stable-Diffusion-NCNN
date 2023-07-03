#include "encoder_slover.h"

EncodeSlover::EncodeSlover(int h, int w)
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
		net.load_param("assets/AutoencoderKL-encoder-512-512-fp16.param");
	else
	{
		generate_param(h, w);
		net.load_param(("assets/tmp-AutoencoderKL-encoder-" + std::to_string(h) + "-" + std::to_string(w) + "-fp16.param").c_str());
	}
	net.load_model("assets/AutoencoderKL-encoder-512-512-fp16.bin");

	h_size = h;
	w_size = w;
}

void EncodeSlover::generate_param(int height, int width)
{
    std::string line;
    std::ifstream encoder_file("assets/AutoencoderKL-encoder-512-512-fp16.param");
    std::ofstream encoder_file_new("assets/tmp-AutoencoderKL-encoder-" + std::to_string(height) + "-" + std::to_string(width) + "-fp16.param");

	int cnt = 0;
	while (getline(encoder_file, line))
	{
		if (line.substr(0, 7) == "Reshape")
		{
			switch (cnt)
			{
			case 0: line = line.substr(0, line.size() - 12) + "0=" + std::to_string(width * height / 8 / 8) + " 1=512"; break;
			case 1: line = line.substr(0, line.size() - 15) + "0=" + std::to_string(width / 8) + " 1=" + std::to_string(height / 8) + " 2=512"; break;
			default: break;
			}
			
			cnt++;
		}
		encoder_file_new << line << std::endl;
	}
	encoder_file_new.close();
	encoder_file.close();
}

std::vector<ncnn::Mat> EncodeSlover::encode(ncnn::Mat& rgb_image)
{
    std::vector<ncnn::Mat> mean_std(2);
    {
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(
            reinterpret_cast<const unsigned char*>(rgb_image.data),
            ncnn::Mat::PIXEL_RGB,
            rgb_image.w,
            rgb_image.h,
            w_size,
            h_size);
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
