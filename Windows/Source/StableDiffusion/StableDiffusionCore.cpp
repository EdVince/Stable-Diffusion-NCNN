/*
 * PROJECT:   NCNN-based Stable Diffusion
 * FILE:      StableDiffusionCore.cpp
 * PURPOSE:   Implementation for NCNN-based Stable Diffusion Core Implementation
 *
 * LICENSE:   The MIT License
 *
 * DEVELOPER: MouriNaruto (KurikoMouri@outlook.jp)
 */

#include "StableDiffusionCore.h"

#define STB_IMAGE_IMPLEMENTATION
#if defined(_M_ARM64) || defined(_M_HYBRID_X86_ARM64) || defined (_M_ARM64EC)
#define STBI_NEON
#endif
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "getmem.h"

#include "prompt_slover.h"
#include "encoder_slover.h"
#include "diffusion_slover.h"
#include "decoder_slover.h"

#include <format>

HWND StableDiffusion::MainWindowHandle = nullptr;

std::function<void(std::wstring const&)> StableDiffusion::WriteLine;

ncnn::Mat StableDiffusion::ReadImage(
    std::string const& FileName)
{
    ncnn::Mat Image;

    int w = 0;
    int h = 0;
    int c = 0;
    unsigned char* data = ::stbi_load(
        FileName.c_str(),
        &w,
        &h,
        &c,
        0);
    if (data)
    {
        Image.create(w, h, c);
        std::memcpy(Image.data, data, w * h * c);
        ::stbi_image_free(data);
    }

    return Image;
}

bool StableDiffusion::WriteImage(
    std::string const& FileName,
    ncnn::Mat const& Image)
{
    return ::stbi_write_png(
        FileName.c_str(),
        Image.w,
        Image.h,
        Image.c,
        Image.data,
        0);
}

void StableDiffusion::GenerateImage(
    int Width,
    int Height,
    bool FastMode,
    int Step,
    int Seed,
    std::string const& InputImageFileName,
    std::string const& PositivePrompt,
    std::string const& NegativePrompt,
    ncnn::Mat& GeneratedImageForShow,
    ncnn::Mat& GeneratedImageForSave)
{
    StableDiffusion::WriteLine(
        L"----------------[ prompt  ]----------------");
    ncnn::Mat PositiveCondition;
    ncnn::Mat NegativeCondition;
    {
        PromptSlover Slover;

        PositiveCondition = Slover.get_conditioning(
            const_cast<std::string&>(PositivePrompt));
        NegativeCondition = Slover.get_conditioning(
            const_cast<std::string&>(NegativePrompt));
    }
    StableDiffusion::WriteLine(std::format(
        L"Memory Usage: {:.2f} GiB / {:.2f} GiB",
        getCurrentRSS() / 1024.0 / 1024.0 / 1024.0,
        getPeakRSS() / 1024.0 / 1024.0 / 1024.0));

    std::vector<ncnn::Mat> init_latents;
    ncnn::Mat img = StableDiffusion::ReadImage(InputImageFileName);
    if (!img.empty()) {
        StableDiffusion::WriteLine(
            L"----------------[ encoder ]----------------");
        EncodeSlover encode_slover(Height, Width);
        init_latents = encode_slover.encode(img);
        StableDiffusion::WriteLine(std::format(
            L"Memory Usage: {:.2f} GiB / {:.2f} GiB",
            getCurrentRSS() / 1024.0 / 1024.0 / 1024.0,
            getPeakRSS() / 1024.0 / 1024.0 / 1024.0));
    }

    StableDiffusion::WriteLine(
        L"----------------[diffusion]----------------");
    ncnn::Mat sample;
    {
        DiffusionSlover Slover(Height, Width, FastMode);

        if (!img.empty())
        {
            sample = Slover.sampler_img2img(
                Seed,
                Step,
                PositiveCondition,
                NegativeCondition,
                init_latents);
        }
        else
        {
            sample = Slover.sampler_txt2img(
                Seed,
                Step,
                PositiveCondition,
                NegativeCondition);
        }
    }
    StableDiffusion::WriteLine(std::format(
        L"Memory Usage: {:.2f} GiB / {:.2f} GiB",
        getCurrentRSS() / 1024.0 / 1024.0 / 1024.0,
        getPeakRSS() / 1024.0 / 1024.0 / 1024.0));

    StableDiffusion::WriteLine(
        L"----------------[ decoder ]----------------");
    ncnn::Mat x_samples_ddim;
    {
        DecodeSlover Slover(Height, Width);
        x_samples_ddim = Slover.decode(sample);
    }
    StableDiffusion::WriteLine(std::format(
        L"Memory Usage: {:.2f} GiB / {:.2f} GiB",
        getCurrentRSS() / 1024.0 / 1024.0 / 1024.0,
        getPeakRSS() / 1024.0 / 1024.0 / 1024.0));

    StableDiffusion::WriteLine(
        L"----------------[  save   ]----------------");
    GeneratedImageForShow.create(Width, Height, 4);
    x_samples_ddim.to_pixels(
        reinterpret_cast<unsigned char*>(GeneratedImageForShow.data),
        ncnn::Mat::PIXEL_RGB2BGRA);
    GeneratedImageForSave.create(Width, Height, 3);
    x_samples_ddim.to_pixels(
        reinterpret_cast<unsigned char*>(GeneratedImageForSave.data),
        ncnn::Mat::PIXEL_RGB);
    StableDiffusion::WriteLine(std::format(
        L"Memory Usage: {:.2f} GiB / {:.2f} GiB",
        getCurrentRSS() / 1024.0 / 1024.0 / 1024.0,
        getPeakRSS() / 1024.0 / 1024.0 / 1024.0));
}
