/*
 * PROJECT:   NCNN-based Stable Diffusion
 * FILE:      StableDiffusionCore.h
 * PURPOSE:   Definition for NCNN-based Stable Diffusion Core Implementation
 *
 * LICENSE:   The MIT License
 *
 * DEVELOPER: MouriNaruto (KurikoMouri@outlook.jp)
 */

#ifndef STABLE_DIFFUSION_CORE
#define STABLE_DIFFUSION_CORE

#include <Windows.h>

#include <functional>
#include <string>

#include <ncnn/mat.h>

namespace StableDiffusion
{
    extern HWND MainWindowHandle;

    extern std::function<void(std::wstring const&)> WriteLine;

    ncnn::Mat ReadImage(
        std::string const& FileName);

    bool WriteImage(
        std::string const& FileName,
        ncnn::Mat const& Image);

    void GenerateImage(
        int Width,
        int Height,
        bool FastMode,
        int Step,
        int Seed,
        std::string const& InputImageFileName,
        std::string const& PositivePrompt,
        std::string const& NegativePrompt,
        ncnn::Mat& GeneratedImageForShow,
        ncnn::Mat& GeneratedImageForSave);
}

namespace winrt::StableDiffusion
{
    using namespace ::StableDiffusion;
}

#endif // !STABLE_DIFFUSION_CORE
