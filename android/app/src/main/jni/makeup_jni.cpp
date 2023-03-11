// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>
#include <codecvt>

// ncnn
#include "net.h"
#include "benchmark.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "prompt_slover.h"
#include "diffusion_slover.h"
#include "decoder_slover.h"
#include "encoder_slover.h"

static std::string UTF16StringToUTF8String(const char16_t* chars, size_t len) {
    std::u16string u16_string(chars, len);
    return std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t>{}
            .to_bytes(u16_string);
}

std::string JavaStringToString(JNIEnv* env, jstring str) {
    if (env == nullptr || str == nullptr) {
        return "";
    }
    const jchar* chars = env->GetStringChars(str, NULL);
    if (chars == nullptr) {
        return "";
    }
    std::string u8_string = UTF16StringToUTF8String(
            reinterpret_cast<const char16_t*>(chars), env->GetStringLength(str));
    env->ReleaseStringChars(str, chars);
    return u8_string;
}

static PromptSlover prompt_slover;
static EncodeSlover encode_slover;
static DiffusionSlover diffusion_slover;
static DecodeSlover decode_slover;
extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "makeup", "JNI_OnLoad");

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "makeup", "JNI_OnUnload");
}

// public native boolean Init(AssetManager mgr);
JNIEXPORT jboolean JNICALL Java_com_tencent_makeup_StableDiffusion_Init(JNIEnv* env, jobject thiz, jobject assetManager, jstring jvocab, jstring jbin)
{
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    std::string vocab = JavaStringToString(env,jvocab);
    if(prompt_slover.load(mgr, vocab) < 0)
        return JNI_FALSE;

    std::string bin = JavaStringToString(env,jbin);
    if(diffusion_slover.load(mgr, bin) < 0)
        return JNI_FALSE;

    if(decode_slover.load(mgr) < 0)
        return JNI_FALSE;

    if(encode_slover.load(mgr) < 0)
        return JNI_FALSE;

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_tencent_makeup_StableDiffusion_txt2imgProcess(JNIEnv* env, jobject thiz, jobject show_bitmap, jint jstep, jint jseed, jstring jpositivePromptText, jstring jnegativePrompt)
{
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, show_bitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return JNI_FALSE;

    int step = jstep;
    int seed = jseed;
    std::string positive_prompt = "" + JavaStringToString(env,jpositivePromptText);
    std::string negative_prompt = "" + JavaStringToString(env,jnegativePrompt);
    if (positive_prompt == "" || negative_prompt == "")
    {
        positive_prompt = "floating hair, portrait, ((loli)), ((one girl)), cute face, hidden hands, asymmetrical bangs, beautiful detailed eyes, eye shadow, hair ornament, ribbons, bowties, buttons, pleated skirt, (((masterpiece))), ((best quality)), colorful";
        negative_prompt = "((part of the head)), ((((mutated hands and fingers)))), deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, Octane renderer, lowres, bad anatomy, bad hands, text";
    }

    ncnn::Mat cond = prompt_slover.get_conditioning(positive_prompt);
    ncnn::Mat uncond = prompt_slover.get_conditioning(negative_prompt);

    ncnn::Mat sample = diffusion_slover.sampler_txt2img(seed, step, cond, uncond);

    ncnn::Mat x_samples_ddim = decode_slover.decode(sample);

    x_samples_ddim.to_android_bitmap(env,show_bitmap,ncnn::Mat::PIXEL_RGB);

    cv::Mat image(256, 256, CV_8UC3);
    x_samples_ddim.to_pixels(image.data, ncnn::Mat::PIXEL_RGB);

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_tencent_makeup_StableDiffusion_img2imgProcess(JNIEnv* env, jobject thiz, jobject show_bitmap, jint jstep, jint jseed, jstring jpositivePromptText, jstring jnegativePrompt)
{
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, show_bitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return JNI_FALSE;

    int step = jstep;
    int seed = jseed;
    std::string positive_prompt = "" + JavaStringToString(env,jpositivePromptText);
    std::string negative_prompt = "" + JavaStringToString(env,jnegativePrompt);
    if (positive_prompt == "" || negative_prompt == "")
    {
        positive_prompt = "floating hair, portrait, ((loli)), ((one girl)), cute face, hidden hands, asymmetrical bangs, beautiful detailed eyes, eye shadow, hair ornament, ribbons, bowties, buttons, pleated skirt, (((masterpiece))), ((best quality)), colorful";
        negative_prompt = "((part of the head)), ((((mutated hands and fingers)))), deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, Octane renderer, lowres, bad anatomy, bad hands, text";
    }
    ncnn::Mat init_image = ncnn::Mat::from_android_bitmap(env, show_bitmap, ncnn::Mat::PIXEL_RGB);


    ncnn::Mat cond = prompt_slover.get_conditioning(positive_prompt);
    ncnn::Mat uncond = prompt_slover.get_conditioning(negative_prompt);

    vector<ncnn::Mat> init_latents = encode_slover.encode(init_image);

    ncnn::Mat sample = diffusion_slover.sampler_img2img(seed, step, cond, uncond, init_latents);

    ncnn::Mat x_samples_ddim = decode_slover.decode(sample);

    x_samples_ddim.to_android_bitmap(env,show_bitmap,ncnn::Mat::PIXEL_RGB);

    cv::Mat image(256, 256, CV_8UC3);
    x_samples_ddim.to_pixels(image.data, ncnn::Mat::PIXEL_RGB);

    return JNI_TRUE;
}

}
