# Stable Diffusion NCNN

基于c++版ncnn实现的“屎山+盲盒”版stable-diffusion

## 说明
1. 对于模型的使用，请参照官方stable-diffusion的模型license的说明，这里不在赘述，请自觉遵守
2. 目前的实现可用，但非常不稳定，结果存在“盲盒”性，大多数情况下都会返回无意义的结果
3. 该代码只使用cpu，需要约19G的内存，请最少为它准备20G的内存
4. 目前测试出来的结果是，step在5以内，结果都不会崩，但图的质量一般

## 一些结果
![image](./resources/result_2_42.png)

step:2 seed:42，由于目前采样器不稳定，加step反而会崩，将就看，但能说明代码是没有大问题的

![image](./resources/result_5_42.png)

step:5 seed:42

![image](./resources/result_5_1668147633.png)

step:5 seed:1668147633

## 实现细节
1. stable-diffusion就三个步骤：
    1. CLIP做文本的嵌入
    2. 使用采样算法进行迭代采样
    3. decode解码采样结果到最终图像
2. 模型细节：
    1. 模型：Naifu（懂得都懂）
    2. 采样：k-diffusion的euler ancestral
    3. 分辨率：512*512
    4. 降噪器：CFGDenoiser和CompVisDenoiser
    4. prompt：支持positive和negative

## 代码细节
1. 由于目前结果不稳定，因此不分发exe了，大家自己编译吧
2. 从[百度网盘](https://pan.baidu.com/s/1kO8HtTZRcyDbzA32ZzafSQ?pwd=6666)下载三个bin放到对应的assets目录下进行编译
3. 代码里面给了一个简单的测试prompt

## 存在问题
1. 采样算法不稳定，目前用的“euler ancestral”十分不稳定，导致几乎90%的结果都是无意义的
2. 有很多的magic number，调一调结果又会翻天覆地
3. 慢，需要gpu的版本

## 可优化点
1. 使用更稳定的采样算法（最迫切需求）
2. 使用更好的随机数策略

## 参考
1. [ncnn](https://github.com/Tencent/ncnn)
2. [opencv-mobile](https://github.com/nihui/opencv-mobile)
3. [stable-diffusion](https://github.com/CompVis/stable-diffusion)
4. [k-diffusion](https://github.com/crowsonkb/k-diffusion)
5. [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)