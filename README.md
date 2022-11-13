# Stable Diffusion NCNN

基于c++版ncnn实现的“屎山+盲盒”版stable-diffusion

知乎介绍文章: https://zhuanlan.zhihu.com/p/582552276

## 说明
1. 对于模型的使用，请参照官方stable-diffusion的模型license的说明，这里不在赘述，请自觉遵守
2. 该代码只使用cpu，经过调整，只需要8G内存！！！
3. 感谢nihui的pr，目前出图质量稳定(咒语要写得好)，欢迎尝试

## 一些结果
![image](./resources/result_15_42.png)

![image](./resources/result_15_42_1.png)

![image](./resources/result_15_1668336058.png)

![image](./resources/result_15_1668336279.png)

![image](./resources/result_15_1668336723.png)

![image](./resources/result_15_1668337168.png)

![image](./resources/result_15_1668337577.png)

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
1. 由于目前速度不咋快，因此不分发exe了，大家自己编译吧
2. 从[百度网盘](https://pan.baidu.com/s/1kO8HtTZRcyDbzA32ZzafSQ?pwd=6666)下载三个bin放到对应的assets目录下进行编译
3. 代码里面给了一个简单的测试prompt

## 存在问题
1. 对prompt很敏感，要想出图好，prompt就得写得好
2. 速度较慢，一个step在5~10s不等

## 参考
1. [ncnn](https://github.com/Tencent/ncnn)
2. [opencv-mobile](https://github.com/nihui/opencv-mobile)
3. [stable-diffusion](https://github.com/CompVis/stable-diffusion)
4. [k-diffusion](https://github.com/crowsonkb/k-diffusion)
5. [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)