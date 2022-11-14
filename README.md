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

## ONNX模型

这里把stable diffusion用到的三个onnx模型放出来，方便大家做一些有意思的工作，模型也在上面的百度网盘链接里面

### 使用声明
1. 请自觉遵守的stable diffusion模型的协议，不要用于非法用途！
2. 如果你用这些onnx做了开源项目，还请来踢一踢我，我去给大佬捧个人场

### 使用说明
一共有三个模型，按照使用顺序依次为：FrozenCLIPEmbedder、UNetModel、AutoencoderKL。使用方法可以参考ncnn的代码。但ncnn中使用的模型，相比于onnx模型，还额外合并了一些辅助计算用以加速整体代码，下面是ncnn模型和onnx模型的输入输出对齐说明：

1. FrozenCLIPEmbedder
```
ncnn输入输出: token, multiplier, cond, conds
onnx输入输出: onnx::Reshape_0, 2271

z = onnx(onnx::Reshape_0=token)
origin_mean = z.mean()
z *= multiplier
new_mean = z.mean()
z *= origin_mean / new_mean
conds = torch.concat([cond,z], dim=-2)
```
2. UNetModel
```
ncnn输入输出: in0, in1, in2, c_in, c_out, outout
onnx输入输出: x, t, cc, out

outout = in0 + onnx(x=in0 * c_in, t=in1, cc=in2) * c_out
```
3. AutoencoderKL模型ncnn与onnc一致，直接使用即可

## 参考
1. [ncnn](https://github.com/Tencent/ncnn)
2. [opencv-mobile](https://github.com/nihui/opencv-mobile)
3. [stable-diffusion](https://github.com/CompVis/stable-diffusion)
4. [k-diffusion](https://github.com/crowsonkb/k-diffusion)
5. [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)