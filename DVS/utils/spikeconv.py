import torch
from torchvision import datasets, transforms

def generate_spike_sequence(image, threshold,pulse_width=1):
    """
    将图片展开并按照像素值累加，生成脉冲序列。
    :param image: 输入图片 (28, 28)
    :param threshold: 阈值
    :return: 脉冲序列 (1D 张量，值为 0 或 1)
    """
    # 将图片展开为一维序列
    flattened_image = image.flatten()

    # 初始化累加器和脉冲序列
    accumulator = 0
    spike_sequence = []
    threshold1 = threshold //pulse_width

    # 遍历每个像素值
    for pixel_value in flattened_image:
        accumulator += pixel_value
        if accumulator >= threshold1:
            multiple = int(accumulator // threshold1)
            multiple = min(multiple, pulse_width)
            spike_sequence.extend([1.0] * multiple + [0.0] * (pulse_width - multiple))
            accumulator -= (threshold1*multiple)
        else:
            spike_sequence.extend([0]*pulse_width)  # 不生成脉冲
    return torch.tensor(spike_sequence, dtype=torch.float32)

def main():
    # 示例图片 (28x28)
    image = torch.randint(0, 256, (3, 3)).float()  # 随机生成一张 28x28 的图片

    print(image)

    # 设置阈值
    threshold = 256

    # 生成脉冲序列
    spike_sequence = generate_spike_sequence(image, threshold, 4)
    print(spike_sequence)
    print(spike_sequence.shape)  # 输出: torch.Size([784])


if __name__ == '__main__':
    main()

