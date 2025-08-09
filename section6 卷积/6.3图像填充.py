import torch
import torch.nn as nn

def comp_conv2d(X, conv2d):
    """
    将二维输入X扩展为(1,1,H,W)，使用指定的conv2d层计算输出，
    并去掉批量和通道维度，返回二维输出。
    """
    X = X.unsqueeze(0).unsqueeze(0)  # (H,W) -> (1,1,H,W)
    Y = conv2d(X)
    return Y.squeeze(0).squeeze(0)  # (1,1,H_out,W_out) -> (H_out,W_out)

def calc_output_shape(H_in, W_in, kernel_size, padding_total, stride):
    """
    使用公式(6.3.2)计算输出形状
    参数:
        H_in, W_in: 输入高宽
        kernel_size: 卷积核大小，假设高宽相同
        padding_total: padding总量，即上下或左右padding之和
        stride: 步幅
    返回:
        (H_out, W_out)
    """
    H_out = (H_in + padding_total - kernel_size) // stride + 1
    W_out = (W_in + padding_total - kernel_size) // stride + 1
    return H_out, W_out

def example_padding_and_stride():
    """
    示例不同kernel_size、padding、stride对输出形状的影响，
    并对比PyTorch计算结果与公式计算结果。
    """
    H_in, W_in = 8, 8
    X = torch.randn(H_in, W_in)

    examples = [
        # (kernel_size, padding_per_side, stride)
        (3, 0, 1),
        (3, 1, 1),
        (3, 2, 1),
        (3, 1, 2),
        (5, 0, 1),
        (5, 2, 1),
        (5, 2, 2),
        (7, 3, 1),
        (7, 3, 2),
    ]

    for kernel_size, pad_side, stride in examples:
        padding_total = pad_side * 2
        conv2d = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=pad_side, stride=stride)
        Y = comp_conv2d(X, conv2d)
        H_out_formula, W_out_formula = calc_output_shape(H_in, W_in, kernel_size, padding_total, stride)
        print(f"kernel_size={kernel_size}, padding_per_side={pad_side}, stride={stride}")
        print(f"PyTorch output shape: {Y.shape}")
        print(f"Formula output shape: ({H_out_formula}, {W_out_formula})")
        print('-'*40)

def main():
    example_padding_and_stride()

if __name__ == "__main__":
    main()
