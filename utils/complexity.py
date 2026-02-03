from thop import profile
import torch
from torchvision.models import resnet50
import time


def count_parameters(model, trainable_only=False):
    params = sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
    return params

if __name__ == "__main__":

    # 以resnet50为示例模型
    model = resnet50()

    # 参数量parameters
    print("Total params:", count_parameters(model, trainable_only=False))
    print("Trainable params:", count_parameters(model, trainable_only=True))

    # 浮点运算量FLOPs(G), 参数量parameters(M)
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs={flops / 1e9:.2f}G, Params={params / 1e6:.2f}M")

    # 推理时间Time(ms)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    batch_size = 16
    inp = torch.randn(batch_size, 3, 224, 224).to(device)
    # 预热
    for _ in range(10):
        _ = model(inp)
    # 测量
    # start = time.time()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    iter_n = 100
    for _ in range(iter_n):
        _ = model(inp)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)  # 毫秒
    # end = time.time()
    # ms = (end - start) * 1000
    print("avg ms per batch:", ms/iter_n)
    print("avg ms per image:", ms/iter_n / batch_size)