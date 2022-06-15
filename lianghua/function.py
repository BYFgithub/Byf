from torch.autograd import Function

#定义伪量化
class FakeQuantize(Function):

    @staticmethod   #静态方法
    def forward(ctx, x, qparam):    #前向传播
        x = qparam.quantize_tensor(x)
        x = qparam.dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):   #反向传播计算梯度
        return grad_output, None