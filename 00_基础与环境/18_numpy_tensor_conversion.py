#学习NumPy数组与pytorch张量的相互转换
#比较共享内存与复制数据，观察修改一个对象是否影响另一个对象

import numpy as np
import torch

numpy_array=np.array(
    [
        [1.0,2.0,3.0],
        [4.0,5.0,6.0]
    ],
    dtype=np.float32
)

shared_tensor=torch.from_numpy(numpy_array)
copied_tensor=torch.tensor(numpy_array)

print("原始 NumPy 数组：")
print(numpy_array)
print("数组形状：", numpy_array.shape)
print("数组数据类型：", numpy_array.dtype)

print("\n转换后的张量：")
print(shared_tensor)
print("张量形状：", shared_tensor.shape)
print("张量数据类型：", shared_tensor.dtype)
print("张量所在设备：", shared_tensor.device)

numpy_array[0, 0] = 100.0

print("\n修改 NumPy 数组后：")
print("共享内存的张量：")
print(shared_tensor)
print("复制数据的张量：")
print(copied_tensor)


shared_tensor[1, 2] = 200.0

print("\n修改共享张量后，NumPy 数组：")
print(numpy_array)

converted_array=shared_tensor.numpy()
converted_array[0,1]=300.0



print("\n修改转换回来的数组后：")
print("共享张量：")
print(shared_tensor)
print("最初的 NumPy 数组：")
print(numpy_array)