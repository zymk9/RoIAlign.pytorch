import numpy as np
import torch
from torch.autograd import Variable, gradcheck

from roi_align.roi_align import roi_align_3d


def to_varabile(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


def fn(x):
    is_cuda = False

    boxes_data = np.asarray([[0, 0, 0, 2, 2, 2]], dtype=np.float32)
    box_index_data = np.asarray([0], dtype=np.int32)

    boxes = to_varabile(boxes_data, requires_grad=False, is_cuda=is_cuda)
    box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=is_cuda)

    y = roi_align_3d(
        x, boxes, box_index, 3, 3, 3,
    )

    return y


# the data you want
is_cuda = False
image_data = np.tile(np.arange(7, dtype=np.float32), 7).reshape(7, 7)
image_data = [image_data + i for i in range(7)]
image_data = np.asarray(image_data)

image_data = image_data[np.newaxis, np.newaxis]
boxes_data = np.asarray([[0, 0, 0, 3, 3, 3]], dtype=np.float32)
box_index_data = np.asarray([0], dtype=np.int32)

image_torch = to_varabile(image_data, requires_grad=True, is_cuda=is_cuda)
boxes = to_varabile(boxes_data, requires_grad=False, is_cuda=is_cuda)
box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=is_cuda)

print(roi_align_3d(
    image_torch, boxes, box_index, 3, 3, 3,
))

input = torch.randn(1, 1, 3, 3, 3, dtype=torch.float, requires_grad=True)
test = gradcheck(fn, input)
print(test)
