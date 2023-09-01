import onnxruntime
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import common


def vec2Text(vec):
    vec = torch.argmax(vec, dim=1)  # 把为1的取出来
    text = ''
    for i in vec:
        text += common.captcha_array[i]
    return text


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    filepath = 'code.jpg'
    import requests
    res = requests.get("https://example.com/prod-api/captchaImage", verify=False)
    print(f'resp {res.text}')
    import json
    import base64
    obj = json.loads(res.text)
    b64str = obj['img']
    img_content = base64.b64decode(b64str, validate=True)
    with open(filepath, "wb") as f:
        f.write(img_content)

    onnxFile = 'mathcode.onnx'

    img = Image.open(filepath)
    trans = transforms.Compose([
        transforms.Resize((60, 160)),
        # transforms.Grayscale(),
        transforms.ToTensor()
    ])
    img_tensor = trans(img)
    img_tensor = img_tensor.reshape(1, 3, 60, 160)  # 1张图片 1 灰色
    ort_session = onnxruntime.InferenceSession(onnxFile)

    modelInputName = ort_session.get_inputs()[0].name
    # onnx 网络输出
    onnx_out = ort_session.run(None, {modelInputName: to_numpy(img_tensor)})
    onnx_out = torch.tensor(np.array(onnx_out))
    onnx_out = onnx_out.view(-1, common.captcha_array.__len__())
    captcha_text = vec2Text(onnx_out)
    import re
    captchaRegex = re.compile(r"(?P<oprand1>\d+)(?P<operator>[+×÷-])(?P<oprand2>\d+)(?P<equalSign>=)(?P<questionMark>[\uFF1F\x3f]{1})", re.IGNORECASE | re.MULTILINE | re.UNICODE)
    re_ret = re.match(captchaRegex, captcha_text)
    op1 = re_ret.group("oprand1")
    op2 = re_ret.group("oprand2")
    opr = re_ret.group("operator")
    if opr == '+':
        result = int(op1) + int(op2)
    elif opr == '-':
        result = int(op1) - int(op2)
    elif opr == '×':
        result = int(op1) * int(op2)
    elif opr == '÷':
        result = int(op1) // int(op2)
    else:
        raise f"unknown operator {opr}"
    print(vec2Text(onnx_out), result)
