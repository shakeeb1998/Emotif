from __future__ import unicode_literals

import base64
import io
import json

import frappe
import cv2
import numpy as np
from PIL import Image
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import emotif.ESRGAN.RRDBNet_arch as arch

@frappe.whitelist(allow_guest=True)
def add_feedback(name,message,subject,email):


    # doc.email=email
    #
    # doc.name=name
    # doc.message=message
    # doc.subject=subject
    # doc.insert(ignore_permissions=True,ignore_if_duplicate=True)


    print('done inserting')
    return {"resp": name}


@frappe.whitelist(allow_guest=True)
def test(name,message,subject,email):
    feedback = frappe.get_doc({
         'doctype':"Feedback",

        "name1":name,
        "email":email,
        "message":message,
        "subject":subject

    }


     )

    #
    feedback.insert(
        ignore_permissions=True,  # ignore write permissions during insert
        ignore_links=True,  # ignore Link validation in the document
        ignore_if_duplicate=True,  # dont insert if DuplicateEntryError is thrown
        ignore_mandatory=True  # insert even if mandatory fields are not set
    )
    frappe.db.commit()


@frappe.whitelist(allow_guest=True)
def increase_resolution():
    # def stringToImage(base64_string):
    #     imgdata = base64.b64decode(base64_string)
    #     return Image.open(io.BytesIO(imgdata))
    #
    # # convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
    # def toRGB(image):
    #     return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    #
    # img = toRGB(stringToImage(img_data))
    # print ('done')
    # print(img)

    print('shakeeb')
    # print(json.loads(frappe.request.data))
    resp=json.loads(frappe.request.data)

    def stringToImage(base64_string):
        imgdata = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(imgdata))

    # convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
    def toRGB(image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    img = toRGB(stringToImage(resp['b'].replace("data:image/png;base64,","")))

    def SuperRes(path,img):
        model_path = '/home/frappe/frappe-bench/apps/emotif/emotif/ESRGAN/models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
        print('device loading')
        device = torch.device('cpu')  # if you want to run on CPU, change 'cuda' -> cpu
        print('device loaded')
        model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        model = model.to(device)

        print('Model path {:s}. \nTesting...'.format(model_path))

        # img = cv2.imread(path, cv2.IMREAD_COLOR)
        print(img.shape)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        print("writing result")
        # retval, buffer = cv2.imencode('.jpg', img)
        # jpg_as_text = base64.b64encode(buffer)

        cv2.imwrite('gitex/public/files/shakku.png', output)
        print(output.shape)
        return output

    imgb64=SuperRes('/home/frappe/frappe-bench/apps/emotif/emotif/ESRGAN/LR/baboon.png',img)
    retval, buffer = cv2.imencode('.png', imgb64)
    print('here1')
    jpg_as_text = base64.b64encode(buffer)
    print('here2')
    help = str(jpg_as_text.decode())
    print('here3')
    help = 'data:image/png;base64,' + help
    return help


@frappe.whitelist(allow_guest=True)
def test1():
    def SuperRes(path):
        model_path = '/home/frappe/frappe-bench/apps/emotif/emotif/ESRGAN/models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
        print('device loading')
        device = torch.device('cpu')  # if you want to run on CPU, change 'cuda' -> cpu
        print('device loaded')
        model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        model = model.to(device)

        print('Model path {:s}. \nTesting...'.format(model_path))

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        print(img.shape)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        print("writing result")
        cv2.imwrite('/home/frappe/frappe-bench/apps/emotif/emotif/ESRGAN/LR/result.png', output)
        print(output.shape)
        return output

    SuperRes('/home/frappe/frappe-bench/apps/emotif/emotif/ESRGAN/LR/baboon.png')
    # print(len(img_data))
    print('shakeeb pot')
    return 'jhh'

@frappe.whitelist(allow_guest=True)
def test2():
    return 'file updated  iuoioid'