import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from datetime import datetime
import torch
import cv2
from PIL import Image
from io import BytesIO
import base64
import traceback
import numpy as np
import os
import re
import glob
import shutil
from models import load_textdetector_model, dispatch_textdetector,  dispatch_inpainting, load_inpainting_model, OCRMIT48pxCTC

from openai import OpenAI

with open('api_key.txt', 'r') as f:
    key = f.readline().strip()
print("OPENAI KEY", key)

MODEL="gpt-4o"
openai_client = OpenAI(api_key=key)

use_cuda = torch.cuda.is_available()

load_inpainting_model(use_cuda, 'default')

def chatgpt(bb_list, img):
    content = [
        {   
            "type": "text", 
            "text": f"Translate texts inside bellow images"
        },
    ]
    for bbox in bb_list:
        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = img.shape[1] if xmax >  img.shape[1] else xmax
        ymax = img.shape[0] if ymax >  img.shape[0] else ymax

        frame = img[int(ymin):int(ymax), int(xmin):int(xmax), :]

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img2str(frame).decode('utf-8')}"
                }
            }
        )

    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant, You can translate texts from images into English. The inputs will be multiple images with text on it. You have to only provide the answers respectively with the structure ascending_index_number_from_0_to_maximum_number_of_images # original text # the translated text. Do not adding more newline character. If there is no text to translate please use the sentence '...' as the translated text. If the original text is English, please use original text as translated text"},
            {"role": "user", "content": content}
        ],
        temperature=0.0,
    )

    return response.choices[0].message.content

def infer(img, imgb64, foldername, filename, lang, tech):
    separator = '@@@@@-mangatool-@@@@@'
    re_str = r'@@@@@-mangatool-@@@@@'
    mask, mask_refined, blk_list = dispatch_textdetector(img, use_cuda)
    # ocr.ocr_blk_list(img, blk_list)
    torch.cuda.empty_cache()

    ez_result = reader.readtext(img)

    for i, blk in enumerate(ez_result):
        xmin, ymin = blk[0][0]
        xmax, ymax = blk[0][2]
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = img.shape[1] if xmax >  img.shape[1] else xmax
        ymax = img.shape[0] if ymax >  img.shape[0] else ymax
        mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 255

    mask = cv2.dilate((mask > 170).astype('uint8')*255, np.ones((5,5), np.uint8), iterations=5)
    kernel = np.ones((9,9), np.uint8)
    mask_refined = cv2.dilate(mask_refined, kernel, iterations=2)

    img_inpainted =  dispatch_inpainting(True, False, use_cuda, img, ((mask + mask_refined) > 0).astype('uint8')*255, 2048)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    filter_mask = np.zeros_like(mask)
    for i, blk in enumerate(blk_list):
        xmin, ymin, xmax, ymax = blk.xyxy
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = img.shape[1] if xmax >  img.shape[1] else xmax
        ymax = img.shape[0] if ymax >  img.shape[0] else ymax
        filter_mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1

    for i, blk in enumerate(ez_result):
        xmin, ymin = blk[0][0]
        xmax, ymax = blk[0][2]
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = img.shape[1] if xmax >  img.shape[1] else xmax
        ymax = img.shape[0] if ymax >  img.shape[0] else ymax
        filter_mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1

    bboxes = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for i, contour in enumerate(contours):
        bbox = cv2.boundingRect(contour)
        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = img.shape[1] if xmax >  img.shape[1] else xmax
        ymax = img.shape[0] if ymax >  img.shape[0] else ymax
        # index = np.bincount(np.ravel(filter_mask[int(ymin):int(ymax), int(xmin):int(xmax)])).argmax()
        index = np.sum(filter_mask[int(ymin):int(ymax), int(xmin):int(xmax)])
        if index > 0:
            bboxes.append(list(bbox))
            filter_mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 0
    
    final_text = []
    final_bboxes = None
    for i, bbox in enumerate(bboxes):
        xmin, ymin, w, h = bbox
        final_bboxes = np.concatenate((final_bboxes, np.array([[xmin, ymin, w, h]]))) if not final_bboxes is None else np.array([[xmin, ymin, w, h]])
    
    response = chatgpt(bboxes, img)

    print(response)

    for sentence in response.split("\n"):
        if len(sentence.strip()) > 0:
            final_text.append(sentence.strip().split("#")[2])
    
    text = separator.join(final_text)
    text_ref = separator.join(final_text)
    text_ref = re.sub(re_str, '', text_ref)

    if not text_ref == "" and final_bboxes is not None:
        if not os.path.exists('output/'):
            os.mkdir('output/')
        if not os.path.exists('output/' + foldername + "/"):
            os.mkdir('output/' + foldername + "/")
        cv2.imwrite('output/' + foldername + '/' + filename.split('.')[0] + '.png', img_inpainted.astype('uint8'))
        with open('output/' + foldername + "/" + filename.split('.')[0] + '_text.txt', 'w+', encoding="utf-8") as f:
            f.write(text)
        f.close()
        np.savetxt('output/' + foldername + '/' + filename.split('.')[0] + '_bbox.txt', final_bboxes.astype(int))
        np.savetxt('output/' + foldername + '/' + filename.split('.')[0] + '_order.txt', np.array(range(len(final_bboxes))).astype(int), fmt="%d")

    return text, np.array2string(final_bboxes, precision=2, separator=',')
        
def sub(img, imgb64, foldername, filename, lang='jp', tech="MangaOCR"):
    img = cv2.cvtColor(np.array(img).astype('uint8'), cv2.COLOR_RGB2BGR)
    res =  infer(img, imgb64, foldername, filename, lang, tech)
    return res


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

async def read_image(request):
    form = await request.form()
    file = await form["file"].read()
    image = Image.open(BytesIO(file))
    return image

def img2str(result):
    _, buffer = cv2.imencode('.png', result)
    img_str = base64.b64encode(buffer)
    return img_str

def img2base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str


@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.get("/text/{foldername}/{filename}")
async def text_file(foldername, filename):
    with open('output/' + foldername + "/" + filename + '_text.txt', 'r', encoding="utf-8") as f:
        lines = f.readlines()
    res = lines[0].split('@@@@@-mangatool-@@@@@')
    return {"result": res}

@app.get("/bbox/{foldername}/{filename}")
async def bbox_file(foldername, filename):
    bbox = np.loadtxt('output/' + foldername + "/" + filename + "_bbox.txt")
    if bbox.ndim == 1:
        bbox = np.array([bbox])
    return {"result": bbox.tolist()}

@app.get("/order/{foldername}/{filename}")
async def order_file(foldername, filename):
    order = np.loadtxt('output/' + foldername + "/" + filename + "_order.txt")
    result = order.tolist()
    if isinstance(order.tolist(), int) or isinstance(order.tolist(), float):
        result = [order.tolist()]
    if result[0] == -1:
        result = []
    return {"result": result}

@app.get("/folderlist")
async def folderlist():
    lists = os.listdir("output")
    times = []
    counts = []
    for file in lists:
        counts.append(len(glob.glob("output/" + file + "/*_bbox.txt")))
        times.append(os.path.getmtime('output/' + file))
    return {"file_list": lists, "times": times, "counts": counts}

@app.post("/update/{foldername}/{filename}")
async def update_file(request: Request, foldername, filename):
    payload = await request.json()
    try:
        if 'order' in payload:
            if len(payload['order']) == 0:
                payload['order'] = ['-1']
            order = [int(i) for i in payload['order']]
            np.savetxt('output/' + foldername + '/' + filename + '_order.txt', np.array(order).astype(int), fmt="%d")
        if 'text' in payload:
            separator = '@@@@@-mangatool-@@@@@'
            with open('output/' + foldername + "/" + filename + '_text.txt', 'w+', encoding="utf-8") as f:
                f.write(separator.join(payload['text']))
        return {"message": "SUCCESS"}
    except:
        print(traceback.format_exc())
        return JSONResponse(content={"message": "FAILURE"}, status_code=500)

@app.post('/delete/{foldername}')
async def delete(foldername):
    shutil.rmtree(os.path.join("output", foldername))
    return {"message": "SUCCESS"}

@app.post('/generate/{foldername}')
async def generate(foldername):
    separator = '@@@@@-mangatool-@@@@@'
    for file in glob.glob("output/" + foldername + "/*_text.txt"):
        with open(file, 'r', encoding="utf8") as f:
            lines = f.readlines()
        order = np.loadtxt(file[:-9] + '_order.txt').astype(int)
        result = order.tolist()
        if isinstance(order.tolist(), int) or isinstance(order.tolist(), float):
            result = [order.tolist()]
        if result[0] == -1:
            result = []
        order = result
        if not os.path.exists('output/' + foldername + "/final/"):
            os.mkdir('output/' + foldername + "/final/")
        with open('output/' + foldername + "/final/" + (file.split('\\')[-1])[:-9] + '.txt', 'w+', encoding="utf8") as f1:
            lines_ = lines[0].split(separator)
            new_lines = [lines_[int(i)] for i in order]
            f1.write('\n'.join(new_lines))
    return {"message": "SUCCESS"}

@app.post('/scan')
async def sub_(request: Request):
    form = await request.form()
    image = await read_image(request)
    param = [image, img2base64(image), 'example', 'test', 'jp', 'MangaOCR']
    keys = ['image', 'imgb64', 'foldername', 'filename', 'lang', 'tech']
    for i, key in enumerate(keys[2:]):
        if key in form:
            param[i+2] = form[key]
        else:
            return JSONResponse(content={"message": "MISSING PARAM " + key}, status_code=400)
    
    try:
        texts, bbs = sub(*tuple(param))
        sub_text = ""
        for text, bb in zip(texts, bbs):
            sub_text += bb + '\n' + text
        return {"message": "SUCCESS", "sub_text": sub_text}
    except Exception:
        print(traceback.format_exc())
        return JSONResponse(content={"message": "FAILURE"}, status_code=500)

uvicorn.run(app, host='0.0.0.0', port=8000)