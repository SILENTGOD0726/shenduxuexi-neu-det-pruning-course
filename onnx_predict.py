"""Predict local images with ONNX Runtime CPU; no CUDA inference needed."""
import onnx_bootstrap
import bootstrap
import argparse
import ast
import json
from pathlib import Path
import cv2
import torch
from PIL import Image,ImageDraw,ImageFont
from ultralytics.yolo.utils.ops import non_max_suppression
from onnx_runtime import prepare,session


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=bootstrap.ROOT/'.supplement/onnx/seed42_control.onnx')
    parser.add_argument('--limit',type=int,default=6)
    parser.add_argument('--save-dir',type=Path,default=bootstrap.ROOT/'predictions/onnx_cpu')
    args=parser.parse_args();torch.set_num_threads(4)
    runtime=session(args.model)
    names=ast.literal_eval(runtime.get_modelmeta().custom_metadata_map['names'])
    paths=sorted(p for p in args.source.glob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}) if args.source.is_dir() else [args.source]
    if args.limit:paths=paths[:args.limit]
    if not paths:raise FileNotFoundError('No images found')
    args.save_dir.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    for path in paths:
        image=cv2.imread(str(path))
        if image is None:raise ValueError(f'Cannot decode {path}')
        output=runtime.run(['predictions'],{'images':prepare(image)})[0]
        prediction=non_max_suppression(torch.from_numpy(output),conf_thres=.25,iou_thres=.7,max_det=100)[0].numpy()
        h,w=image.shape[:2];visible=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(visible)
        detections=[]
        for x1,y1,x2,y2,confidence,cls in prediction:
            box=[float(x1/320*w),float(y1/320*h),float(x2/320*w),float(y2/320*h)]
            label=str(names[int(cls)]);draw.rectangle(box,outline='#E24A33',width=2)
            draw.text((box[0],max(0,box[1]-17)),f'{label} {confidence:.2f}',font=font,fill='#E24A33')
            detections.append({'class':label,'confidence':float(confidence),'xyxy_original_pixels':box})
        visible.save(args.save_dir/(path.stem+'_onnx.png'))
        (args.save_dir/(path.stem+'_onnx.json')).write_text(json.dumps({'source':path.name,'provider':runtime.get_providers(),
            'device':'CPU','input':[1,3,320,320],'confidence':.25,'detections':detections},indent=2),encoding='utf-8')
        print('CPU_ONNX',path.name,'detections=',len(detections),flush=True)


if __name__=='__main__':main()
