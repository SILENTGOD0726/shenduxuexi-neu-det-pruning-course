"""Run the trained detector on local images and save visible boxes plus JSON."""
import bootstrap
import argparse
import json
from pathlib import Path
import cv2
from PIL import Image,ImageDraw,ImageFont
from evaluate import predict
from experiment import load_model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True,help='A local image or image directory')
    parser.add_argument('--checkpoint',type=Path,help='A trusted project-generated checkpoint')
    parser.add_argument('--save-dir',type=Path,default=bootstrap.ROOT/'predictions')
    parser.add_argument('--confidence',type=float,default=.25)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    if not 0<args.confidence<1:parser.error('confidence must be between zero and one')
    if args.checkpoint is None:
        selected=json.loads((bootstrap.ROOT/'results/selected_model.json').read_text(encoding='utf-8'))
        args.checkpoint=bootstrap.ROOT/selected['checkpoint']
    bootstrap.configure_torch(42)
    model=load_model(args.checkpoint)
    extensions={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
    files=sorted(p for p in args.source.glob('*') if p.suffix.lower() in extensions) if args.source.is_dir() else [args.source]
    if args.limit:files=files[:args.limit]
    if not files:raise FileNotFoundError('No supported local images found')
    args.save_dir.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    for source in files:
        image=cv2.imread(str(source))
        if image is None:raise ValueError(f'Cannot decode {source}')
        predictions=predict(model,image,args.confidence)
        height,width=image.shape[:2]
        visible=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(visible)
        detections=[]
        for x1,y1,x2,y2,confidence,cls in predictions:
            box=[float(x1/320*width),float(y1/320*height),float(x2/320*width),float(y2/320*height)]
            name=str(model.names[int(cls)])
            draw.rectangle(box,outline='#E24A33',width=2)
            draw.text((box[0],max(0,box[1]-17)),f'{name} {confidence:.2f}',fill='#E24A33',font=font)
            detections.append({'class':name,'confidence':float(confidence),'xyxy_original_pixels':box})
        visible.save(args.save_dir/(source.stem+'_detected.png'))
        record={'source_name':source.name,'original_shape':[height,width],
                'preprocessing':'Resize directly to320x320, matching square NEU-DET experiments',
                'confidence_threshold':args.confidence,'detections':detections}
        (args.save_dir/(source.stem+'_detected.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(source.name,'detections=',len(detections),flush=True)


if __name__=='__main__':
    main()
