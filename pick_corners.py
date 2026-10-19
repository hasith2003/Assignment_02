"""Optional GUI: click TL, TR, BR, BL corners; Enter saves, R resets, Esc cancels."""
import argparse, json
from pathlib import Path
import cv2 as cv
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',choices=['pair1_background.jpg','pair2_background.jpg'])
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    image=cv.imread(str(root/'my_images'/args.image))
    if image is None:raise FileNotFoundError(args.image)
    scale=min(1.,1100/image.shape[1],800/image.shape[0])
    preview=cv.resize(image,None,fx=scale,fy=scale)
    points=[]
    title='Click TL, TR, BR, BL | Enter save | R reset | Esc cancel'
    cv.namedWindow(title)
    def click(event,x,y,flags,param):
        if event==cv.EVENT_LBUTTONDOWN and len(points)<4:
            points.append([round(x/scale,2),round(y/scale,2)])
    cv.setMouseCallback(title,click)
    try:
        while True:
            frame=preview.copy()
            for i,p in enumerate(points):
                q=tuple(np.round(np.array(p)*scale).astype(int))
                cv.circle(frame,q,5,(0,0,255),-1)
                cv.putText(frame,str(i+1),q,cv.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
            cv.imshow(title,frame); key=cv.waitKey(30)&255
            if key==27:return
            if key in (ord('r'),ord('R')):points.clear()
            if key in (10,13) and len(points)==4:
                target=root/'my_images/corners.json'
                saved=json.loads(target.read_text()) if target.exists() else {}
                saved[args.image]=points
                target.write_text(json.dumps(saved,indent=2))
                print('Saved',target);return
    finally:cv.destroyAllWindows()

if __name__=='__main__':main()

