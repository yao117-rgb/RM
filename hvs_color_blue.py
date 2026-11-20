import cv2
import numpy as np

class HSVThreshold:

    def __init__(self, lower, upper):
        self.lower = np.array(lower)
        self.upper = np.array(upper)

    def get_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self.lower, self.upper)

    
def blue_preprocess_image(img,kernel):
  threshold = HSVThreshold([68,134,59],[122,255,255])
  mark=threshold.get_mask(img)
  kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
  kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
  kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  erosion =  cv2.morphologyEx(mark, cv2.MORPH_CLOSE, kernel)
  dilation= cv2.erode(erosion, kernel1, iterations=1)
  erosion_image = cv2.dilate(dilation, kernel2, iterations=1)
  mj=img.copy()
  contours,hierarchy=cv2.findContours(erosion_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

  light_boxes=[]
  for cnt in contours:
   area = cv2.contourArea(cnt)
   if area < 30:
    continue
   rest=cv2.minAreaRect(cnt)
   box=cv2.boxPoints(rest)
   box = box.astype(int)
   light_boxes.append(box)
   cv2.drawContours(mj, [box], -1, (0,255,0), 2)

  a_box_list=[]
  if len(light_boxes)>=2:
       pts_all=np.vstack([light_boxes[0],light_boxes[1]])
       a_rest=cv2.minAreaRect(pts_all)
       cxr,cyr=a_rest[0]
       wr,hr=a_rest[1]
       ange=a_rest[2]
       side=max(wr,hr)
       sq_a_rest=((cxr,cyr),(side,side),ange)
       sq_a_box=cv2.boxPoints(sq_a_rest)
       sq_a_box=sq_a_box.astype(int)
       a_box_list.append(sq_a_box)

  return light_boxes,a_box_list,erosion_image

import cv2
import numpy as np

def print_multiple_rects_centers(rects_list, img=None):

    all_centers = []

    for i, rect_points in enumerate(rects_list):
        xs = [p[0] for p in rect_points]
        ys = [p[1] for p in rect_points]

        cx = (max(xs) + min(xs)) / 2.0
        cy = (max(ys) + min(ys)) / 2.0

        a_cx = int(cx)
        a_cy = int(cy)

        print(f"矩形{i+1}: x={a_cx:.1f},y={a_cy:.1f}")

        if img is not None:

            cv2.circle(img, (a_cx, a_cy), 4, (0, 0, 255), -1)

            text = f"x={a_cx:.1f},y={a_cy:.1f}"
            cv2.putText(img, text, (a_cx + 5, a_cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        all_centers.append([a_cx, a_cy])

    return all_centers


img =cv2.imread(r"image (2).png",1)
mj=img.copy()
kernel=np.ones((3,3),np.uint8)


light_boxes2,a_box_list2,erosion_image=blue_preprocess_image(img,kernel)


cv2.drawContours(mj, light_boxes2, -1, (0,255,0), 2)
cv2.drawContours(mj, a_box_list2, -1, (0,255,0), 2)


print_multiple_rects_centers(light_boxes2, img=mj)
print_multiple_rects_centers(a_box_list2, img=mj)
cv2.imshow("erosion",erosion_image)
cv2.imshow("mj",mj)
cv2.waitKey(0)
cv2.destroyAllWindows()
