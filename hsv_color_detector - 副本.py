import cv2
import numpy as np


class HSVThreshold:

    def __init__(self, lower, upper):
        self.lower = np.array(lower)
        self.upper = np.array(upper)

    def get_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self.lower, self.upper)


def red_preprocess_image(img,kernel):
  threshold = HSVThreshold([141,0,198],[180,255,255])
  mark=threshold.get_mask(img)
  kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
  erosion_image=cv2.erode(mark,kernel,2)
  mj=img.copy()
  contours,hierarchy=cv2.findContours(erosion_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  light_boxes=[]
  a_box_list=[]
  light_centers=[]
  for cnt in contours:
   area = cv2.contourArea(cnt)
   if area < 1:
    continue
   rest=cv2.minAreaRect(cnt)
   box=cv2.boxPoints(rest)
   box = box.astype(int)
   light_boxes.append(box)
   center=rest[0]
   light_centers.append(center)

  n_light=len(light_boxes)
  used=[False]*n_light

  for i in range(n_light):
       if used[i]:
         continue
       for j in range(i+1,n_light):
        if used[j]:
          continue
        box1=light_boxes[i]
        box2=light_boxes[j]
        c1=light_centers[i]
        c2=light_centers[j]
        dx=c2[0]-c1[0]
        dy=c2[1]-c1[1]
        dist=np.sqrt(dx**2+dy**2)
        if 40<dist<110:
         pts_all=np.vstack([box1,box2])
         a_rest=cv2.minAreaRect(pts_all)
         cxr,cyr=a_rest[0]
         wr,hr=a_rest[1]
         ange=a_rest[2]
         side=max(wr,hr)
         sq_a_rest=((cxr,cyr),(side,side),ange)
         sq_a_box=cv2.boxPoints(sq_a_rest)
         sq_a_box=sq_a_box.astype(int)
         cv2.drawContours(mj, [sq_a_box], -1, (0,255,0), 2)
         a_box_list.append(sq_a_box)
         used[i]=True
         used[j]=True
         break

  return light_boxes,a_box_list




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

light_boxes,a_box_list,erosion_image=red_preprocess_image(img,kernel)

cv2.drawContours(mj, light_boxes, -1, (0,0,255), 2)
cv2.drawContours(mj, a_box_list, -1, (0,0,255), 2)

print_multiple_rects_centers(light_boxes, img=mj)
print_multiple_rects_centers(a_box_list, img=mj)
cv2.imshow("erosion",erosion_image)
cv2.imshow("mj",mj)
cv2.waitKey(0)
cv2.destroyAllWindows()
