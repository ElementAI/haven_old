import cv2 
from . import haven_utils as hu
import numpy as np 
from PIL import Image


def overlay_pil(image, mask):
    image = np.array(image)
    mask = np.array(mask)
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[1:]

    # polygons = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1][0]
    red = np.zeros(image.shape, dtype='uint8')
    red[:,:,2] = 255
    alpha = 0.5
    for o in obj_ids:
        ind = mask==o
        image[ind] = image[ind] * alpha + red[ind] * (1-alpha)
        pos = np.where(ind)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        image = cv2.rectangle(image, (xmin, ymin), 
                                                 (xmax, ymax), 
                                                 color=(0,255,0), 
                                                 thickness=2) 
    # Image.fromarray(image).save('/mnt/datasets/public/issam/prototypes/dais/overlay.png')                                                 
    return Image.fromarray(image)
    
def resize_points(points, h, w):
    points = points.squeeze()
    h_old, w_old = points.shape
    y_list, x_list = np.where(points.squeeze())

    points_new = np.zeros((h, w))

    for y, x in zip(y_list, x_list):
        y_new = int((y/h_old) * h)
        x_new = int((x/w_old) * w)
        points_new[y_new, x_new ] = 1

    return points_new

def gray2cmap(gray, cmap="jet", thresh=0):
    # Gray has values between 0 and 255 or 0 and 1
    gray = hu.t2n(gray)
    gray = gray / max(1, gray.max())
    gray = np.maximum(gray - thresh, 0)
    gray = gray / max(1, gray.max())
    gray = gray * 255

    gray = gray.astype(int)
    #print(gray)

    from pylab import get_cmap
    cmap = get_cmap(cmap)

    output = np.zeros(gray.shape + (3, ), dtype=np.float64)

    for c in np.unique(gray):
        output[(gray == c).nonzero()] = cmap(c)[:3]

    return hu.l2f(output)

def text_on_image(text, image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,40)
    fontScale              = 0.8
    fontColor              = (1,1,1)
    lineType               = 1
    # img_mask = skimage.transform.rescale(np.array(img_mask), 1.0)
    # img_np = skimage.transform.rescale(np.array(img_points), 1.0)
    img_np = cv2.putText(image, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness=2
        # lineType
        )
    return img_np

def bbox_on_image(bbox_xyxy, image):
    image = hu.f2l(image.squeeze())
    image_uint8 = (image*254).astype("uint8").copy()
    H, W, _ = image.shape
    for bb in bbox_xyxy:
        x1, y1, x2, y2 = bb
        start_point = (int(x1*W), int(y1*H), ) 
        end_point = ( int(x2*W), int(y2*H),)
        
        # Blue color in BGR 
        color = (255, 0, 0) 
        
        # Line thickness of 2 px 
        thickness = 2
        
        # Using cv2.rectangle() method 
        # Draw a rectangle with blue line borders of thickness of 2 px 
        image_uint8 = cv2.rectangle(image_uint8, start_point, end_point, color, thickness) 
    # hu.save_image("/mnt/datasets/public/issam/prototypes/wscl/tmp.jpg", image_uint8)
    return image_uint8 / 255.

def points_on_image(y_list, x_list, image, radius=3):
    image = hu.f2l(image.squeeze())
    image_uint8 = (image*254).astype("uint8").copy()
    H, W, _ = image.shape
    for y, x in zip(y_list, x_list):
        
        # Blue color in BGR 
        color = (255, 0, 0) 
        
        # Line thickness of 2 px 
        thickness = 5
        # Using cv2.rectangle() method 
        # Draw a rectangle with blue line borders of thickness of 2 px 
        image_uint8 = cv2.circle(image_uint8, (x,y), radius, color, thickness) 

        start_point = (x-radius*2, y-radius*2) 
        end_point = (x+radius*2, y+radius*2) 
        thickness = 2
        color = (0, 255, 0) 
        
        image_uint8 = cv2.rectangle(image_uint8, start_point, end_point, color, thickness) 

    # hu.save_image("/mnt/datasets/public/issam/prototypes/wscl/tmp.jpg", image_uint8)
    return image_uint8 / 255.
