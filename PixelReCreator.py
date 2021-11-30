
import PIL
import matplotlib.pyplot as plt

import cv2
import numpy as np

import math

import sys

_bar_len = 100
_last_prog = _bar_len
_last_count = 0
def progress(count, total, status=''):
    
    filled_len = int(np.floor(_bar_len * count / float(total)))

    global _last_prog
    if _last_prog == filled_len : return
    _last_prog = filled_len

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (_bar_len - filled_len)

    sys.stdout.write('[%s] %s%s - %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

    global _last_count

    diff = count - _last_count

    if diff + count >= total :
        _last_count = 0
        bar = '=' * _bar_len
        sys.stdout.write('[%s] %s%s - %s\r' % (bar, 100, '%', status))
        sys.stdout.flush()
        print()
        return
    else : 
        _last_count = count

def rgb2hsv (color) :
    color = color / 255
    c_max = np.max(color)
    c_min = np.min(color)
    delta = c_max - c_min

    H = 0; S = 0; V = c_max

    if delta != 0 :
        if c_max == color[0] : H =  (60 * ((color[1] - color[2]) / delta) + 360) % 360
        elif c_max == color[1] : H = 60 * ((color[2] - color[0]) / delta + 2)
        else : H = 60 * ((color[0] - color[1]) / delta + 4)
    
    if c_max != 0 : S = delta / c_max

    hsv = np.asarray([H, S, V]) #.astype(np.uint8)
    return hsv

def integer_scale (old_img, scale) :
    new_img = np.empty([old_img.shape[0] * scale, old_img.shape[1] * scale, old_img.shape[2]], dtype=np.uint8)
    
    for channel in range(old_img.shape[2]) :
        for iy in range(old_img.shape[0]) :
            for ix in range(old_img.shape[1]) :
                new_img[iy * scale: iy * scale + scale, ix * scale : ix * scale + scale, channel] = old_img[iy, ix, channel]
    return new_img

def img2hsv (img) :
    imghsv = np.empty(img.shape)
    for iy in range(img.shape[0]) :
        for ix in range(img.shape[1]) :
            rgb_p = np.asarray([img[iy, ix, 2], img[iy, ix, 1], img[iy, ix, 0]])
            hsv_p = rgb2hsv(rgb_p)

            imghsv[iy, ix, 0] = hsv_p[0]
            imghsv[iy, ix, 1] = hsv_p[1]
            imghsv[iy, ix, 2] = hsv_p[2]

            progress(iy * img.shape[1] + ix, img.shape[0] * img.shape[1])

    return imghsv

def imghsv2rgb (img, palette_HSV, palette_RGB) : # trick
    imgrgb = np.empty(img.shape)
    for iy in range(img.shape[0]) :
        for ix in range(img.shape[1]) :
            hsv_p = np.asarray([img[iy, ix, 0], img[iy, ix, 1], img[iy, ix, 2]])
            traduc = 0
            for i in range(palette_HSV.shape[0]) :
                comparison = hsv_p == palette_HSV[i,:]
                if comparison.all() :
                    traduc = i
                    break
            imgrgb[iy,ix,0] = palette_RGB[i,0]
            imgrgb[iy,ix,1] = palette_RGB[i,1]
            imgrgb[iy,ix,2] = palette_RGB[i,2]

            progress(iy * img.shape[1] + ix, img.shape[0] * img.shape[1])

    return imgrgb

def distance_between_colors (A, B) : # in HSV space

    if np.array_equal(A, B) : return 0

    h1 = A[0] * math.pi / 180; s1 = A[1]; v1 = A[2]
    h2 = B[0] * math.pi / 180; s2 = B[1]; v2 = B[2]
    distance = \
        pow(math.sin(h1)*s1*v1 - math.sin(h2)*s2*v2, 2) + \
        pow(math.cos(h1)*s1*v1 - math.cos(h2)*s2*v2, 2) + \
        pow(v1 - v2, 2)

    return np.abs(distance)

def closest2palette (A, palette_HSV) :

    closest = 0
    max_dist = distance_between_colors(A, palette_HSV[0,:])

    for i in range(palette_HSV.shape[0]) :
        dist = distance_between_colors(A, palette_HSV[i,:])
        if dist == 0 :
            return palette_HSV[i,:]
        elif dist <= max_dist :
            closest = i
            max_dist = dist

    return palette_HSV[closest,:]

def passThroughPallet (img_HSV, palette) :

    for iy in range(img_HSV.shape[0]) :
        for ix in range(img_HSV.shape[1]) :
            new_color = closest2palette([img_HSV[iy,ix,0], img_HSV[iy,ix,1], img_HSV[iy,ix,2]], palette)
            img_HSV[iy,ix,0], img_HSV[iy,ix,1], img_HSV[iy,ix,2] = new_color[0], new_color[1], new_color[2]

            progress(iy * img_HSV.shape[1] + ix, img_HSV.shape[0] * img_HSV.shape[1])

    return img_HSV


if __name__=="__main__":

    cuts = 1
    pixel_density = 8
    user_scale = 1

    check_metod = None # average | mean | None

    name_input = 'c:/Users/Rinders/Desktop/PixelReCreator/example_in.jpg'
    name_output = 'c:/Users/Rinders/Desktop/PixelReCreator/example_out_palette.png'
    name_palette = None # path 'c:/Users/Rinders/Desktop/PixelReCreator/brick_palette.png' or None

    # -------------------------------- Load image -------------------------------- #

    img = cv2.imread(name_input)
    img = np.asarray(img, dtype=np.uint8)

    #img[:,:,0], img[:,:,2] = np.copy(img[:,:,2]), np.copy(img[:,:,0]) # RGB => BGR

    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=np.inf)

    print(img.shape)

    # ------------- Scale image for the user to select color palette ------------- #

    print("Scaling for user visualization...")

    imgk = integer_scale(img, user_scale) if user_scale != 1 else img
    
    palette_RGB = np.asarray([]).reshape(0,3)

    # -------------- Create sub images and apply palette by subimage ------------- #

    hc, wc = int(np.floor(imgk.shape[0] / cuts)), int(np.floor(imgk.shape[1] / cuts))
    h, w = int(np.floor(img.shape[0] / cuts)), int(np.floor(img.shape[1] / cuts))

    cut_container = np.empty([hc,wc], dtype=np.uint8)

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            global palette_RGB
            palette_RGB = np.vstack((palette_RGB, [cut_container[y,x,2], cut_container[y,x,1], cut_container[y,x,0]])) # RGB => BGR
            print("Color added: ", [cut_container[y,x,2], cut_container[y,x,1], cut_container[y,x,0]])
    
    # ------------------------------ Pass cut by cut ----------------------------- #

    pd = pixel_density

    final_container = np.zeros([int(np.floor(img.shape[0] / pd)), int(np.floor(img.shape[1] / pd)), img.shape[2]])

    for iy in range(cuts) :
        for ix in range(cuts) :

            # ---------------------------- Clean palette color --------------------------- #
            
            palette_RGB = np.asarray([]).reshape(0,3)

            if name_palette == None : # get the color pallett by hand

                # ------------------------------- Load the cut ------------------------------- #

                cut_container = imgk[iy*hc : iy*hc + hc, ix*wc : ix*wc + wc ]

                cv2.imshow('image', cut_container)
                cv2.setMouseCallback('image', click_event)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            else :

                pal = cv2.imread(name_palette)
                palette_RGB = np.asarray(pal, dtype=np.uint8)
                
                temp_pal = np.empty((palette_RGB.shape[0]*palette_RGB.shape[1], 3))

                for i in range(palette_RGB.shape[0]) :
                    for j in range(palette_RGB.shape[1]) :
                        temp_pal[i*palette_RGB.shape[1] + j, 2] = palette_RGB[i,j,0]
                        temp_pal[i*palette_RGB.shape[1] + j, 1] = palette_RGB[i,j,1]
                        temp_pal[i*palette_RGB.shape[1] + j, 0] = palette_RGB[i,j,2]

                palette_RGB = temp_pal


            # ---------------------- Create HSV pallet for the piece --------------------- #

            palette_RGB = np.unique(palette_RGB, axis=0)

            palette_HSV = np.apply_along_axis(rgb2hsv, 1, palette_RGB)

            #print("RGB palette: \n ", palette_RGB)
            #print("HSV palette: \n ", palette_HSV)

            # --------------- Shrink original image cut by pixel pixel_density --------------- #

            cut_img = img[iy*h : iy*h + h, ix*w : ix*w + w]
            neo_img = np.empty((int(cut_img.shape[0] / pd), int(cut_img.shape[1] / pd), cut_img.shape[2]), dtype=np.uint8)

            print("Apply pixel density")

            if pd == 1:
                neo_img = cut_img
            else :
                for RGB in range(cut_img.shape[2]) :
                    for i in range(neo_img.shape[0]) :
                        for j in range(neo_img.shape[1]) :
                            sub_matrix = cut_img[i*pd:i*pd+pd, j*pd:j*pd+pd, RGB]
                            neo_img[i,j, RGB] = np.average(sub_matrix) if check_metod == "average" else sub_matrix[int(np.floor(pd / 2)), int(np.floor(pd / 2))]

                            progress(i * neo_img.shape[1] + j, neo_img.shape[0] * neo_img.shape[1])

                            j += pd
                        i += pd
            
            # ------------------------------ PP image to HSV ----------------------------- #

            print("Image to HSV")
            img_HSV = img2hsv(neo_img)

            # ------------------------------- Apply pallet ------------------------------- #

            print("Indexing by palette")
            img_clear = passThroughPallet(img_HSV, palette_HSV)

            # ---------------------------- Return to PP image ---------------------------- #

            print("HSV to Image")
            img_clear = imghsv2rgb(img_clear, palette_HSV, palette_RGB)

            img_clear[:,:,0], img_clear[:,:,2] = np.copy(img_clear[:,:,2]), np.copy(img_clear[:,:,0]) # RGB => BGR

            # -------------------------- Save to final container ------------------------- #

            hn = img_clear.shape[0]
            wn = img_clear.shape[1]

            final_container[iy*hn : iy*hn + hn, ix*wn : ix*wn + wn, :] = img_clear
            
            #cv2.imshow('image', integer_scale(final_container,2).astype(np.uint8))
            #cv2.setMouseCallback('image', click_event)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

    cv2.imwrite(name_output, final_container)
    

    









    





