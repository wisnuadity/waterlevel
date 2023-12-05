from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from apscheduler.schedulers.background import BackgroundScheduler
####################################################
import csv
import os
import math
import cv2
import numpy as np
# import pytesseract
# read the image
from datetime import datetime as datetimes


def findYellow(image):
    ksize = (3, 3)
    image = cv2.blur(image, ksize)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 93, 0], dtype="uint8")
    upper = np.array([45, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    return mask


from statistics import mode, multimode


def treshold_img(img_gray, tresh_bot=150):
    ret, thresh = cv2.threshold(img_gray, tresh_bot, 255, cv2.THRESH_BINARY)
    # visualize the binary image
    cv2.imshow('Binary image' + str(tresh_bot), img_gray)
    cv2.imshow('image_thres1' + str(tresh_bot), thresh)
    return img_gray, thresh


# from scipy import ndimage
# import math
# from skimage.metrics import structural_similarity
from scipy import stats as st


def getLineHough(detect, time, mask_img, folder, detect2, prev_y, crop_img, crop_img2,is_multiple):
    is_gray = False
    #print("folder",folder)
    #exit()
    if (detect[:, :, 0] == detect[:, :, 1]).all():
        is_gray = True
    hsv_img = cv2.cvtColor(detect, cv2.COLOR_BGR2HSV)
    # lower_red = np.array([120, 40, 160], np.uint8)
    # upper_red = np.array([179, 255, 255], np.uint8)
    # mask1 = cv2.inRange(hsv_img, lower_red, upper_red)
    # get_red1 = np.nonzero(mask1)
    ORANGE_MIN = np.array([90, 100, 130], np.uint8)
    ORANGE_MAX = np.array([110, 255, 255], np.uint8)
    mask1_orange = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)

    lower_red = np.array([125, 50, 160], np.uint8)
    upper_red = np.array([179, 255, 255], np.uint8)
    mask1 = cv2.inRange(hsv_img, lower_red, upper_red)

    lower_red2 = np.array([0, 90, 200], np.uint8)
    upper_red2 = np.array([10, 100, 255], np.uint8)
    mask1_red2 = cv2.inRange(hsv_img, lower_red2, upper_red2)

    lower_red_night = np.array([0, 20, 190], np.uint8)
    upper_red_night = np.array([180, 40, 255], np.uint8)
    mask_night = cv2.inRange(hsv_img, lower_red_night, upper_red_night)
    get_red_night = np.nonzero(mask_night)

    lower_yellow_day = np.array([20, 90, 170], np.uint8)
    upper_yellow_day = np.array([40, 255, 255], np.uint8)
    mask_yellow_day = cv2.inRange(hsv_img, lower_yellow_day, upper_yellow_day)
    get_yellow_day = np.nonzero(mask_yellow_day)

    lower_yellow_night = np.array([0, 0, 0], np.uint8)
    upper_yellow_night = np.array([40, 180, 255], np.uint8)
    mask_yellow_night = cv2.inRange(hsv_img, lower_yellow_night, upper_yellow_night)
    get_yellow_night = np.nonzero(mask_yellow_night)

    lower_blue1 = np.array([90, 60, 150], np.uint8)
    upper_blue1 = np.array([110, 180, 255], np.uint8)
    mask1_blue1 = cv2.inRange(hsv_img, lower_blue1, upper_blue1)
    get_blue1 = np.nonzero(mask1_blue1)

    lower_blue2 = np.array([90, 60, 110], np.uint8)
    upper_blue2 = np.array([110, 150, 255], np.uint8)
    mask1_blue2 = cv2.inRange(hsv_img, lower_blue2, upper_blue2)
    get_blue2 = np.nonzero(mask1_blue2)

    lower_red3 = np.array([0, 90, 100], np.uint8)
    upper_red3 = np.array([10, 255, 255], np.uint8)
    mask1_red3 = cv2.inRange(hsv_img, lower_red3, upper_red3)
    
    lower_red4 = np.array([10, 50, 90], np.uint8)
    upper_red4 = np.array([15, 90, 110], np.uint8)
    mask1_red4 = cv2.inRange(hsv_img, lower_red4, upper_red4)

    lower_red5 = np.array([0, 35, 35], np.uint8)
    upper_red5 = np.array([10, 70, 100], np.uint8)
    mask1_red5 = cv2.inRange(hsv_img, lower_red5, upper_red5)
    
    # get_red2 = np.nonzero(mask1_red2)
    get_red1 = np.nonzero(mask1)
    get_red2 = np.nonzero(mask1_red2)
    get_red3 = np.nonzero(mask1_red3)
    get_red4 = np.nonzero(mask1_red4)
    get_red5 = np.nonzero(mask1_red5)

    get_orange1 = np.nonzero(mask1_orange)
    GRAY_MIN1 = np.array([0, 0, 160], np.uint8)
    GRAY_MAX1 = np.array([180, 255, 195], np.uint8)
    mask1_gray = cv2.inRange(hsv_img, GRAY_MIN1, GRAY_MAX1)
    get_gray1 = np.nonzero(mask1_gray)
    result = detect
    im1 = detect
    pinks = cv2.bitwise_and(result, result, mask=mask1)
    pinks = cv2.resize(pinks, (200, 600))
    # cv2.imshow("pinks", pinks)

    print("get_blue1[0]len(get_red1[0])", len(get_blue1[0]),len(get_orange1[0]), len(get_red1[0]), len(get_gray1[0]))
    # result = cv2.bitwise_and(result, result, mask=mask1_blue1)
    # cv2.imshow("result",result)
    # cv2.imshow("detect",detect)
    # cv2.waitKey(0)
    if is_gray == False:
        if len(get_blue1[0]) > 5 and (folder=="TC.04" or folder=="TTC01" or folder=="TC.12"):
            # img_count += 1

            # len(get_red1[0])
            result = cv2.bitwise_and(result, result, mask=mask1_blue1)
            get_red = np.nonzero(result)
            if True:
                # print("red float", len(get_red[0]),get_red[0])
                maxy_red = get_red[0][len(get_red[0]) - 1]
                maxx_red = get_red[1][len(get_red[0]) - 1]
                modepink = st.mode(get_red[0])
            # print(img_count,len(get_red1[0]),maxy_red,modepink)
            # print("pink1", modepink[0][0])
            # cv2.imshow("im12",im12)
            # im1 = cv2.circle(im1, (maxx_red,maxy_red), 2, (0,0,255), 20)
            print("maxx_red, modepink[0][0]",maxx_red, modepink)
            im1 = cv2.circle(im1, (maxx_red, modepink[0]), 2, (255, 0, 255), 5)
            # im1[maxy_red,:] = 255
            maxy_red = modepink[0]
            print("maxy_blue",maxy_red)
            # cv2.imshow("pink1", im1)
            # cv2.imshow("111", result)
            # cv2.waitKey(0)
            # exit()
            # continue
            print(1111)
        elif len(get_blue2[0]) > 5 and (folder=="TC.04" or folder=="TTC01" or folder=="TC.12"):
            # img_count += 1

            # len(get_red1[0])
            result = cv2.bitwise_and(result, result, mask=mask1_blue2)
            get_red = np.nonzero(result)
            if True:
                # print("red float", len(get_red[0]),get_red[0])
                maxy_red = get_red[0][len(get_red[0]) - 1]
                maxx_red = get_red[1][len(get_red[0]) - 1]
                modepink = st.mode(get_red[0])
            # print(img_count,len(get_red1[0]),maxy_red,modepink)
            # print("pink1", modepink[0][0])
            # cv2.imshow("im12",im12)
            # im1 = cv2.circle(im1, (maxx_red,maxy_red), 2, (0,0,255), 20)
            im1 = cv2.circle(im1, (maxx_red, modepink[0][0]), 2, (255, 0, 255), 5)
            # im1[maxy_red,:] = 255
            maxy_red = modepink[0][0]
            print("maxy_blue",maxy_red)
            # cv2.imshow("pink1", im1)
            # cv2.imshow("111", result)
            # cv2.waitKey(0)
            # exit()
            # continue
            print(2222)
        elif len(get_red4[0]) > 1 and (folder=="TC.12x"):
            # img_count += 1
            #print(4)
            #exit()
            # len(get_red1[0])
            result = cv2.bitwise_and(result, result, mask=(mask1_red4))
            get_red = np.nonzero(result)
            if True:
                # print("red float", len(get_red[0]),get_red[0])
                maxy_red = get_red[0][len(get_red[0]) - 1]
                maxx_red = get_red[1][len(get_red[0]) - 1]
                modepink = st.mode(get_red[0])
            # print(img_count,len(get_red1[0]),maxy_red,modepink)
            print("pink1", modepink[0][0])
            # cv2.imshow("im12",im12)
            # im1 = cv2.circle(im1, (maxx_red,maxy_red), 2, (0,0,255), 20)
            im1 = cv2.circle(im1, (maxx_red, modepink[0][0]), 2, (255, 0, 255), 5)
            # im1[maxy_red,:] = 255
            # cv2.imshow("pink1", im1)
            # cv2.imshow("111", result)
            maxy_red = modepink[0][0]

            # continue
            print(3333)
        elif len(get_red5[0]) > 1 and (folder=="TC.12x"):
            # img_count += 1
            #print(5)
            #exit()

            # len(get_red1[0])
            result = cv2.bitwise_and(result, result, mask=(mask1_red5))
            get_red = np.nonzero(result)
            if True:
                # print("red float", len(get_red[0]),get_red[0])
                maxy_red = get_red[0][len(get_red[0]) - 1]
                maxx_red = get_red[1][len(get_red[0]) - 1]
                modepink = st.mode(get_red[0])
            # print(img_count,len(get_red1[0]),maxy_red,modepink)
            print("pink1", modepink[0][0])
            # cv2.imshow("im12",im12)
            # im1 = cv2.circle(im1, (maxx_red,maxy_red), 2, (0,0,255), 20)
            im1 = cv2.circle(im1, (maxx_red, modepink[0][0]), 2, (255, 0, 255), 5)
            # im1[maxy_red,:] = 255
            # cv2.imshow("pink1", im1)
            # cv2.imshow("111", result)
            maxy_red = modepink[0][0]

            # continue
            print(4444)
        elif len(get_red1[0]) > 10 and (folder=="TC.12x"):
            # img_count += 1

            # len(get_red1[0])
            result = cv2.bitwise_and(result, result, mask=mask1)
            get_red = np.nonzero(result)
            if True:
                # print("red float", len(get_red[0]),get_red[0])
                maxy_red = get_red[0][len(get_red[0]) - 1]
                maxx_red = get_red[1][len(get_red[0]) - 1]
                modepink = st.mode(get_red[0])
            # print(img_count,len(get_red1[0]),maxy_red,modepink)
            print("pink1", modepink[0])
            # cv2.imshow("im12",im12)
            # im1 = cv2.circle(im1, (maxx_red,maxy_red), 2, (0,0,255), 20)
            im1 = cv2.circle(im1, (maxx_red, modepink[0]), 2, (255, 0, 255), 5)
            # im1[maxy_red,:] = 255
            # cv2.imshow("pink1", im1)
            # cv2.imshow("111", result)
            maxy_red = modepink[0]

            # continue
            print(5555)
        elif len(get_yellow_night[0]) > 10:
            # img_count += 1

            # len(get_red1[0])
            result = cv2.bitwise_and(result, result, mask=mask_yellow_night)
            get_red = np.nonzero(result)
            if True:
                # print("red float", len(get_red[0]),get_red[0])
                maxy_red = get_red[0][len(get_red[0]) - 1]
                maxx_red = get_red[1][len(get_red[0]) - 1]
                modepink = st.mode(get_red[0])
            # print(img_count,len(get_red1[0]),maxy_red,modepink)
            # print("pink2", modepink[0][0])
            # cv2.imshow("im12",im12)
            # im1 = cv2.circle(im1, (maxx_red,maxy_red), 2, (0,0,255), 20)
            im1 = cv2.circle(im1, (maxx_red, get_red[0][-1]), 2, (255, 0, 255), 5)
            # im1[maxy_red,:] = 255
            # cv2.imshow("pink1", im1)
            # cv2.imshow("111", result)
            maxy_red = get_red[0][-1]

            # continue
            print("yellow night")

        elif len(get_yellow_day[0]) > 10:
            # img_count += 1

            # len(get_red1[0])
            result = cv2.bitwise_and(result, result, mask=mask_yellow_day)
            get_red = np.nonzero(result)
            if True:
                # print("red float", len(get_red[0]),get_red[0])
                maxy_red = get_red[0][len(get_red[0]) - 1]
                maxx_red = get_red[1][len(get_red[0]) - 1]
                modepink = st.mode(get_red[0])
            # print(img_count,len(get_red1[0]),maxy_red,modepink)
            # print("pink2", get_red[0][-1])
            # cv2.imshow("im12",im12)
            # im1 = cv2.circle(im1, (maxx_red,maxy_red), 2, (0,0,255), 20)
            im1 = cv2.circle(im1, (maxx_red, get_red[0][-1]), 2, (255, 0, 255), 5)
            # im1[maxy_red,:] = 255
            # cv2.imshow("pink1", im1)
            # cv2.imshow("111", result)
            maxy_red = get_red[0][-1]

            # continue
            print("yellow day")

        elif len(get_red2[0]) > 10:
            # img_count += 1

            # len(get_red1[0])
            result = cv2.bitwise_and(result, result, mask=mask1_red2)
            get_red = np.nonzero(result)
            if True:
                # print("red float", len(get_red[0]),get_red[0])
                maxy_red = get_red[0][len(get_red[0]) - 1]
                maxx_red = get_red[1][len(get_red[0]) - 1]
                modepink = st.mode(get_red[0])
            # print(img_count,len(get_red1[0]),maxy_red,modepink)
            print("pink2", modepink[0][0])
            # cv2.imshow("im12",im12)
            # im1 = cv2.circle(im1, (maxx_red,maxy_red), 2, (0,0,255), 20)
            im1 = cv2.circle(im1, (maxx_red, modepink[0][0]), 2, (255, 0, 255), 5)
            # im1[maxy_red,:] = 255
            # cv2.imshow("pink1", im1)
            # cv2.imshow("111", result)
            maxy_red = modepink[0][0]

            # continue
            print(6666)
        elif len(get_red_night[0]) > 10:
            # img_count += 1

            # len(get_red1[0])
            result = cv2.bitwise_and(result, result, mask=mask_night)
            get_red = np.nonzero(result)
            if True:
                # print("red float", len(get_red[0]),get_red[0])
                maxy_red = get_red[0][len(get_red[0]) - 1]
                maxx_red = get_red[1][len(get_red[0]) - 1]
                modepink = st.mode(get_red[0])
            # print(img_count,len(get_red1[0]),maxy_red,modepink)
            print("pink2", modepink[0])
            # cv2.imshow("im12",im12)
            # im1 = cv2.circle(im1, (maxx_red,maxy_red), 2, (0,0,255), 20)
            im1 = cv2.circle(im1, (maxx_red, modepink[0]), 2, (255, 0, 255), 5)
            # im1[maxy_red,:] = 255
            # cv2.imshow("pink1", im1)
            # cv2.imshow("111", result)
            maxy_red = modepink[0]

            # continue
            print(7777)
        elif len(get_red3[0]) > 10 and is_multiple==False:
            # img_count += 1

            # len(get_red1[0])
            result = cv2.bitwise_and(result, result, mask=mask1_red3)
            get_red = np.nonzero(result)
            if True:
                # print("red float", len(get_red[0]),get_red[0])
                maxy_red = get_red[0][len(get_red[0]) - 1]
                maxx_red = get_red[1][len(get_red[0]) - 1]
                modepink = st.mode(get_red[0])
            # print(img_count,len(get_red1[0]),maxy_red,modepink)
            print("pink3", modepink[0][0])
            # cv2.imshow("im12",im12)
            # im1 = cv2.circle(im1, (maxx_red,maxy_red), 2, (0,0,255), 20)
            im1 = cv2.circle(im1, (maxx_red, modepink[0][0]), 2, (255, 0, 255), 5)
            # im1[maxy_red,:] = 255
            # cv2.imshow("pink1", im1)
            # cv2.imshow("111", result)
            maxy_red = modepink[0][0]

            # continue
            print(8888)
        elif len(get_orange1[0]) > 1 and is_multiple==False:
            # img_count += 1
            result = im1
            # len(get_red1[0])
            result = cv2.bitwise_and(result, result, mask=mask1_orange)
            get_red = np.nonzero(result)
            if True:
                # print("red float", len(get_red[0]),get_red[0])
                maxy_red = get_red[0][len(get_red[0]) - 1]
                maxx_red = get_red[1][len(get_red[0]) - 1]
                modepink = st.mode(get_red[0])
            # print(img_count,len(get_red1[0]),maxy_red,modepink)
            print("orange", modepink[0][0])
            im12 = im1[modepink[0][0] - 20:modepink[0][0] + 20, maxx_red - 20:maxx_red + 20]
            # cv2.imshow("im12",im12)
            # im1 = cv2.circle(im1, (maxx_red,maxy_red), 2, (0,0,255), 20)
            im1 = cv2.circle(im1, (maxx_red, modepink[0][0]), 2, (255, 0, 255), 5)
            # im1[maxy_red,:] = 255
            # cv2.imshow("orange1", im1)
            # cv2.imshow("111", result)
            maxy_red = modepink[0][0]

            # continue
            print(9999)
        elif len(get_gray1[0]) > 10 and is_multiple==False:
            # img_count += 1
            result = im1
            # len(get_red1[0])
            result = cv2.bitwise_and(result, result, mask=mask1_gray)
            get_red = np.nonzero(result)
            if True:
                # print("red float", len(get_red[0]),get_red[0])
                maxy_red = get_red[0][len(get_gray1[0]) - 1]
                maxx_red = get_red[1][len(get_gray1[0]) - 1]
                modepink = st.mode(get_red[0])
            # print(img_count,len(get_red1[0]),maxy_red,modepink)
            print("gray", modepink[0][0])
            im12 = im1[modepink[0][0] - 20:modepink[0][0] + 20, maxx_red - 20:maxx_red + 20]
            # cv2.imshow("im12",im12)
            # im1 = cv2.circle(im1, (maxx_red,maxy_red), 2, (0,0,255), 20)
            im1 = cv2.circle(im1, (maxx_red, modepink[0][0]), 2, (255, 0, 255), 5)
            # im1[maxy_red,:] = 255
            # cv2.imshow("gray1", im1)
            # cv2.imshow("111", result)

            # continue
        else:
            print(10101010)
            if is_multiple==False:
                print("lower")
                lower_red = np.array([80, 40, 120], np.uint8)
                # lower_red = np.array([80, 0, 120], np.uint8)
                upper_red = np.array([180, 255, 255], np.uint8)
                # lower_red = np.array([180, 70, 100], np.uint8)
                # upper_red = np.array([255, 80, 200], np.uint8)
                mask1 = cv2.inRange(hsv_img, lower_red, upper_red)
                get_red1 = np.nonzero(mask1)
                result = cv2.bitwise_and(im1, im1, mask=mask1)
                ORANGE_MIN = np.array([90, 100, 130], np.uint8)
                ORANGE_MAX = np.array([180, 255, 255], np.uint8)
                mask1_orange = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)
                get_orange1 = np.nonzero(mask1_orange)
                GRAY_MIN1 = np.array([0, 0, 110], np.uint8)
                GRAY_MAX1 = np.array([180, 255, 120], np.uint8)
                mask1_gray = cv2.inRange(hsv_img, GRAY_MIN1, GRAY_MAX1)
                get_gray1 = np.nonzero(mask1_gray)
                # cv2.imshow("111", mask1_orange)
                print("len(get_red1[0])", len(get_orange1[0]), len(get_red1[0]), len(get_gray1[0]))
                if len(get_red1[0]) > 10:
                    # img_count += 1
                    result = im1
                    # len(get_red1[0])
                    result = cv2.bitwise_and(result, result, mask=mask1)
                    get_red = np.nonzero(result)
                    if True:
                        # print("red float", len(get_red[0]),get_red[0])
                        maxy_red = get_red[0][len(get_red[0]) - 1]
                        maxx_red = get_red[1][len(get_red[0]) - 1]
                        modepink = st.mode(get_red[0])
                    # print(img_count,len(get_red1[0]),maxy_red,modepink)
                    print("lpink1", modepink[0][0])
                    im12 = im1[modepink[0][0] - 20:modepink[0][0] + 20, maxx_red - 20:maxx_red + 20]
                    # cv2.imshow("im12",im12)
                    # im1 = cv2.circle(im1, (maxx_red,maxy_red), 2, (0,0,255), 20)
                    im1 = cv2.circle(im1, (maxx_red, modepink[0][0]), 2, (255, 0, 255), 5)
                    # im1[maxy_red,:] = 255
                    # cv2.imshow("lpink1", im1)
                    # cv2.imshow("111", result)
                    maxy_red = modepink[0][0]
                    maxy_red = get_red[0][-1]

                    # continue
                elif len(get_orange1[0]) > 10:
                    # img_count += 1
                    result = im1
                    # len(get_red1[0])
                    result = cv2.bitwise_and(result, result, mask=mask1_orange)
                    get_red = np.nonzero(result)
                    if True:
                        # print("red float", len(get_red[0]),get_red[0])
                        maxy_red = get_red[0][len(get_red[0]) - 1]
                        maxx_red = get_red[1][len(get_red[0]) - 1]
                        modepink = st.mode(get_red[0])
                    # print(img_count,len(get_red1[0]),maxy_red,modepink)
                    print("lorange", modepink[0][0])
                    im12 = im1[modepink[0][0] - 20:modepink[0][0] + 20, maxx_red - 20:maxx_red + 20]
                    # cv2.imshow("im12",im12)
                    # im1 = cv2.circle(im1, (maxx_red,maxy_red), 2, (0,0,255), 20)
                    im1 = cv2.circle(im1, (maxx_red, modepink[0][0]), 2, (255, 0, 255), 5)
                    # im1[maxy_red,:] = 255
                    # cv2.imshow("lorange1", im1)
                    # cv2.imshow("111", result)
                    maxy_red = modepink[0][0]
                    # continue
                elif len(get_gray1[0]) > 10:
                    # img_count += 1
                    result = im1
                    # len(get_red1[0])
                    result = cv2.bitwise_and(result, result, mask=mask1_gray)
                    get_red = np.nonzero(result)
                    if True:
                        # print("red float", len(get_red[0]),get_red[0])
                        maxy_red = get_red[0][len(get_red[0]) - 1]
                        maxx_red = get_red[1][len(get_red[0]) - 1]
                        modepink = st.mode(get_red[0])
                    # print(img_count,len(get_red1[0]),maxy_red,modepink)
                    print("lgray", modepink[0][0])
                    im12 = im1[modepink[0][0] - 20:modepink[0][0] + 20, maxx_red - 20:maxx_red + 20]
                    # cv2.imshow("im12",im12)
                    # im1 = cv2.circle(im1, (maxx_red,maxy_red), 2, (0,0,255), 20)
                    im1 = cv2.circle(im1, (maxx_red, maxy_red), 2, (255, 0, 255), 5)
                    # im1[maxy_red,:] = 255
                    # cv2.imshow("lgray1", im1)
                    # cv2.imshow("111", result)
            else:
                maxy_red = 9999
    else:
        hsv_gray = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        result2 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        GRAY_MIN1 = np.array([0, 0, 220], np.uint8)
        GRAY_MAX1 = np.array([180, 255, 250], np.uint8)
        mask1_gray = cv2.inRange(hsv_gray, GRAY_MIN1, GRAY_MAX1)

        result_gray = cv2.bitwise_and(result2, result2, mask=mask1_gray)
        contours, hierarchy = cv2.findContours(result_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(result_gray, contours, -1, (0, 255, 0), 3)

        get_gray1 = np.nonzero(result_gray)
        
        #print("get_gray1 ", len(get_gray1[0]))
        if len(get_gray1[0])>5:
            maxy_red = get_gray1[0][-1]
        else:
            GRAY_MIN1 = np.array([0, 0, 120], np.uint8)
            GRAY_MAX1 = np.array([180, 255, 240], np.uint8)
            mask1_gray = cv2.inRange(hsv_gray, GRAY_MIN1, GRAY_MAX1)
            result_gray = cv2.bitwise_and(result2, result2, mask=mask1_gray)
            contours, hierarchy = cv2.findContours(result_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # cv2.drawContours(result_gray, contours, -1, (0, 255, 0), 3)

            get_gray1 = np.nonzero(result_gray)
            maxy_red = get_gray1[0][-1]
    result[maxy_red:, :] = 0

    # cv2.imshow("drawnLines1", drawnLines1)
    # cv2.waitKey(0)

    maxy = maxy_red
    print("maxy_red1",maxy_red)
    # if abs(prev_y - maxy) > 200 and prev_y > 0:
    #     lower_yellow = np.array([0, 80, 180], np.uint8)
    #     upper_yellow = np.array([40, 255, 255], np.uint8)
    #     maskyellow1 = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    #     get_yellow1 = np.nonzero(maskyellow1)
    #     result = detect
    #     if len(get_yellow1[0]) > 10:
    #         result = im1
    #         # len(get_red1[0])
    #         result = cv2.bitwise_and(result, result, mask=maskyellow1)
    #         get_red = np.nonzero(result)
    #         maxy = get_red[0][len(get_red[0]) - 1]
    #         maxx = get_red[1][len(get_red[0]) - 1]
    #         result = cv2.circle(result, (maxx, maxy), 2, (255, 0, 255), 5)
    #         # print("yellow",maxy)
    #         # cv2.imshow("getyellow",result)
    #         # if abs(prev_y - maxy) > 200:
    #         #     maxy = prev_y
    #     else:
    #         maxy = prev_y
    # cv2.waitKey(0)
    scale_mask_top = mask_img[0:20, :, :]
    image_hough = result
    is_night = False
    prev_y = maxy
    return image_hough, maxy, maxy_red, is_night, prev_y


def getLineHough_ori(detect, time, mask_img, folder):
    gray_detect = cv2.cvtColor(detect, cv2.COLOR_BGR2GRAY)
    # Apply edge detection method on the image
    t_lower = 50  # Lower Threshold
    t_upper = 150  # Upper threshold
    edges_detect = cv2.Canny(gray_detect, t_lower, t_upper)
    ###########leaf

    img_green = detect.copy()
    hsv = cv2.cvtColor(img_green, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))

    ## slice the green
    imask = mask > 0
    sharp_green = detect.copy()
    green = np.zeros_like(img_green, np.uint8)
    green[imask] = 255
    kernel_green = np.ones((2, 2), np.uint8)
    green = cv2.dilate(green, kernel_green, iterations=1)
    sharp_green[green > 0] = 0

    image_hough = detect
    image_hough[green > 0] = 0

    ##############
    # Applying the Canny Edge filter
    edge = cv2.Canny(sharp_green, t_lower, t_upper)
    # lines_detect = cv2.HoughLines(edges_detect, 1, np.pi / 180, 100)
    kernel = np.ones((1, 5), np.uint8)
    horizontalLines = cv2.erode(edge, kernel, iterations=1)
    # print(horizontalLines.shape)
    # cv2.imshow("horizontalLines", horizontalLines)
    horizontalLines = cv2.resize(horizontalLines, (horizontalLines.shape[1] * 3, horizontalLines.shape[0] * 3))
    image_hough = cv2.resize(image_hough, (image_hough.shape[1] * 3, image_hough.shape[0] * 3))
    # horizontalLines = cv2.dilate(horizontalLines, (5, 5), iterations=2)
    kernel_2 = np.ones((1, 2), np.uint8)
    horizontalLines = cv2.dilate(horizontalLines, kernel_2, iterations=2)
    # cv2.imshow("Canny",edge)
    # cv2.imshow("sdas",horizontalLines)
    # cv2.waitKey(0)
    # exit()
    # This returns an array of r and theta values
    # lines_detect = cv2.HoughLines(image_hough, 1, np.pi / 180, 200)

    # detect red
    lower_red = np.array([0, 0, 200], dtype="uint8")
    maxy_red = -1
    upper_red = np.array([50, 50, 255], dtype="uint8")
    ksize = (3, 3)
    image = detect.copy()
    cv2.imshow("image1", image)
    # image = cv2.resize(image, (image_hough.shape[1], image_hough.shape[0]))
    # cv2.imshow("image2", image)
    image = cv2.blur(image, ksize)
    result = detect.copy()
    mask = cv2.inRange(image, lower_red, upper_red)
    # print("np.max(mask)",np.max(mask))
    if np.max(mask) > 0:
        result = cv2.bitwise_and(result, result, mask=mask)
        get_red = np.nonzero(result)
        # print(len(get_red[0]), get_red)

        if len(get_red[0]) > 30:
            # print("red float", len(get_red[0]))
            maxy_red = get_red[0][0] * 3

    lines = cv2.HoughLinesP(horizontalLines, 1, np.pi / 180, 70)
    # print("lines",lines)
    lines_list = []

    lines_listy = []
    try:
        for points in lines:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]
            # Draw the lines joing the points
            # On the original image
            cv2.line(image_hough, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Maintain a simples lookup list for points
            lines_listy.append(y2)
            lines_listy.sort()
            diff_y = lines_listy[len(lines_listy) - 1] - lines_listy[len(lines_listy) - 2]
            if diff_y > 100:
                lines_listy.pop()
                # print(diff_y,lines_listy[len(lines_listy)-1],lines_listy[len(lines_listy)-2])
                continue
            # lines_list.append([(x1, y1), (x2, y2)])
            cv2.line(image_hough, (x2, y2), (x1, y1), (0, 0, 255), 2)

        maxy = max(lines_listy)

    except:
        get_white = np.nonzero(horizontalLines)
        # print(horizontalLines.shape,get_white[0])

        try:
            if time == 0:
                maxy = max(get_white[0])
            else:
                maxy = -1000
        except:
            maxy = -1000
    scale_mask_top = mask_img[0:50, :, :]
    if maxy_red == -1:
        # print("+++++",maxy,horizontalLines.shape,mask_img.shape)
        maxy_scale = int(maxy / (int(horizontalLines.shape[0] / mask_img.shape[0])))
    else:
        maxy_scale = int(maxy_red / (int(horizontalLines.shape[0] / mask_img.shape[0])))
    scale_mask_bot = mask_img[maxy_scale - 2:maxy_scale, :, :]
    # print("++++++++++++s",scale_mask_top.shape)
    sc_top = np.nonzero(scale_mask_top)[1][-1] - np.nonzero(scale_mask_top)[1][0]
    # print("++++++++++++d")
    # sc_bot = np.nonzero(scale_mask_bot)[1][-1] - np.nonzero(scale_mask_bot)[1][0]
    # print("++++++++++++f")
    sc_bot = 1
    # print("++++++++++++0")
    div_mask_scale = sc_top / sc_bot
    # print("++++++++++++1")
    div_height_scale = 1 - (maxy_scale * (int(horizontalLines.shape[0] / mask_img.shape[0]))) / image_hough.shape[0]
    # print("++++++++++++2")
    # cv2.imshow("image_hough", image_hough)
    # cv2.waitKey(0)
    # exit()
    return image_hough, maxy, div_mask_scale, maxy_red, div_height_scale
    # return detect, image_hough, rotated


def getLineHough2(detect):
    gray_detect = cv2.cvtColor(detect, cv2.COLOR_BGR2GRAY)
    # ksize
    ksize = (10, 10)

    # Using cv2.blur() method
    gray_detect = cv2.blur(gray_detect, ksize)
    image_hough = detect.copy()
    # Apply edge detection method on the image
    edges_detect = cv2.Canny(gray_detect, 100, 200, apertureSize=5)

    # This returns an array of r and theta values
    lines_detect = cv2.HoughLines(edges_detect, 1, np.pi / 180, 200)

    # # The below for loop runs till r and theta values
    # # are in the range of the 2d array
    # deg = 0
    # print(len(lines_detect))
    for r_theta in lines_detect:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        # Stores the value of cos(theta) in a
        a = np.cos(theta)

        # Stores the value of sin(theta) in b
        b = np.sin(theta)

        # x0 stores the value rcos(theta)
        x0 = a * r

        # y0 stores the value rsin(theta)
        y0 = b * r

        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000 * (-b))

        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000 * (a))

        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000 * (-b))

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000 * (a))
        #     myradians = math.atan2(y2 - y1, x2 - x1)
        #     mydegrees = math.degrees(myradians)
        #     print("mydegrees",mydegrees)
        #     if mydegrees<0:
        #         deg = 90+mydegrees
        #     else:
        #         deg = mydegrees - 90
        # rotated = ndimage.rotate(detect, deg)
        # print(deg,x1, y1, x2, y2)
        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        cv2.line(image_hough, (x2, y2), (x1, y1), (0, 0, 255), 2)
    return detect, image_hough, len(lines_detect)
    # return detect, image_hough, rotated


def getcntyellow(img):
    kernel = np.ones((3, 3), "uint8")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_light = np.array([20, 20, 200], np.uint8)
    yellow_dark = np.array([60, 255, 255], np.uint8)
    yellow = cv2.inRange(hsv, yellow_light, yellow_dark)
    yellow = cv2.dilate(yellow, kernel)
    res_yellow = cv2.bitwise_and(img, img, mask=yellow)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ksize = (13, 13)
    img = cv2.blur(img, ksize)
    # gray = imgs
    # gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    # apply thresholding on the gray image to create a binary image
    ret, thresh = cv2.threshold(gray, 0, 255, 0)

    # find the contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for _, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            cnt = contours[len(contours) - 1]
            cv2.drawContours(img, cnt, -1, (0, 255, 255), 2)
            cv2.putText(img, "Yellow", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
    blur = cv2.medianBlur(res_yellow, 15)
    # cv2.imshow("Blur", blur)
    # cv2.imshow("Color Tracker", img)
    cv2.imshow("mask", res_yellow)


def getContour(img, r_maskimage):
    ksize = (3, 3)

    img = cv2.blur(r_maskimage, ksize)
    # gray = r_maskimage
    gray = cv2.cvtColor(r_maskimage, cv2.COLOR_BGR2GRAY)
    # apply thresholding on the gray image to create a binary image
    # cv2.imshow("gray",gray)
    # cv2.waitKey(0)
    ret, thresh = cv2.threshold(gray, 0, 255, 0)

    # find the contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("len contour",len(contours))
    # take the first contour
    c = max(contours, key=cv2.contourArea)
    cnt = contours[0]

    # compute the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(c)
    # print(x, y, w, h)
    # draw contour
    # img = cv2.drawContours(rotated_black, [cnt],-1, (0, 255, 0), 5, cv2.LINE_AA)

    # draw the bounding rectangle
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return img, x, y, w, h


def getContour2(img, r_maskimage):
    image = img
    ksize = (3, 3)

    image = cv2.blur(image, ksize)
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([22, 93, 0], dtype="uint8")
    upper = np.array([45, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(original, (x, y), (x + w, y + h), (36, 255, 12), 2)
    # cv2.imshow('mask', mask)
    # cv2.imshow('original', original)
    # cv2.waitKey(0)


import pathlib
import datetime


def get_waterlevel_newdata(stn, rtu_img, rtu_img2, prev_y,last_maxy,last_is_night):
    mask_name_list = []
    folder = stn
    # print(stn)
    list_dir = os.listdir('../RTU data/' + folder + '/images')
    list_dir.sort(reverse=True)
    # print(list_dir)
    list_water_level = []
    list_img = []

    mask_lists = list(pathlib.Path('static/' + folder).glob('*.txt'))
    # print(mask_lists)
    # exit()
    mask_list = []
    count_mask = len(mask_lists) - 1
    for f_m in mask_lists:
        split_f_m = str(f_m).split("\\")
        mask_file_ori = split_f_m[2]
        # print(split_f_m)
        m_name = split_f_m[0] + "/" + split_f_m[1] + "/" + mask_file_ori
        # m_name = split_f_m[0] + "/" + split_f_m[1] + "/mask_" + str(count_mask) + ".txt"
        mask_list.append(m_name)
        count_mask = count_mask - 1
    mask_list.sort(reverse=True)
    # print(mask_lists)
    list_mask_date = []

    cur_mask = 0
    stat_mask = 0
    cek_calc = 0
    

    for msk in mask_list:
        is_multiple = False
        # continue
        list_water_scale_f1 = []
        list_water_scale_f1_div = []
        list_water_scale_f2 = []
        list_max_level_cm = []
        list_max_level_m = []
        list_water_scale = []
        list_water_scale_div = []
        mask_img = []
        file1 = open(msk, 'r')
        Lines = file1.readlines()
        # print("multi",msk,len(Lines))
        if len(Lines)>1:
            is_multiple = True
        # exit()
        count = 0
        # print("-----------",Lines,line(Lines))
        prev_maxy = 0
        for line in Lines:
            count += 1
            split_line = line.strip().split(',')
            # print("====",line,split_line)
            mask = cv2.imread('static/' + folder + '/' + split_line[1] + ".jpg")
            print('==static/' + folder + '/' + split_line[1] + ".jpg")
            # cv2.imshow(split_line[1],mask)
            # cv2.waitKey(1000)

            # continue
            mask_img.append(mask)
            mask_name_list.append(split_line[1])
            list_max_level_m.append(float(split_line[2]))
            list_max_level_cm.append(float(split_line[3]))
            list_water_scale.append(split_line[4])
            list_water_scale_div.append(split_line[6])
            list_water_scale_f1.append(split_line[7])
            list_water_scale_f1_div.append(split_line[8])
            list_water_scale_f2.append(split_line[9])

            mask_name = "mask_2_46_0"
            mask_datetime_data = datetime.datetime(int(split_line[5][0:4]), int(split_line[5][4:6]),
                                                   int(split_line[5][6:8]), int(split_line[5][8:10]),
                                                   int(split_line[5][10:12]))
        print(mask_name_list)
        list_mask_date.append(mask_datetime_data)

        # for file in list_dir:
        if True:
            split_rtu_img = rtu_img.split("/")
            split_rtu_img_2 = rtu_img2.split("/")
            split_rtu_img2 = split_rtu_img[-1].split("\\")
            split_rtu_img22 = split_rtu_img_2[-1].split("\\")
            file = split_rtu_img2[-1]
            file2 = split_rtu_img22[-1]
            show_level_list = []
            show_maxy_list = []
            time = 1
            split_filename = file.split("_")
            time = 1
            if True:
                # try:
                if int(split_filename[-2][8:10]) > 17 or int(split_filename[-2][8:10]) < 6:
                    #print("night")
                    time = 0
                else:
                    #print("day")
                    time = 1

            split_file = file.split("_")
            print(split_file,folder +'/images/'+ file)
            file_date = datetime.datetime(int(split_file[-2][0:4]), int(split_file[-2][4:6]), int(split_file[-2][6:8]),
                                          int(split_file[-2][8:10]), int(split_file[-2][10:12]))
            # print("==================",'../RTU data/'+folder +'/images/'+ file)
            image = cv2.imread('../RTU data/' + folder + '/images/' + file)
            image2 = cv2.imread('../RTU data/' + folder + '/images/' + file2)
            image = cv2.resize(image, (1920, 1080))
            image2 = cv2.resize(image, (1920, 1080))
            if int(last_maxy)>0:
                image[last_maxy+150:,:] = 0
                image[:last_maxy-150:,:] = 0
            # cv2.imshow("crop threshold",image)
            # cv2.waitKey(0)
            # exit()
            c_mask = 0
            max_line = 0
            is_night = 0
            show_img_masking = image.copy()

            max_date = max(list_mask_date)
            list_mask_date.sort(reverse=True)
            #print("===",list_mask_date)

            # for msk2 in mask_list:
            if True:
                if file_date < mask_datetime_data:
                    print("1")
                    print(file_date, "<", mask_datetime_data)
                    continue
                else:
                    stat_mask += 1
                    print("2")
                    print(file_date, ">", mask_datetime_data)

            #print("===mask_img",mask_img,len(mask_img))
            if stat_mask > 1:
                print("=mask_list_mask_list",stat_mask)
                continue

            else:
                cek_calc += 1
                print("==========", cek_calc)
            for i in range(len(mask_img)):
                print("=mask_name_list",mask_name_list[i])
                image_masking = image.copy()
                image_masking2 = image2.copy()
                # print(i)
                split_mask_name = mask_name_list[i].split("_")

                max_level_m = int(list_max_level_m[i])
                max_level_cm = int(list_max_level_cm[i])

                # r_maskimage = cv2.resize(mask_img[i], (1280, 770))
                r_maskimage = mask_img[i]

                r_maskimage = cv2.resize(mask_img[i], (1920, 1080))
                # print(image_masking.shape,r_maskimage.shape)
                # print(image_masking.shape,r_maskimage.shape)
                image_masking[r_maskimage < 100] = 0
                image_masking2[r_maskimage < 100] = 0

                # print(image_masking.shape)
                img, x, y, w, h = getContour(image_masking, r_maskimage)

                ############# new crop
                crop_img = image_masking[0:y + h, x:x + w]
                crop_img2 = image_masking2[0:y + h, x:x + w]
                crop_mask_img = r_maskimage[0:y + h, x:x + w]

                print('mask_list_mask_list', msk, mask_list)
                # print("crop_img"+msk)
                # cv2.imshow("crop_img"+msk, crop_img)
                # cv2.waitKey(1)
                # continue
                ############# oriinal crop
                # crop_img = image_masking[y:y + h, x:x + w]
                # crop_img2 = image_masking2[y:y + h, x:x + w]
                # crop_mask_img = r_maskimage[y:y + h, x:x + w]
                #############
                # print("===============",crop_img.shape,split_scale)
                # cv2.imshow("===============",crop_img)
                # cv2.waitKey(0)
                # exit()
                # getContour2(crop_img, crop_img)
                t_lower = 50  # Lower Threshold
                t_upper = 150  # Upper threshold

                # Applying the Canny Edge filter
                edge = cv2.Canny(crop_img, t_lower, t_upper)
                kernel = np.ones((1, 5), np.uint8)

                kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                # image_sharp = cv2.filter2D(src=crop_img, ddepth=-5, kernel=kernel_sharp)
                # image_sharp = cv2.blur(image_sharp, (3, 3))
                image_sharp2 = cv2.filter2D(src=crop_img2, ddepth=-5, kernel=kernel_sharp)
                image_sharp2 = cv2.blur(image_sharp2, (3, 3))
                image_sharp2 = crop_img
                image_sharp = crop_img
                if True:
                    # try:
                    horizontalLines = cv2.erode(edge, kernel, iterations=1)
                    # print(image_sharp.shape)
                    imagehough, maxy, maxy_red, is_night, p = getLineHough(image_sharp, time, crop_mask_img, folder,
                                                                           image_sharp2, prev_y, crop_img, crop_img2,
                                                                           is_multiple)
                    prev_y = p
                    # print("+++++")
                    # print("maxy,div_mask_scale,maxy_red,div_height_scale",maxy,div_mask_scale,maxy_red,div_height_scale)
                    imagehough2 = cv2.resize(imagehough, (int(imagehough.shape[1] / 3), int(imagehough.shape[0] / 3)))
                    # cv2.imwrite(file,imagehough)
                    # print("maxy",maxy)
                    # print("imagehough.shape",imagehough.shape)
                    split_num = imagehough.shape[0]
                    # split_scale = int(imagehough.shape[0] / split_num)
                    split_scale = int(8)
                    water_level = 0
                    water_level2 = 0
                    # print("split_num",imagehough.shape,maxy,div_mask_scale,maxy_red,div_height_scale)
                    # print("maxy",maxy,split_num)
                    for h in range(split_num):
                        h_split_min = split_scale * h
                        h_split_max = split_scale * (h + 1)
                        # print(split_scale,h_split_max,maxy)
                        # print(h_split_min,h_split_max,0,crop_img.shape[1])
                        # split_img = crop_img[0:32,0:68]
                        split_img = imagehough[h_split_min:h_split_max, 0:imagehough.shape[1]]
                        # img_gray = cv2.cvtColor(split_img, cv2.COLOR_BGR2GRAY)
                        avg_value = np.average(split_img[:, :, 1:2])
                        water_level2 += 1 * (float(list_water_scale[i]) + (h * float(list_water_scale_div[i])))

                        # print("h_split_max:", h_split_max,h_split_min,maxy,maxy_red)
                        if maxy_red > 0 and maxy_red<1081:
                            maxy = maxy_red

                        if maxy_red>1081:
                            show_level_list.append(9999)
                            show_maxy_list.append(9999)
                            break
                        if h_split_max > maxy:
                            # if h_split_min>maxy :

                            a = ((float(list_water_scale_f1[i]) * (float(list_water_scale_f1_div[i]))) * (maxy * maxy))
                            b = (float(list_water_scale_f2[i]) * maxy)
                            # print("a,b",a,b)

                            water_level2 = a + b
                            c = max_level_m + (max_level_cm / 100)
                            d = c + water_level2 + float(list_water_scale[i])

                            # a = ((-4.1438588 * (1 / 10000000)) * (maxy * maxy))
                            # b = (0.00140098 * maxy)
                            # water_level2 = a - b
                            # c = max_level_m + (max_level_cm / 100)
                            # d = c + water_level2
                            # print("water_level",max_level_m,max_level_cm / 100,"-",float(list_water_scale_f2[i]), "a:",a, "b:",b, a + b, c, d,maxy,maxy_red)
                            # print("water_level",float(list_water_scale_f1[i]),float(list_water_scale_f1_div[i]),float(list_water_scale_f2[i]), a, b, a + b, c, d,maxy,maxy_red)
                            dec = (max_level_cm) - (water_level2)
                            show_level = (max_level_m) + dec / 100
                            # print("maxy-=-=-=-",maxy)
                            if maxy < 0:
                                show_level_list.append(9999)
                                show_maxy_list.append(maxy)
                            else:
                                # if d < 0:
                                #    d = 0
                                show_level_list.append(d)
                                show_maxy_list.append(maxy)
                            print("ddddd",d)

                            # print("water_level",max_level,water_level,max_level-water_level)
                            break

                # except:
                #     print("error detect water level ",'../RTU data/'+folder +'/images/'+ file)
                # continue
                # print(len(mask_img),len(show_level_list))
                '''
                if len(show_level_list) >= len(mask_img) :
                    show_level = min(show_level_list)
                    # print(show_level_list,maxy,show_level)
                    list_water_level.append(show_level)
                    list_img.append(folder + file)
                    # print(list_img,list_water_level)
                '''
            # print("========")
            if True:
                # if len(show_level_list)>0:
                show_level = min(show_level_list)
                min_idx = int(show_level_list.index(min(show_level_list)))
                show_maxy = show_maxy_list[min_idx]
                if show_level < 9999:
                    # print(show_level_list,maxy,show_level)
                    list_water_level.append(show_level)
                    list_img.append(file)
                    # print(list_img,list_water_level)
        # count_mask += 1
        cur_mask += 1
    print("getwaterlevel",list_water_level,list_img,show_maxy,show_maxy_list,show_level)
    return list_water_level, list_img, prev_y,show_maxy, is_night


def get_waterlevel(stn):
    mask_name_list = []
    folder = stn
    # print(stn)
    list_dir = os.listdir('../RTU data/' + folder + '/images')
    list_dir.sort(reverse=True)
    # print(list_dir)
    list_water_level = []
    list_img = []

    mask_list = list(pathlib.Path('static/' + folder).glob('*.txt'))
    mask_list.sort(reverse=True)
    # print(mask_list)
    # exit()
    # print(mask_list, mask_list[0])

    list_mask_date = []
    count_mask = 0
    list_water_scale_f1 = []
    list_water_scale_f1_div = []
    list_water_scale_f2 = []
    for msk in mask_list:
        mask_img = []
        list_max_level_cm = []
        list_max_level_m = []
        list_water_scale = []
        list_water_scale_div = []
        file1 = open(msk, 'r')
        Lines = file1.readlines()
        # print(msk)
        count = 0
        for line in Lines:
            count += 1
            split_line = line.strip().split(',')
            # print(folder + split_line[1] + ".jpg",split_line)
            mask = cv2.imread('static/' + folder + '/' + split_line[1] + ".jpg")
            # print(folder + split_line[1] + ".jpg")
            mask_img.append(mask)
            mask_name_list.append(split_line[1])
            list_max_level_m.append(float(split_line[2]))
            list_max_level_cm.append(float(split_line[3]))
            list_water_scale.append(split_line[4])
            list_water_scale_div.append(split_line[6])
            list_water_scale_f1.append(split_line[7])
            list_water_scale_f1_div.append(split_line[8])
            list_water_scale_f2.append(split_line[9])
            # 2022-11-21 14:41:53.261941
            # d1 = datetime.datetime(2020, 5, 13, 22, 50, 55)
            # print(int(split_line[5][0:4]),int(split_line[5][4:6]),int(split_line[5][6:8]),int(split_line[5][8:10]),int(split_line[5][10:12]))

            mask_name = "mask_2_46_0"
            mask_datetime_data = datetime.datetime(int(split_line[5][0:4]), int(split_line[5][4:6]),
                                                   int(split_line[5][6:8]), int(split_line[5][8:10]),
                                                   int(split_line[5][10:12]))
        list_mask_date.append(mask_datetime_data)
        print("list_dir", list_dir, cnt_file)

        cnt_file = 0
        for file in list_dir:
            if cnt_file == 0:
                file2 = file
            else:
                file2 = list_dir[cnt_file]
            cnt_file += 1
            show_level_list = []

            time = 1
            # image = cv2.imread('3.jpg')
            if "mask" in file:
                # mask = cv2.imread(folder + file)
                # mask_img.append(mask)
                continue
            # print("======")
            # print(folder+file)
            split_filename = file.split("_")
            # print(split_filename,split_filename[-1],split_filename[-2])
            # print(split_filename[-2][4:12])
            time = 1
            try:
                if int(split_filename[-2][8:10]) > 17 or int(split_filename[-2][8:10]) < 6:
                    # print("night")
                    time = 0
                else:
                    # print("day")
                    time = 1
            except:
                time = 1
            # print(split_filename,file)

            # Strips the newline character

            # print("Line{}: {}".format(count, line.strip()), len(list_water_scale), split_line[4])

            # print(folder + file,mask_name)
            # continue
            # exit()
            split_file = file.split("_")
            # print(split_file,folder +'/images/'+ file)
            file_date = datetime.datetime(int(split_file[-2][0:4]), int(split_file[-2][4:6]), int(split_file[-2][6:8]),
                                          int(split_file[-2][8:10]), int(split_file[-2][10:12]))
            image = cv2.imread('../RTU data/' + folder + '/images/' + file)
            image2 = cv2.imread('../RTU data/' + folder + '/images/' + file2)
            image = cv2.resize(image, (1920, 1080))
            image2 = cv2.resize(image2, (1920, 1080))
            print('../RTU data/' + folder + '/images/' + file, '../RTU data/' + folder + '/images/' + file2)
            # image = cv2.resize(image, (1280, 770))

            # cv2.imshow("d", image)
            c_mask = 0
            max_line = 0
            is_night = 0
            show_img_masking = image.copy()

            print("mask_img", i, len(mask_img))

            if len(list_mask_date) == 1:

                if file_date < list_mask_date[count_mask]:
                    # print(file_date, "<", list_mask_date[count_mask])
                    continue
            elif len(list_mask_date) > 1:
                # print(i,len(mask_img),len(list_mask_date))
                # print(i, file_date, list_mask_date[i],list_mask_date[i-1],file_date < list_mask_date[i],file_date >= list_mask_date[i-1])
                if file_date < list_mask_date[count_mask] or file_date >= list_mask_date[count_mask - 1]:
                    # print(file_date, "<", list_mask_date[i], "and", file_date, ">=", list_mask_date[count_mask - 1])
                    continue
            for i in range(len(mask_img)):
                # print("=====", count_mask, i, len(mask_img))
                # print(file)
                # Convert images to grayscale
                # print("mask", mask_img[i].shape)
                image_masking = image.copy()
                image_masking2 = image.copy()
                # print(i)
                split_mask_name = mask_name_list[i].split("_")
                # max_level_m = int(split_mask_name[2])
                # max_level_cm = int(split_mask_name[3])
                max_level_m = int(list_max_level_m[i])
                max_level_cm = int(list_max_level_cm[i])
                r_maskimage = mask_img[i]

                r_maskimage = cv2.resize(mask_img[i], (1920, 1080))
                print(image_masking.shape, r_maskimage.shape)
                image_masking[r_maskimage < 100] = 0
                image_masking2[r_maskimage < 100] = 0

                # print(image_masking.shape)
                img, x, y, w, h = getContour(image_masking, r_maskimage)
                crop_img = image_masking[y:y + h, x:x + w]
                crop_img2 = image_masking[y:y + h, x:x + w]

                crop_mask_img = r_maskimage[y:y + h, x:x + w]
                # print("===============",crop_img.shape,split_scale)

                # getContour2(crop_img, crop_img)
                t_lower = 50  # Lower Threshold
                t_upper = 150  # Upper threshold

                # Applying the Canny Edge filter
                edge = cv2.Canny(crop_img, t_lower, t_upper)
                kernel = np.ones((1, 5), np.uint8)

                kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                image_sharp = cv2.filter2D(src=crop_img, ddepth=-5, kernel=kernel_sharp)
                image_sharp = cv2.blur(image_sharp, (3, 3))
                image_sharp2 = cv2.filter2D(src=crop_img, ddepth=-5, kernel=kernel_sharp)
                image_sharp2 = cv2.blur(image_sharp2, (3, 3))
                image_sharp2 = crop_img
                # if True:
                try:
                    horizontalLines = cv2.erode(edge, kernel, iterations=1)
                    # print(image_sharp.shape)
                    imagehough, maxy, div_mask_scale, maxy_red, div_height_scale = getLineHough(image_sharp, time,
                                                                                                crop_mask_img, folder,
                                                                                                image_sharp2)
                    print("+++++", imagehough, maxy, div_mask_scale, maxy_red, div_height_scale)
                    # print("maxy,div_mask_scale,maxy_red,div_height_scale",maxy,div_mask_scale,maxy_red,div_height_scale)
                    imagehough2 = cv2.resize(imagehough, (int(imagehough.shape[1] / 3), int(imagehough.shape[0] / 3)))
                    # cv2.imwrite(file,imagehough)
                    # print(maxy)
                    # print("imagehough.shape",imagehough.shape)
                    split_num = imagehough.shape[0]
                    # split_scale = int(imagehough.shape[0] / split_num)
                    split_scale = int(8)
                    water_level = 0
                    water_level2 = 0

                    # print("maxy",maxy,split_num)
                    for h in range(split_num):
                        h_split_min = split_scale * h
                        h_split_max = split_scale * (h + 1)
                        # print(split_scale,h_split_max,maxy)
                        # print(h_split_min,h_split_max,0,crop_img.shape[1])
                        # split_img = crop_img[0:32,0:68]
                        split_img = imagehough[h_split_min:h_split_max, 0:imagehough.shape[1]]
                        img_gray = cv2.cvtColor(split_img, cv2.COLOR_BGR2GRAY)
                        avg_value = np.average(split_img[:, :, 1:2])
                        water_level2 += 1 * (float(list_water_scale[i]) + (h * float(list_water_scale_div[i])))

                        # print("h:", h,water_level2)
                        if maxy_red > 0:
                            maxy = maxy_red
                        if h_split_max > maxy:

                            # water_level = h * list_water_scale[i]
                            # dec = max_level_cm - water_level
                            # show_level = max_level_m + dec / 100
                            # mul_scale = (1/round(div_mask_scale, 2))-(float(list_water_scale_div[i])*1)
                            if folder == "STN22":
                                mul_scale = (1 - (div_height_scale / float(list_water_scale_div[i]))) + (
                                            float(list_water_scale_div[i]) / 100)
                            else:
                                mul_scale = (1 - (div_height_scale / float(list_water_scale_div[i])))
                            # print(div_height_scale, div_mask_scale, mul_scale, list_water_scale_div[i])
                            water_level = h * (float(list_water_scale[i]) * (mul_scale))

                            # print(h, max_level_cm, max_level_m, water_level,float(list_water_scale[i]), list_water_scale[i],len(list_water_scale))
                            # exit()
                            # print(list_water_scale_f1[i],list_water_scale_f1_div[i])
                            a = ((float(list_water_scale_f1[i]) * (float(list_water_scale_f1_div[i]))) * (maxy * maxy))
                            b = (float(list_water_scale_f2[i]) * maxy)
                            water_level2 = a + b
                            c = max_level_m + (max_level_cm / 100)
                            d = c + water_level2
                            # a = ((-4.1438588 * (1 / 10000000)) * (maxy * maxy))
                            # b = (0.00140098 * maxy)
                            # water_level2 = a - b
                            # c = max_level_m + (max_level_cm / 100)
                            # d = c + water_level2
                            # print("water_level",max_level_m,max_level_cm / 100,"-",float(list_water_scale_f2[i]), "a:",a, "b:",b, a + b, c, d,maxy,maxy_red)
                            print("water_level", float(list_water_scale_f1[i]), float(list_water_scale_f1_div[i]),
                                  float(list_water_scale_f2[i]), a, b, a + b, c, d, maxy, maxy_red)
                            dec = (max_level_cm) - (water_level2)
                            show_level = (max_level_m) + dec / 100
                            # print("maxy",maxy)
                            if maxy < 0:
                                show_level_list.append(999)
                            else:
                                show_level_list.append(d)
                            # print()

                            # print("water_level",max_level,water_level,max_level-water_level)
                            break

                except:
                    continue
                # print(len(mask_img),len(show_level_list))
                '''
                if len(show_level_list) >= len(mask_img) :
                    show_level = min(show_level_list)
                    # print(show_level_list,maxy,show_level)
                    list_water_level.append(show_level)
                    list_img.append(folder + file)
                    # print(list_img,list_water_level)
                '''
            # print("========")
            if True:
                # if len(show_level_list)>0:
                show_level = min(show_level_list)

                if show_level < 999:
                    # print(show_level_list,maxy,show_level)
                    prev_maxy = show_level
                    list_water_level.append(show_level)
                    list_img.append(folder + file)
                    # print(list_img,list_water_level)
        count_mask += 1
    # print(list_water_level)
    return list_water_level, list_img


####################################################

UPLOAD_FOLDER = './static'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/index")
def index():
    stn = 0
    stn_up = "-"
    if request.method == 'POST':
        # check if the post request has the file part
        file = request.files['file']
        stn = request.form.get("stn")
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            files = request.files.getlist("file")
            for file in files:
                # file.save(secure_filename(file.filename))
                filename = secure_filename(file.filename)
                filename_up = filename.split("/")
                print(filename, filename_up)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], "STN" + str(stn), filename))
    return render_template('index.html', name='Water Level Index', stn_up=stn)


@app.route("/addData")
def addData():
    stn = 0
    stn_up = "-"
    if request.method == 'POST':
        # check if the post request has the file part
        file = request.files['file']
        stn = request.form.get("stn")
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            files = request.files.getlist("file")
            for file in files:
                # file.save(secure_filename(file.filename))
                filename = secure_filename(file.filename)
                filename_up = filename.split("/")
                print(filename, filename_up)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], "STN" + str(stn), filename))
    return render_template('addData.html', name='Water Level Index', stn_up=stn)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# import pandas as pd


def showResTxtSearch(stn, start="20221229", end="20221230"):
    start_date = datetime.datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end, "%Y-%m-%d")
    daterange = end_date - start_date
    # print("daterange",daterange.days)
    html_str = "<button onclick='tableToCSV(\"" + stn + "\")'>Download CSV</button>"
    html_str += "<table style='width:100%;' id='myTable' name='myTable'>"
    html_str += "<thead>"
    html_str += "<tr>"
    html_str += "<th style='text-align: center;'>#</th>"
    html_str += "<th style='text-align: center;'>สถาน Station</th>"
    html_str += "<th style='text-align: center;'>Image</th>"
    html_str += "<th style='text-align: center;'>วันที่ - เวลา Date Time</th>"
    html_str += "<th style='text-align: center;'>ระดับน้ำ Water Level(msl)</th>"
    html_str += "</tr>"
    html_str += "</thead>"
    html_str += "<tbody>"
    html_str += "<tr style='display:none;'>"
    html_str += "<td style='text-align: center;'>#</td>"
    html_str += "<td style='text-align: center;'>Station</td>"
    html_str += "<td style='text-align: center;'>Image</th>"
    html_str += "<td style='text-align: center;'>Date Time</td>"
    html_str += "<td style='text-align: center;'>Water Level(msl)</td>"
    html_str += "</tr>"
    nmbr = 0
    if end_date == start_date:
        ranges = 0
    else:
        ranges = daterange.days
    print(ranges, daterange.days)
    mask_lists = list(pathlib.Path('static/' + stn).glob('*.txt'))
    # print(mask_lists)
    # exit()
    mask_list = []
    count_mask = len(mask_lists) - 1
    for f_m in mask_lists:
        split_f_m = str(f_m).split("\\")
        mask_file_ori = split_f_m[2]
        # print(split_f_m)
        m_name = split_f_m[0] + "/" + split_f_m[1] + "/" + mask_file_ori
        # m_name = split_f_m[0] + "/" + split_f_m[1] + "/mask_" + str(count_mask) + ".txt"
        mask_list.append(m_name)
        count_mask = count_mask - 1
    mask_list.sort(reverse=True)
    for daterange in range(0, ranges + 1):
        if True:
        # try:
            # dates = start_date + datetime.timedelta(days=daterange)
            # dates = end_date - datetime.timedelta(days=daterange)
            # if daterange==0:
            #    # dates = start_date + datetime.timedelta(days=1)
            #    dates = end_date + datetime.timedelta(days=1)
            # else:
            #    # dates = start_date + datetime.timedelta(days=daterange)
            #    dates = end_date + datetime.timedelta(days=daterange)
            dates = end_date - datetime.timedelta(days=daterange)
            dates = datetime.datetime.strftime(dates, "%Y-%m-%d")

            # print("dates",dates,'static/' + stn +'/csv/'+ stn + '_'+ str(dates) + '.csv')
            # file1 = pd.read_csv('static/' + stn +'/csv/'+ stn + '_'+ str(dates) + '.csv')
            # Lines = file1.sort_values(by=["DATETIME"], ascending=False)
            str_img_date = dates.replace("-", "")
            print('../RTU data/' + stn + '/' + stn + '_' + dates + '.csv', 'r')
            file1 = open('../RTU data/' + stn + '/' + stn + '_' + dates + '.csv', 'r')
            Lines = file1.readlines()
            Lines = Lines[1:]
            # Lines = csv.reader(file1)
            Lines.sort(reverse=True)
            # print(stn + '.txt')
            jpgFilenamesList = glob.glob(
                'static/' + stn + '/images/*' + str_img_date + '*.jpg')
            jpgFilenamesList.sort(reverse=True)
            print(len(jpgFilenamesList), len(Lines))
            row_count = 0
            for line in Lines:
                s_line = line.strip().split(',')
                # print(s_line, jpgFilenamesList[row_count])
                if len(s_line) > 1:
                    strdate = s_line[1]
                    strdate = strdate.replace("-", "")
                    strdate = strdate.replace(":", "")
                    strdate = strdate.replace(" ", "")
                    # subs = 'Geek'
                    # get_our
                    res = [i for i in jpgFilenamesList if strdate[:-2] in i]
                    # print("==========================================================")
                    # print(-2,len(res),s_line[1],strdate[:-2],s_line[1][14:16],s_line[1])
                    # print("--",res,strdate[:-2])
                    if len(res) == 0:
                        if s_line[1][14:16] != "00":
                            date_str2 = str(strdate[0:4]) + "-" + str(strdate[4:6]) + "-" + str(
                                strdate[6:8]) + " " + str(strdate[8:10]) + ":00:00"
                            given_time = datetime.datetime.strptime(date_str2, "%Y-%m-%d %H:%M:%S")
                            div_minutes = math.ceil(int(strdate[10:12]) / 15)
                            get_minutes = (div_minutes * 15)
                            # print(-4,len(res),s_line[1],strdate[:-4],res,strdate)
                            # strdate[10:12] = str(get_minutes)
                            res = [i for i in jpgFilenamesList if strdate[:-4] in i]
                            for xx in range(0, len(res)):
                                split_image = res[xx].split('\\')
                                split_image2 = split_image[1].split('_')
                                div_minutes_img = math.ceil(int(split_image2[2][10:12]) / 15)
                                get_minutes_img = (div_minutes_img * 15)
                                # print(get_minutes,strdate,strdate[10:12],"---",get_minutes_img,split_image2[2],split_image2[2][10:12])
                                if get_minutes == get_minutes_img:
                                    res[0] = res[xx]
                                # strdate = strdate.replace(strdate[0:12], strdate[0:10] + str(get_minutes))
                                # print("strdate2", strdate)
                        else:
                            # print("xxx===")
                            date_str2 = str(strdate[0:4]) + "-" + str(strdate[4:6]) + "-" + str(
                                strdate[6:8]) + " " + str(strdate[8:10]) + ":00:00"
                            given_time = datetime.datetime.strptime(date_str2, "%Y-%m-%d %H:%M:%S")
                            save_date2 = given_time - datetime.timedelta(minutes=55)
                            save_date2 = save_date2.strftime("%Y%m%d%H%M%S")

                            # div_minutes = round(int(split_img_name[-2][10:12]) / 15)
                            # get_minutes = (div_minutes * 15)
                            # save_date1 = given_time + datetime.timedelta(minutes=get_minutes)

                            # img_file = res[0]
                            # # print(save_date,strdate[:-2])
                            res = [i for i in jpgFilenamesList if save_date2[:-4] in i]
                            # res.sort(reverse=False)
                            # print("xxx", len(res),save_date2[:-4])
                    print("find in jpgFilenamesList", res, strdate)
                    nmbr += 1
                    # print("====",s_line,len(s_line))
                    # list_img_in_txt.append(s_line[3])
                    # print(split_img_path,split_img_name)
                    # exit()
                    # "{{url_for('static', filename='"+img[w]+"')}}"
                    print(len(res),mask_list)
                    offset_value = 0
                    status_mask = 0
                    for msk in mask_list:
                        if status_mask>1:
                            continue
                        file1 = open(msk, 'r')
                        Lines = file1.readlines()
                        print("msk",msk)
                        for line in Lines:
                            # count += 1
                            split_line = line.strip().split(',')
                            mask_datetime_data = datetime.datetime(int(split_line[5][0:4]), int(split_line[5][4:6]),
                                                                   int(split_line[5][6:8]), int(split_line[5][8:10]),
                                                                   int(split_line[5][10:12]))

                            print("mask_datetime_data", mask_datetime_data)
                            file_date = datetime.datetime.strptime(s_line[1],"%Y-%m-%d %H:%M:%S")
                            print("file_date", s_line[1])
                            if len(res) > 0:
                                if file_date < mask_datetime_data:
                                    print("1")
                                    status_mask = 1
                                    print(file_date, "<", mask_datetime_data)
                                    continue
                                else:
                                    # stat_mask += 1
                                    offset_value = split_line[4]
                                    # offset_value = 5
                                    status_mask = 2
                                    print("2")
                                    print(file_date, ">", mask_datetime_data)
                    # exit()
                    if True:
                        html_str += "<tr style='text-align: center;'>"
                        # html_str += "<td width='5%'>"+str(nmbr)+"-"+str(get_minutes)+"-"+str(div_minutes)+"-"+str(int(strdate[10:12]))+"-"+(strdate[10:12])+"</td>"
                        html_str += "<td width='5%'>" + str(nmbr) + "</td>"

                        html_str += "<td width='20%'>" + stn + "</td>"

                        html_str += "<td width='25%'><img src='" + res[0] + "' width='100%' height='100%'/></td>"
                        html_str += "<td width='25%'>" + s_line[1] + "</td>"
                        if stn == 'TC.12':
                            html_str += ""
                            html_str += "<td width='25%'>" + str(float(s_line[2]) + float(offset_value)) +"-"+str(s_line[2])+"-"+str(offset_value)+"</td>"
                        else:
                            html_str += ""
                            html_str += "<td width='25%'>" + str(float(s_line[2]) + float(offset_value))+"-"+str(s_line[2])+"-"+str(offset_value) + "</td>"
                        html_str += "</tr>"
                        row_count += 1
        # except:
        #     html_str += ""
    html_str += "</tbody>"
    html_str += "</table>"
    return html_str


def showResTxt(stn):
    file1 = open('static/' + stn + '.txt', 'r')
    Lines = file1.readlines()
    # print(stn + '.txt')
    html_str = "<table style='width:100%;' id='myTable' name='myTable'>"
    html_str += "<thead>"
    html_str += "<tr>"
    html_str += "<th style='text-align: center;'>#</th>"
    html_str += "<th style='text-align: center;'>สถาน Station</th>"
    # html_str += "<th style='text-align: center;'>Image</th>"
    html_str += "<th style='text-align: center;'>วันที่ - เวลา Date Time</th>"
    html_str += "<th style='text-align: center;'>ระดับน้ำ Water Level(msl)</th>"
    html_str += "</tr>"
    html_str += "</thead>"
    html_str += "<tbody>"
    html_str += "<tr style='display:none;'>"
    html_str += "<td style='text-align: center;'>#</td>"
    html_str += "<td style='text-align: center;'>Station</td>"
    # html_str += "<td style='text-align: center;'>Image</th>"
    html_str += "<td style='text-align: center;'>Date Time</td>"
    html_str += "<td style='text-align: center;'>Water Level(msl)</td>"
    html_str += "</tr>"
    nmbr = 0
    Lines.sort(reverse=True)
    for line in Lines:
        s_line = line.split(',')
        # print(s_line)
        if len(s_line) > 3:
            nmbr += 1
            # print("====",s_line,len(s_line))
            # list_img_in_txt.append(s_line[3])
            # print(split_img_path,split_img_name)
            # exit()
            # "{{url_for('static', filename='"+img[w]+"')}}"
            html_str += "<tr style='text-align: center;'>"
            html_str += "<td width='5%'>" + str(nmbr) + "</td>"
            html_str += "<td width='20%'>" + stn + "</td>"
            if True:
                # html_str += "<td width='25%'><img src='../RTU data/"+stn+"/images/"+s_line[3]+"' width='100%' height='100%'/></td>"
                html_str += "<td width='25%'>" + s_line[1] + "</td>"

            html_str += "<td width='25%'>" + str(s_line[2]) + "</td>"
            html_str += "</tr>"
    html_str += "</tbody>"
    html_str += "</table>"
    return html_str


def showRes(stn, waterlevel, img):
    html_str = "<table style='width:100%;' id='myTable' name='myTable'>"
    html_str += "<thead>"
    html_str += "<tr>"
    html_str += "<th style='text-align: center;'>#</th>"
    html_str += "<th style='text-align: center;'>สถาน Station</th>"
    html_str += "<th style='text-align: center;'>Image</th>"
    html_str += "<th style='text-align: center;'>วันที่ - เวลา Date Time</th>"
    html_str += "<th style='text-align: center;'>ระดับน้ำ Water Level(msl)</th>"
    html_str += "</tr>"
    html_str += "</thead>"
    html_str += "<tbody>"
    html_str += "<tr style='display:none;'>"
    html_str += "<td style='text-align: center;'>#</td>"
    html_str += "<td style='text-align: center;'>Station</td>"
    html_str += "<td style='text-align: center;'>Image</th>"
    html_str += "<td style='text-align: center;'>Date Time</td>"
    # html_str += "<td style='text-align: center;'>Water Level(msl)</td>"
    html_str += "</tr>"
    for w in range(len(waterlevel)):
        split_img_path = img[w].split('/')
        split_img_name = split_img_path[2].split('_')
        # print(split_img_path,split_img_name)
        # exit()
        # "{{url_for('static', filename='"+img[w]+"')}}"
        html_str += "<tr style='text-align: center;'>"
        html_str += "<td width='5%'>" + str(w + 1) + "</td>"
        html_str += "<td width='20%'>" + stn + "</td>"
        if True:
            html_str += "<td width='25%'><img src='../RTU data/" + stn + "/images/" + img[
                w] + "' width='100%' height='100%'/></td>"
            html_str += "<td width='25%'>" + str(split_img_name[-2][0:4]) + "/" + str(split_img_name[-2][4:6]) \
                        + "/" + str(split_img_name[-2][6:8]) + "-" + str(split_img_name[-2][8:10]) + "." + str(
                split_img_name[-2][10:12]) + "</td>"

        # html_str += "<td width='25%'>"+str(str(round(waterlevel[w], 2)))+"</td>"
        html_str += "</tr>"
    html_str += "</tbody>"
    html_str += "</table>"
    return html_str


@app.route("/getlistSTN", methods=['GET', 'POST'])
def getlistSTN():
    html_str = '<p style="font-size:20px;">'
    with open('static/station_name.txt') as f:
        lines = f.readlines()
        # print(lines)
        split_line = lines[0].split(',')
        for i in split_line:
            html_str += '<button><a onmouseover="" style="cursor: pointer;" id="' + i + '" onclick="get_STN(\'' + i + '\')">' + i + '</a></button> | '
    html_str += '<a onmouseover="" style=" cursor: pointer;" id="download"></a>'
    html_str += '<div id="upload" class="row"></div>'
    html_str += '</p>'

    return html_str


@app.route("/get_STN_searchs", methods=['GET', 'POST'])
def get_STN_searchs():
    html_str = ''
    with open('static/station_name.txt') as f:
        lines = f.readlines()
        # print(lines)

        split_line = lines[0].split(',')
        html_str += '<p style="font-size:20px;">'
        html_str += '<label for="stn">Station : </label>'
        html_str += '<select name="stn" id="stn">'
        html_str += '<option value = "-" > -Select Station-</option>'
        for i in split_line:
            html_str += '<option value = "' + i + '" > ' + i + ' </option>'
            # html_str += '<button><a onmouseover="" style="cursor: pointer;" id="' + i + '" onclick="get_STN(\'' + i + '\')">' + i + '</a></button> | '
        html_str += '</select>'
        html_str += '</p>'
        html_str += '<p style="font-size:20px;">'
        html_str += '<label for="start">Start date : </label><input value ="today();" type="date" id="start" name="start">'
        html_str += '</p>'
        html_str += '<p style="font-size:20px;">'
        html_str += '<label for="end">End date : </label><input value ="today();" type="date" id="end" name="end">'
        html_str += '</p>'
        html_str += '<p style="font-size:20px;">'
        html_str += '<button onclick="get_STN_search()"><a onmouseover="" style="cursor: pointer;" >Search</a></button>'
        html_str += '</p>'
        # html_str += '<button><a onmouseover="" style="cursor: pointer;" id="' + i + '" onclick="get_STN(\'' + i + '\')">' + i + '</a></button> | '
    # html_str += '<button><a onmouseover="" style="cursor: pointer;" id="add_STN" onclick="add_STN()">Add STN</a></button>'
    # html_str += '<div id="upload_file" class="form-group mb-4"><h3>Add STN</h3><form method=POST action="add_STN" enctype=multipart/form-data ><input type="file" name="file" directory="" /><input type="hidden" id="stn" name="stn" value="STN2"><input type="submit" value="Submit"></form></div>'

    return html_str


@app.route("/add_STN", methods=['GET', 'POST'])
def add_STN():
    html_str = '<p style="font-size:20px;">'
    with open('static/station_name.txt') as f:
        lines = f.readlines()
        # print(lines)
        split_line = lines[0].split(',')
        for i in split_line:
            html_str += '<button><a onmouseover="" style="cursor: pointer;" id="' + i + '" onclick="get_STN(\'' + i + '\')">' + i + '</a></button> | '
    # html_str += '<button><a onmouseover="" style="cursor: pointer;" id="add_STN" onclick="add_STN()">Add STN</a></button>'

    html_str += '</p>'

    return html_str


@app.route("/getlistSTNjs", methods=['GET', 'POST'])
def getlistSTNjs():
    '''

    function get_STN2(funct) {
  $('#loading').css('display', 'block');
  $.ajax({type : 'POST',url: '/get_STN',data : {'data':funct}, success: function(result){
    $("#div1").html(result);
    $('#loading').css('display', 'none');
    if(funct=="STN2"){
        $('#TSL32').html("TSL30");
        $('#TSL32').html("TSL32");
        $('#STN2').html("<b>STN2</b>");
        $('#STN3').html("STN3");
        $('#download').html(" | <b onclick=\"tableToCSV('STN2')\">Download CSV</b>");
        $('#upload').html('<div id="upload_file" class="form-group mb-4"><h3>Upload new File</h3><form method=POST action="upload_file" enctype=multipart/form-data ><input type="file" name="file" directory="" /><input type="hidden" id="stn" name="stn" value="STN2"><input type="submit" value="Submit"></form></div>');
    }
    else if(funct=="STN3"){
        $('#TSL32').html("TSL30");
        $('#TSL32').html("TSL32");
        $('#STN3').html("<b>STN3</b>");
        $('#STN2').html("STN2");
        $('#download').html(" | <b onclick=\"tableToCSV('STN3')\">Download CSV</b>");
        $('#upload').html('<div id="upload_file" class="form-group mb-4"><h3>Upload new File</h3><form method=POST action="upload_file" enctype=multipart/form-data ><input type="file" name="file" directory="" /><input type="hidden" id="stn" name="stn" value="STN3"><input type="submit" value="Submit"></form></div>');
    }
    else if(funct=="TSL32"){
        $('#TSL32').html("TSL30");
        $('#TSL32').html("<b>TSL32</b>");
        $('#STN2').html("STN2");
        $('#STN3').html("STN3");
        $('#download').html(" | <b onclick=\"tableToCSV('TSL32')\">Download CSV</b>");
        $('#upload').html('<div id="upload_file" class="form-group mb-4"><h3>Upload new File</h3><form method=POST action="upload_file" enctype=multipart/form-data ><input type="file" name="file" directory="" /><input type="hidden" id="stn" name="stn" value="TSL32"><input type="submit" value="Submit"></form></div>');
    }
    else if(funct=="TSL30"){
        $('#TSL32').html("<b>TSL30</b>");
        $('#TSL32').html("TSL32");
        $('#STN2').html("STN2");
        $('#STN3').html("STN3");
        $('#download').html(" | <b onclick=\"tableToCSV('TSL30')\">Download CSV</b>");
        $('#upload').html('<div id="upload_file" class="form-group mb-4"><h3>Upload new File</h3><form method=POST action="upload_file" enctype=multipart/form-data ><input type="file" name="file" directory="" /><input type="hidden" id="stn" name="stn" value="TSL30"><input type="submit" value="Submit"></form></div>');
    }
    $('#myTable').DataTable( {
        responsive: true,

    } );
  }});
}
    function get_STN2(funct) {
  $('#loading').css('display', 'block');
  $.ajax({type : 'POST',url: '/get_STN',data : {'data':funct}, success: function(result){
    $("#div1").html(result);
    $('#loading').css('display', 'none');
    '''
    html_str = '<script>'
    html_str += 'function get_STN(funct) {'
    html_str += '$("#loading").css("display", "block");'
    html_str += '$.ajax({type : "POST",url: "/get_STN",data : {"data":funct}, success: function(result){'
    html_str += '$("#div1").html(result);'
    html_str += '$("#loading").css("display", "none");'
    with open('static/station_name.txt') as f:
        lines = f.readlines()

        split_line = lines[0].split(',')
        count_stn = 0
        # str_update = ''
        for i in split_line:
            if count_stn < 1:
                html_str += 'if (funct == "' + i + '"){'
            else:
                html_str += 'else if (funct == "' + i + '"){'
            for j in split_line:
                if i == j:
                    html_str += '$("#' + j + '").html("<b>' + j + '</b>");'
                    # str_update = i
                else:
                    html_str += '$("#' + j + '").html("' + j + '");'
            html_str += '$("#download").html(" | <b onclick=\'tableToCSV(\\\"' + i + '\\\")\'>Download CSV</b>");'
            html_str += '$("#upload").html("<div style=\'display:none;\' id=\'upload_file\' class=\'form-group mb-4\'><h3>Upload new File</h3><form method=POST action=\'upload_file\' enctype=multipart/form-data ><input type=\'file\' name=\'file\' directory=\'\' /><input type=\'hidden\' id=\'stn\' name=\'stn\' value=\'' + i + '\'><input type=\'submit\' value=\'Submit\'></form></div>");'
            html_str += '}'
            count_stn += 1
            # html_str += '<button><a onmouseover="" style="cursor: pointer;" id="'+i+'" onclick="get_STN(\''+i+'\')">'+i+'</a></button> | '
    html_str += '$("#myTable").DataTable( {responsive: true,order: [[2, "desc"]],'
    html_str += '} );'
    html_str += '}});'
    html_str += '}'
    # html_str += 'if ("'+str_update+'" !=="0"){'
    # html_str += 'get_STN("' + str_update + '")'
    # html_str += ' }'
    html_str += '</script>'
    return html_str


import glob


# Python program to get average of a list
def Average(lst):
    return sum(lst) / len(lst)

def sum1forline(filename):
    with open(filename) as f:
        return sum(1 for line in f)

def upDateResTxt(stn, list_new_img_in_rtu):
    html_str1 = ""
    html_str2 = ""
    html_str3 = ""
    now = datetimes.now()
    date_str = now.strftime("%Y-%m-%d")
    date_str2 = date_str.replace("-", "")
    path_to_file_img = stn + '_' + date_str + '_img.csv'
    isstnExist = os.path.exists('static/' + stn)
    if isstnExist == False:
        os.makedirs('static/' + stn + '/')
    isstnimgExist = os.path.exists('static/' + stn + '/images')
    if isstnimgExist == False:
        os.makedirs('static/' + stn + '/images')
    isstncsvExist = os.path.exists('../RTU data/' + stn + '/csv')
    if isstncsvExist == False:
        os.makedirs('../RTU data/' + stn + '/csv')
    datestryearmonth = date_str2[:-2]
    #datestryearmonth = "202310"
    is_newcsv = 0
    year = date_str2[0:4]
    if int(date_str2[4:6]) + 1 > 12:
        month = "01"
        year = str(int(date_str2[0:4]) + 1)
    elif int(date_str2[-2]) + 1 >= 10:
        month = str(int(date_str2[4:6]) + 1)
    elif int(date_str2[-2]) + 1 < 10:
        month = "0" + str(int(date_str2[4:6]) + 1)
    if datestryearmonth == "202310":
        string = "202310"
    else:
        string = datestryearmonth
    # exit()
    html_str = ''
    list_image2 = os.listdir('static/' + stn + '/images/')
    list_image2.sort()
    # print(len(list_image2))
    str_list = [string, year + month]
    #print(str_list )
    #exit()
    full_list = os.listdir('../RTU data/' + stn + '/images/')
    jpgFilenamesList = [nm for ps in str_list for nm in full_list if ps in nm]

    # jpgFilenamesList = glob.glob('../RTU data/'+stn+'/images/*['+datestryearmonth+'-'+year+month+']*.jpg')
    # print(str_list,full_list,'../RTU data/' + stn + '/images/',len(jpgFilenamesList))
    # exit()
    cnt = 0
    prev_img = 0
    prev_y = 0
    for new_img_rtus in jpgFilenamesList:
        is_data = 0
        if cnt == 0:
            cnt += 1
            prev_img = '../RTU data/' + stn + '/images/' + new_img_rtus
            # new_img_rtu2 = new_img_rtu
            continue
        else:
            new_img_rtu2 = prev_img
        new_img_rtu = '../RTU data/' + stn + '/images/' + new_img_rtus
        split_new_img_rtu = new_img_rtu.split('/')
        print(new_img_rtu, new_img_rtu2)
        prev_img = new_img_rtu
        # print(split_new_img_rtu[1] in list_image2,split_new_img_rtu)
        if split_new_img_rtu[-1] in list_image2:
            # print("data exist in folder")
            is_data = 1
        else:
            is_data = 0

        if is_data == 0:
            # print("data not exist", is_data,new_img_rtu)

            try:
                rtudata = cv2.imread(new_img_rtu)
                # rtudata = cv2.resize(rtudata, (640, 320))
                rtudata = cv2.resize(rtudata, (1920, 1080))
            except:
                continue
            if True:
                split_img = new_img_rtu.split('/')
                str_img = split_img[-1]
                str_img = str_img.split('_')
                yearmonthimg = str_img[-2][0:8]
                # print("yearmonthimg",yearmonthimg)
                # exit()

                path_to_file = stn + '_' + yearmonthimg[0:4] + '-' + yearmonthimg[4:6] + '-' + yearmonthimg[
                                                                                               6:8] + '.csv'
                filedate = yearmonthimg[0:4] + '-' + yearmonthimg[4:6] + '-' + yearmonthimg[6:8]
                datetime_object = datetime.datetime.strptime(filedate, "%Y-%m-%d")

                prev_datefile = datetime_object - datetime.timedelta(days=1)
                path_to_file2 = stn + '_' + prev_datefile.strftime("%Y-%m-%d") + '.csv'

                isExist = os.path.exists('../RTU data/' + stn + '/' + path_to_file)
                # print('static/' + stn + '/csv/' + path_to_file,isExist)
                val_av_prev_f = 0
                if isExist:
                    # file1 = open('../RTU data/' + stn + '/csv/' + path_to_file, 'r')
                    is_newcsv = 0
                    st = 1
                    countdatat_file = sum1forline('../RTU data/' + stn + '/' + path_to_file)

                else:
                    file1 = open('../RTU data/' + stn + '/' + path_to_file, 'w', newline='')
                    writer = csv.writer(file1)
                    writer.writerow(["DEVICE", "DATETIME", "WL"])
                    is_newcsv = 1
                    st = 0
                    file1.close()
                file1 = open('../RTU data/' + stn + '/' + path_to_file, 'r')
                Lines_wl = file1.readlines()

                file1.close()
                val_last_prev = 0
                val_last_prev_last = 0
                last_maxy = 0
                last_is_night = 0

                if True:

                    try:
                        # print("===is_new0")
                        file_prev = open('../RTU data/' + stn + '/' + path_to_file2, 'r')
                        Lines_file_prev = file_prev.readlines()
                        cnt_prev = 0
                        for line_prev in Lines_file_prev:
                            if cnt_prev > 0:
                                line_str_prev = line_prev.strip()
                                line_split_prev = line_str_prev.split(",")
                                val_av_prev.append(float(line_split_prev[2]))
                            cnt_prev += 1

                        val_av_prev_f = (val_av_prev[-1] + val_av_prev[-2] + val_av_prev[-3] + val_av_prev[-4]) / 4
                        val_last_prev_last = val_av_prev[-1]
                        print(val_last_prev)
                        file_prev.close()
                        is_new = 0
                    except:
                        print("===is_new1")
                        # val_av_prev_f = 0.01
                        # val_last_prev_last = 0.01
                        is_new = 1
                        # exit()
                # exit()
                count_wl = 0
                x_wl = []
                y_wl = []
                list_wl = []
                list_wl2 = []
                list_wl_ = []
                list_mov_avg = []
                wl_max = 2.2
                wl_min = -6
                wl_lim_avg = 0.3
                count_wl = 0

                for line_wl in Lines_wl:
                    if count_wl < 1:
                        count_wl += 1
                        continue
                    count_wl += 1
                    line_str = line_wl.strip()
                    line_split = line_str.split(",")
                    print(line_split)
                    list_wl2.append(float(line_split[2]))
                    list_wl_.append(float(line_split[2]))
                    try:
                        last_maxy = int(line_split[4])
                        last_is_night = line_split[3]
                    except:
                        last_maxy = 0
                        last_is_night = 0
                    if len(list_wl_) > 1:
                        # print(list_wl)
                        print(float(line_split[2]), "-", (list_wl_[-2]))
                        mov = (float(line_split[2]) - (list_wl_[-2]))
                        list_mov_avg.append(abs(mov))
                        # if float(line_split[2]) < wl_max and float(line_split[2]) > wl_min:
                        #     # print(line_split[2],mov)
                        #     list_wl.append(float(line_split[2]))
                        #     list_wl_.append(float(line_split[2]))
                        #     list_mov_avg.append(abs(mov))
                        #     print(":list_wl", len(list_wl), list_wl,line_wl)
                    else:

                        list_mov_avg.append(0.1)
                        # list_mov_avg.append(val_av_prev_f)
                        # if float(line_split[2]) < wl_max and float(line_split[2]) > wl_min:
                        #     list_wl.append(wl)
                        #     list_wl_.append(float(line_split[2]))
                        #     list_mov_avg.append(0.1)
                #         #     print(":list_wl", len(list_wl), list_wl,line_wl)
                # print("-list_mov_avg",len(list_mov_avg),list_mov_avg)
                # print("-list_wl2",len(list_wl2),list_wl2)
                # exit()
                if len(list_wl_) > 4:
                    list_wl = list_wl_[len(list_wl_) - 4:]
                    list_mov_avg = list_mov_avg[-4:]
                else:
                    list_wl = list_wl_
                    list_mov_avg = list_mov_avg
                # print("-list_wlcut", len(list_wl), list_wl, wl)
                print("-list_mov_avgcut", len(list_mov_avg), list_mov_avg)
                str_saveimg = new_img_rtu.replace('../RTU data', 'static/')
                # print("str_saveimg",str_saveimg)

                waterlevel, img, p,maxy, is_night = get_waterlevel_newdata(stn, new_img_rtu, new_img_rtu2, prev_y,last_maxy,last_is_night)
                prev_y = p
                print("maxy-maxy",maxy,waterlevel)
                if maxy<1081:
                    rtudata[maxy,:] = 255
                    cv2.imwrite(str_saveimg, rtudata)
                    # cv2.imwrite(str_saveimg, rtudata)
                    # print("===", img[w], list_img_in_txt[w], img[w].find(list_img_in_txt[w]))
                    # print("===",img[w],l_img)
                    # print("waterlevel, img",waterlevel, img)
                    # exit()
                    cnt += 1
                    countdatat_file = 0
                    val_av_prev = []

                    wl = waterlevel[0]
                    # wl = ((line_split[2]))
                    wl = float(wl)
                    # elif len(list_wl) == 4:
                    #     list_wl = list_wl[-4:]
                    #     list_mov_avg = list_mov_avg[-4:]
                    # elif len(list_wl) == 3:
                    #     list_wl = list_wl[-3:]
                    #     list_mov_avg = list_mov_avg[-3:]
                    # elif len(list_wl) == 2:
                    #     list_wl = list_wl[-2:]
                    #     list_mov_avg = list_mov_avg[-2:]
                    # print("list_wl2", len(list_wl), list_wl, wl)
                    if len(list_wl) == 4:
                        mov = ((wl) - (list_wl[-1]))
                        average = Average(list_wl)
                        mov_average = Average(list_mov_avg)
                        direction = list_wl[3] - list_wl[2]
                        print((wl), (list_wl[-2]), abs(mov), ">", wl_lim_avg, "or", wl, ">", wl_max, "or", wl, "<", wl_min)
                        if (abs(mov) > wl_lim_avg) :#or wl > wl_max or wl < wl_min:
                            avg_wl = (list_wl[-2] + list_wl[-1]) / 2
                            if direction < 0:
                                norm = avg_wl - abs(mov_average)
                                print("down", wl, list_wl[-2], list_wl[-1], avg_wl, norm, list_wl, list_wl[-1],
                                      list_mov_avg, mov_average, list_wl[2], list_wl[1])
                            else:
                                norm = avg_wl + abs(mov_average)
                                print("up", wl, list_wl[-2], list_wl[-1], avg_wl, norm, list_wl, list_wl[-1], list_mov_avg,
                                      mov_average)
                            # norm = average
                            # list_wl[3] = norm
                            wl = norm
                        # print(list_wl, wl, list_wl[-2], abs(abs(wl) - abs(list_wl[-2])), 2 * (abs(average)), average)
                    elif len(list_wl) == 3:
                        mov = ((wl) - (list_wl[-1]))
                        average = Average(list_wl)
                        mov_average = Average(list_mov_avg)
                        direction = list_wl[2] - list_wl[1]
                        print((wl),list_wl, (list_wl[-2]), abs(mov), ">", wl_lim_avg, "or", wl, ">", wl_max, "or", wl, "<", wl_min)
                        if (abs(mov) > wl_lim_avg) :#or wl > wl_max or wl < wl_min:
                            avg_wl = (list_wl[-2] + list_wl[-1]) / 2
                            if direction < 0:
                                norm = avg_wl - abs(mov_average)
                                print("down", wl, list_wl[-2], list_wl[-1], avg_wl, norm, list_wl, list_wl[-1],
                                      list_mov_avg, mov_average, list_wl[2], list_wl[1])
                            else:
                                norm = avg_wl + abs(mov_average)
                                print("up", wl, list_wl[-2], list_wl[-1], avg_wl, norm, list_wl, list_wl[-1], list_mov_avg,
                                      mov_average)
                            # norm = average
                            # list_wl[3] = norm
                            wl = norm
                        # print(list_wl, wl, list_wl[-2], abs(abs(wl) - abs(list_wl[-2])), 2 * (abs(average)), average)
                    elif len(list_wl) == 2:
                        average = Average(list_wl)
                        mov = ((wl) - (list_wl[-1]))
                        if (abs(mov) > wl_lim_avg) :#or wl > wl_max or wl < wl_min:
                            norm = average
                            wl = norm
                        print("2", list_wl, wl, list_wl[-1])
                    else:
                        mov = ((wl) - (val_last_prev))
                        if is_new == 0:
                            if (abs(mov) > wl_lim_avg):
                                norm = val_last_prev_last
                                wl = norm
                        print("1", list_wl,val_av_prev)

                    # print('static/' + stn + '/csv/' + path_to_file, 'a')
                    with open('../RTU data/' + stn + '/' + path_to_file, 'a', newline='') as f:
                        # print("=-=-=-=-=-=-=-=-=-")
                        write_object = csv.writer(f)

                        # heading = next(f)
                        last_water_level = 0
                        # reader_obj = csv.reader(f)
                        # print(datestryearmonth,'../RTU data/'+stn+'/images/[2022-'+datestryearmonth[0:4]+'].jpg')
                        if datestryearmonth[-2:] == 12:
                            year = str(int(datestryearmonth[0:4]) + 1)
                            month = "01"
                        else:
                            year = str(int(datestryearmonth[0:4]))
                            if int(datestryearmonth[-2:]) + 1 > 9:
                                month = str(int(datestryearmonth[-2:]) + 1)
                            elif int(datestryearmonth[-2:]) + 1 < 10:
                                month = str(int(datestryearmonth[-2:]) + 1)
                        for w in range(len(waterlevel)):
                            waterlevel[w] = wl
                        # print("waterlevel",waterlevel)
                        # exit()
                        for w in range(len(waterlevel)):
                            if st == 0:
                                last_water_level = 0
                                st = 1
                            else:
                                last_water_level = waterlevel[w]
                            # is_data = 0
                            # print("w",w)

                            html_str = ""
                            # split_img_path = img[w].split('/')
                            # print("data water level",w,img)
                            split_img_name = img[w].split('_')
                            date_str = str(split_img_name[-2][0:4]) + "-" + str(split_img_name[-2][4:6]) + "-" + str(
                                split_img_name[-2][6:8]) \
                                       + " " + str(split_img_name[-2][8:10]) + ":" + str(split_img_name[-2][10:12]) + ":00"
                            date_str2 = str(split_img_name[-2][0:4]) + "-" + str(split_img_name[-2][4:6]) + "-" + str(
                                split_img_name[-2][6:8]) \
                                        + " " + str(split_img_name[-2][8:10]) + ":00:00"
                            given_time = datetime.datetime.strptime(date_str2, "%Y-%m-%d %H:%M:%S")
                            div_minutes = math.ceil(int(split_img_name[-2][10:12]) / 15)
                            get_minutes = (div_minutes * 15)
                            save_date = given_time + datetime.timedelta(minutes=get_minutes)
                            print(waterlevel[w],wl)
                            diff = abs(last_water_level - round(waterlevel[w], 2))
                            # print(date_str)
                            # print("diff",diff,last_water_level,round(waterlevel[w], 2))
                            if is_newcsv:
                                curr_wl = round(waterlevel[w], 2)
                            else:
                                if diff > 0.1:
                                    curr_wl = last_water_level
                                else:
                                    curr_wl = round(waterlevel[w], 2)
                            if is_data == 0:
                                html_str1 = stn
                                html_str2 = save_date
                                html_str3 = str(curr_wl)
                                html_str4 = str(is_night)
                                html_str5 = str(maxy)
                                html_str = img[w]
                                # file1.write('\n'+html_str)
                                print("write_object", html_str1, html_str2, html_str3, html_str4, html_str5)
                                write_object.writerow([html_str1, html_str2, html_str3, html_str4, html_str5])
                            # print("write_object",html_str1,html_str2,html_str3)
                    f.close()

    return html_str1, html_str2, html_str3


from pathlib import Path


def cek_newData(stn):
    # datetime object containing current date and time
    now = datetimes.now()

    # print("now =", now)

    # dd/mm/YY H:M:S
    minute = now.strftime("%M")

    # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string = now.strftime("%Y%m%d%H0000")
    current_time = datetime.datetime.now()
    # print("date and time =", dt_string)
    # print(request.form['data'])
    station = stn
    isExist = os.path.exists("static/" + stn + '.txt')
    if isExist:
        file1 = open('static/' + stn + '.txt', 'r')
        # print("txt exist")
    else:
        file1 = open("static/" + stn + '.txt', 'w')

    list_new_img_in_rtu = []
    # print(pathlib.Path("../RTU data/"+station+"/images/"))
    # jpg_list = list(pathlib.Path("../RTU data/"+station+"/images/").glob('*.jpg'))
    # jpg_list = os.listdir("../RTU data/"+station+"/images/")
    # sorted(filter(os.path.isfile, jpg_list), key=os.path.getmtime)
    # print("../RTU data/"+station+"/images/")
    jpg_list = sorted(Path("../RTU data/" + station + "/images/").iterdir(), key=os.path.getmtime)
    # print(jpg_list)
    # file1 = open("static/"+stn + '.txt', 'a')
    # Lines = file1.readlines()
    # print("file1",len(Lines),"static/"+stn + '.txt')
    # exit()
    # if len(Lines)>1:
    #     jpg_list.reverse()
    # print("jpg_list",len(jpg_list),jpg_list)
    # jpg_list.sort(reverse=True)
    # print(jpg_list[0],"../RTU data/"+station+"/images/")
    a = 0
    for img_st in jpg_list:
        # for line in Lines:
        # print(img_st)
        if True:
            img = str(img_st)
            # img_split = s_line[3].split('/')
            img_str1 = img.split('\\')
            img_str = img_str1[4].split('_')
            # print(img_str1,img_str)
            if len(img_str) > 3:
                if img_str1[2][0:2] == "ST":
                    date_time_rtu = datetime.datetime(int(img_str[-2][0:4]), int(img_str[-2][4:6]),
                                                      int(img_str[-2][6:8]),
                                                      int(img_str[-2][8:10]), int(img_str[-2][10:12]))
                else:
                    date_time_rtu = datetime.datetime(int(img_str[-2][0:4]), int(img_str[-2][4:6]),
                                                      int(img_str[-2][6:8]),
                                                      int(img_str[-2][8:10]), int(img_str[-2][10:12]))
                date_time_now = datetime.datetime(int(dt_string[0:4]), int(dt_string[4:6]), int(dt_string[6:8]),
                                                  int(dt_string[8:10]), int(dt_string[10:12]))
                # print(date_time_rtu,">",date_time_now)
                if date_time_rtu > date_time_now:
                    # print("newdata")
                    # print(s_line)
                    # if len(s_line) > 3:
                    a = 1
                    # print("rtu has new data","../RTU data/"+station+"/images/",img_str )
                    if True:
                        # print("list_new_img_in_rtu",img_str1[4])
                        list_new_img_in_rtu.append(img_str1[4])
                else:
                    continue
        # print(list_img_in_txt)
        img_folder_split = str(img).split('\\')
        # for line in Lines:
        #     s_line = line.split(',')
        #     # print(len(img_folder_split),img_folder_split)
        #     # if img_folder_split[3] in list_img_in_txt:
        #     if list_img_in_txt in list_img_in_txt:
        #         # print("====exist image", img)
        #         a=1
        #     else:
        #         # print("====new image", img)
        #         a=0
    count = 0
    # Strips the newline character
    html_str = ""
    return a, list_new_img_in_rtu


@app.route("/get_STNtxt", methods=['GET', 'POST'])
def get_STN_from_txt():
    # print(request.form['data'])
    if request.method == "POST":
        stn = request.form['data']
    html_str = ""
    waterlevel, img = get_waterlevel("static/" + stn + "/")
    html_str = showRes(stn, waterlevel, img)
    return html_str


@app.route("/upDateStation", methods=['GET', 'POST'])
def upDateStation():
    print("upDateResTxt")
    html_str = "upDateResTxt"
    with open('static/station_name.txt') as f:
        lines = f.readlines()
        # print(lines)
        split_line = lines[0].split(',')
        for stn in split_line:
            is_new, list_new_img_in_rtu = cek_newData(stn)
            print(list_new_img_in_rtu)
            if True:
                # if is_new==1:
                upDateResTxt(stn, list_new_img_in_rtu)
    return html_str


@app.route("/get_STN_search", methods=['GET', 'POST'])
def get_STN_search():
    # print(request.form['data'])
    stn = ""

    if request.method == "POST":
        print("request.form['data']", request.form['data'])
        stn = request.form['data']
        start = request.form['start']
        end = request.form['end']

    html_str = showResTxtSearch(stn, start, end)
    return html_str


@app.route("/get_STN", methods=['GET', 'POST'])
def get_STN():
    # print(request.form['data'])
    stn = ""

    if request.method == "POST":
        stn = request.form['data']
    # cek_newData(stn)
    # exit()
    html_str = ""

    # waterlevel, img = get_waterlevel(stn)
    # html_str = showRes(stn,waterlevel, img)
    # upDateResTxt(stn)
    html_str = showResTxt(stn)
    return html_str


@app.route("/get_STN_ori", methods=['GET', 'POST'])
def get_STN_ori():
    # print(request.form['data'])
    if request.method == "POST":
        stn = request.form['data']
    html_str = ""
    waterlevel, img = get_waterlevel("" + stn + "/")
    html_str = showRes(stn, waterlevel, img)
    showRes(stn, waterlevel, img)
    return html_str


@app.route("/upload_images", methods=['GET', 'POST'])
def upload_images():
    if request.method == 'POST':
        # check if the post request has the file part
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            files = request.files.getlist("file")
            for file in files:
                # file.save(secure_filename(file.filename))
                filename = secure_filename(file.filename)
                filename_up = filename.split("/")
                print(filename, filename_up)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('index.html', name='Water Level Upload')


@app.route("/upload_file", methods=['GET', 'POST'])
def upload_file():
    stn = '0'
    stn_up = '0'
    if request.method == 'POST':
        # check if the post request has the file part
        file = request.files['file']
        stn = request.form.get("stn")
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            files = request.files.getlist("file")
            for file in files:
                # file.save(secure_filename(file.filename))
                filename = secure_filename(file.filename)
                filename_up = filename.split("/")

                file.save(os.path.join(app.config['UPLOAD_FOLDER'], str(stn), filename))
    # print(stn)
    return render_template('index.html', name='Water Level Upload', stn_up=stn)


@app.route("/schedulerPage", methods=['GET', 'POST'])
def schedulerPage():
    return render_template('schedulerPage.html', name='Scheduler DUration')


scheduler = BackgroundScheduler()


@app.route("/upd_scheduler", methods=['GET', 'POST'])
def upd_scheduler():
    if request.method == 'POST':
        # check if the post request has the file part
        time_minutes = request.form.get("time_minutes")
        print("time_minutes", time_minutes)
        file1 = open("static/scheduler_update.txt", 'w')
        file1.write(time_minutes)
        file1.close
    file1 = open("static/scheduler_update.txt", 'r')
    Lines = file1.readlines()
    interval = Lines[0]
    # print(Lines, len(Lines))
    second = interval * 60
    html = '<form action="/action_page.php">'
    html += '<label for="time_minutes">Update duration (minutes):</label><br>'
    html += '<input type="text" id="time_minutes" name="time_minutes" value="' + interval + '"><br>'
    html += '<input type="button" value="Submit" onclick="upd_scheduler()">'
    html += '</form>'
    file1.close
    start_scheduler("update")
    return html


@app.route("/get_scheduler", methods=['GET', 'POST'])
def get_scheduler():
    file1 = open("static/scheduler_update.txt", 'r')
    Lines = file1.readlines()
    interval = Lines[0]
    # print(Lines, len(Lines))
    second = interval * 60
    html = '<form action="/action_page.php">'
    html += '<label for="time_minutes">Update duration (minutes):</label><br>'
    html += '<input type="text" id="time_minutes" name="time_minutes" value="' + interval + '"><br>'
    html += '<input type="button" value="Submit" onclick="upd_scheduler()">'
    html += '</form>'
    file1.close
    return html


def start_scheduler(status="start"):
    if True:
        global scheduler
        if status == "start":
            print("Start")
            schedulers = scheduler
            upDateStation()
        else:
            print("Update", scheduler)
            scheduler.shutdown()
            scheduler = BackgroundScheduler()
            schedulers = scheduler
        file1 = open("static/scheduler_update.txt", 'r')
        Lines = file1.readlines()
        interval = Lines[0]
        second = float(interval) * 60
        schedulers.add_job(func=upDateStation, trigger="interval", seconds=second)
        schedulers.start()
    # except:
    #     file1 = open("static/scheduler_update.txt", 'r')
    #     Lines = file1.readlines()
    #     interval = Lines[0]
    #     print("Continue",Lines, len(Lines))
    #     second = float(interval)*60
    #     schedulers = BackgroundScheduler()
    #     schedulers.add_job(func=upDateStation, trigger="interval", seconds=second)
    #     schedulers.start()

    print(scheduler, schedulers)


if __name__ == '__main__':
    # upDateStation()
    start_scheduler()
    app.run(host='127.0.0.1', port=5000)
 # python server.py
