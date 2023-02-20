# por executer python .\Detect_Ocr_Mrz.py --image images/teste.jpg
from skimage.filters import threshold_local
from imutils.contours import sort_contours
import numpy as np
import pytesseract
import argparse
import imutils
import sys
import cv2
# from pandas import DataFrame
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\tmp_wade62890\Tesseract-OCR\tesseract.exe'
# load the image and compute the ratio of the old height
# to the new height, clone it, and resize  it
from transform import four_point_transform

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)
# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
# show the original image and the edge detected image
print("STEP 1 Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# loop over the contours

for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

# show the contour (outline) of the piece of paper
print("STEP 2 Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down
# view od the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# convert the warped image to grayscale, the threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method="gaussian")
# warped = (warped > T).astype("uint8") * 255

# show the original and scanned images
print("STEP 3 Apply perspective transform")
cv2.imshow("original", imutils.resize(orig, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey(0)
image = cv2.resize(warped, (1000, 450))
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(H, W) = image.shape
# initialize a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
# smooth the image using a 3x3 Gaussian blur and then apply a
# blackhat morpholigical operator to find dark regions on a light
# background
# gray = cv2.GaussianBlur(gray, (3, 3), 0)
blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, rectKernel)
cv2.imshow("Blackhat", blackhat)

# compute the Scharr gradient of the blackhat image and scale the
# result into the range [0, 255]
grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
grad = np.absolute(grad)
(minVal, maxVal) = (np.min(grad), np.max(grad))
grad = (grad - minVal) / (maxVal - minVal)
grad = (grad * 255).astype("uint8")
cv2.imshow("Gradient", grad)

# apply a closing operation using the rectangular kernel to close
# gaps in between letters -- then apply Otsu's thresholding method
grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(grad, 0, 255,
                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Operation Fermiture", thresh)
# perform another closing operation, this time using the square
# kernel to close gaps between lines of the MRZ, then perform a
# series of erosions to break apart connected components
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
thresh = cv2.erode(thresh, None, iterations=2)
cv2.imshow("Operaration de fermiture", thresh)

# find contours in the thresholded image and sort them from bottom
# to top (since the MRZ will always be at the bottom of the passport)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="bottom-to-top")[0]
# initialize the bounding box associated with the MRZ
mrzBox = None

# lo
for c in cnts:
    # compute the bounding box of the contour and then derive the
    # how much of the image the bounding box occupies in terms of
    # both width and height
    (x, y, w, h) = cv2.boundingRect(c)
    percentWidth = w / float(W)
    percentHeight = h / float(H)
    # if the bounding box occupies > 80% width and > 4% height of the
    # image, then assume we have found the MRZ
    if percentWidth > 0.8 and percentHeight > 0.04:
        mrzBox = (x, y, w, h)
        break

# if the MRZ was not found, exit the script
if mrzBox is None:
    print("[INFO] MRZ could not be found")
    sys.exit(0)
# pad the bounding box since we applied erosions and now need to
# re-grow it
(x, y, w, h) = mrzBox
pX = int((x + w) * 0.03)
pY = int((y + h) * 0.03)
(x, y) = (x - pX, y - pY)
(w, h) = (w + (pX * 2), h + (pY * 2))
# extract the padded MRZ from the image
mrz = image[y:y + h, x:x + w]

# OCR the MRZ region of interest using Tesseract, removing any
# occurrences of spaces

mrzText = pytesseract.image_to_string(mrz)
print(mrz)
mrzText = mrzText.replace(' ', '')
print(mrzText)
mrzText = mrzText.replace('<<', '#')
print('Après replace de << par # :')
print(mrzText)
mrzText = mrzText.replace('\n', '#')
print('Après replace de \\n par # :')
print(mrzText)
x = filter(None, mrzText.split("#"))
print('Après split et filter:')
mrz_list = list(x)
print(mrz_list)
mrz_list[0] = mrz_list[0].replace('<', ' ')
mrz_list[4] = mrz_list[4].replace('<', ' ')
mrz_list[3] = mrz_list[3].replace("'", "")
mrz_list[0] = mrz_list[0].replace(" ", "")
mrz_list_0 = mrz_list[0]
mrz_list[0] = mrz_list_0[1:len(mrz_list_0) - 1]
mrz_list_1 = mrz_list[1]
print("A la fin:")
# print(mrz_list_1[:6])


mrz_list = {'Id_MRZ': mrz_list[0],
            'DateDeNaissance': mrz_list_1[:6],
            'Sexe': mrz_list_1[7],
            'DateExpitarion': mrz_list_1[8:14],
            'Pays': mrz_list_1[15:18],
            'Nom': mrz_list[3],
            'Prenom': mrz_list[4]}


print(mrz_list)


result = mrz_list.items()
data = list(result)
numpyArray = np.array(data)
print(numpyArray)
cv2.imshow("MRZ", mrz)
cv2.waitKey(0)
sys.exit(0)
