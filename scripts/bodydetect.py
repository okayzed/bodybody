#!/usr/bin/env python

import numpy as np
import cv2
import sys

# TWEAKABLE
SHRUNKEN_SIZE=300
SCALE=1.02

help_message = '''
USAGE: bodydetect.py <image_names> ...

Press any key to continue, ESC to stop.
'''

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = (0, 0)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


def normalize_img(img):
  height, width, depth = img.shape

  if height > width:
    fy = float(SHRUNKEN_SIZE) / height
    fx = fy
  else:
    fy = float(SHRUNKEN_SIZE) / width
    fx = fy

  small = cv2.resize(img, (0,0), fx=fx, fy=fy)
#  gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
  return small


  
def detect_body_boundary(img):
  img = normalize_img(img)
  found, w = hog.detectMultiScale(img, winStride=(4,4), padding=(32,32), scale=SCALE)
  found_filtered = []

  for ri, r in enumerate(found):
      for qi, q in enumerate(found):
          if ri != qi and inside(r, q):
              break
      else:
          found_filtered.append(r)

  print len(found), len(found_filtered)
  draw_detections(img, found_filtered, 3)

  cv2.imshow('img', img)
  ch = 0xFF & cv2.waitKey()
  if ch == 27:
      sys.exit(0)
  return found_filtered


# NOT USED
def cascade_body(img):
  img = normalize_img(img)
  face_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
  faces = face_cascade.detectMultiScale(img, 1.04, 5)

  print faces

  for (x,y,w,h) in faces:
      cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

  cv2.imshow('img', img)
  ch = 0xFF & cv2.waitKey()
  if ch == 27:
      sys.exit(0)

if __name__ == '__main__':
    import sys
    from glob import glob
    import itertools as it

    print help_message

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

    for fn in it.chain(*map(glob, sys.argv[1:])):
        print fn, ' - ',
        try:
            img = cv2.imread(fn)

            if img is None:
                print 'Failed to load image file:', fn
                continue
        except:
            print 'loading error'
            continue

        detect_body_boundary(img)
    cv2.destroyAllWindows()
