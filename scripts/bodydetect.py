#!/usr/bin/env python

import numpy as np
import cv2

help_message = '''
USAGE: peopledetect.py <image_names> ...

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
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


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

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width, depth = img.shape

        if height > width:
          fy = 300.0 / height
          fx = 300.0 / height
        else:
          fy = 300.0 / width
          fx = 300.0 / width

        print height, width, fx, fy
        small = cv2.resize(gray, (0,0), fx=fx, fy=fy)

        found, w = hog.detectMultiScale(small, winStride=(8,8), padding=(32,32), scale=1.01)
        found_filtered = []

        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
            else:
                found_filtered.append(r)

        draw_detections(small, found)
        draw_detections(small, found_filtered, 3)
        print '%d (%d) found' % (len(found_filtered), len(found))
        cv2.imshow('img', small)
        ch = 0xFF & cv2.waitKey()
        if ch == 27:
            break
    cv2.destroyAllWindows()
