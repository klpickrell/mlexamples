import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('im1.jpg',0)
v0 = 0.0
m0 = None
n0 = 0
v1 = 0.0
m1 = None
n1 = 0
usto = np.zeros(img.shape)
ret,img_u = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
        
for i, row in enumerate(img):
    print( 'row {}'.format(i))
    for j, val in enumerate(row):
#        if j == 8:
#            import pdb; pdb.set_trace()
        cma0 = m0
        if cma0 is None:
            cma0 = float(val)
            cn0 = 1
        else:
            cma0 = (cma0*n0 + val)/(n0+1.0)

        hv0 = (n0*v0 + (val-cma0)**2)/(n0+1.0)

        cma1 = m1
        if cma1 is None:
            cma1 = float(val)
            cn1 = 1
        else:
            cma1 = (cma1*n1 + val)/(n1+1)

        hv1 = (n1*v1 + (val-cma1)**2)/(n1+1)

        value = None
        if v0 == 0.0:
            # add to v0
            n0 += 1
            v0 = hv0
            m0 = cma0
            value = 255
        elif v1 == 0.0:
            n1 += 1
            v1 = hv1
            m1 = cma1
            value = 0
#        elif hv0/v1 > hv1/v0:
        elif np.abs(hv0-v1) < np.abs(hv1-v0):
            # add to v0
            n0 += 1
            v0 = hv0
            m0 = cma0
            value = 255
        else:
            # add to v1
            n1 += 1
            v1 = hv1
            m1 = cma1
            value = 0

        usto[i][j] = value


plt.imshow(usto,cmap=plt.cm.gray)
plt.show()
import pdb; pdb.set_trace()

blur = cv2.GaussianBlur(img,(5,5),0)
ret,thresh1 = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
print( 'otsu threshold {}'.format(ret) )
plt.imshow(thresh1,cmap=plt.cm.gray)
plt.show()

hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'b')
plt.xlim([0,256])
plt.legend(('histogram'), loc = 'upper left')
plt.show()

hist = cv2.calcHist([img],[0],None,[256],[0,256])
hist_norm = hist.ravel()/hist.max()
Q = hist_norm.cumsum()
bins = np.arange(256)
fn_min = np.inf
#fn_min = 0
thresh = -1
for i in xrange(1,256):
    p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
    q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
    b1,b2 = np.hsplit(bins,[i]) # weights
    # finding means and variances
    m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
    # calculates the minimization function
    fn = v1*q1 + v2*q2
    if fn < fn_min:
        fn_min = fn
        thresh = i

print('calculated threshold {}'.format(thresh))

plt.imshow(thresh1,cmap=plt.cm.gray)
plt.axis('off')
plt.show()

#ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
#ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
#ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
#ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_OTSU)
##ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
#
#titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','OTSU']
#images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
#
#for i in xrange(6):
#    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#    plt.title(titles[i])
#    plt.xticks([]),plt.yticks([])
#
#plt.show()

#hist,bins = np.histogram(img.flatten(),256,[0,256])
#
#cdf = hist.cumsum()
#cdf_normalized = cdf * hist.max()/ cdf.max()
#plt.plot(cdf_normalized, color = 'b')
#plt.hist(img.flatten(),256,[0,256], color = 'b')
#plt.xlim([0,256])
#plt.legend(('histogram'), loc = 'upper left')
#plt.show()
#
#cdf_m = np.ma.masked_equal(cdf,0)
#cdf_m = (cdf_m-cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
#cdf = np.ma.filled(cdf_m,0).astype('uint8')
#img2 = cdf[img]
##plt.imshow(img)
#plt.imshow(img2)
#plt.show()
