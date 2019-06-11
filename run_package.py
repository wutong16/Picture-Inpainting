import cv2
import numpy as np
import dictlearn as dl

img_path = 'pictures/hill.jpg'
mask_path = 'pictures/hill_mask.jpg'

def main():
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    mask = cv2.threshold(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1] / 255
    mask = cv2.medianBlur(cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U), 3)

    channels = ['r','g','b']
    img_mask = np.zeros(image.shape)
    image_inpainted = np.zeros(image.shape)
    for i in range(3):
        image_tomask = image[:,:,i]
        corrupted = image_tomask * mask
        inp = dl.algorithms.Inpaint(corrupted, mask)
        reconstructed = inp.train().inpaint()
        img_mask[:,:,i] = corrupted
        image_inpainted[:,:,i] = reconstructed
        cv2.imwrite('results/package/'+channels[i] +'.jpg', image_tomask)
        cv2.imwrite('results/package/'+channels[i] + '_.jpg', reconstructed)
        print('Channel %s finished processing!'%channels[i])

    cv2.imwrite('results/package/img_mask.jpg',img_mask)
    cv2.imwrite('results/package/mask.jpg', mask*255)
    # cv2.imwrite('results/package/Mask_.jpg', Mask*255)
    cv2.imwrite('results/package/img_inpainted.jpg', image_inpainted)

if __name__ == "__main__":
    main()

