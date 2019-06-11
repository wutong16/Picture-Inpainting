from sklearn.linear_model import Lasso
from skimage.measure import compare_mse
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
import cv2
import numpy as np
import argparse
import time
import os


class PictureInpainting:
    """ This is a class to process picture inpainting
        It takes the original picture, a self defined mask
        And then it process the inpainting operations and reports the results

    Parameters
    ----------
    picture_path : path to the original picture
    mask_path    : path to the mask, it has to be black and white, the black part indicates the missing pixels
    save_dir     : path to save the results
    patch_size   : side-length for a square patch
    step         : step interval to fetch the patches
    alpha        : alpha value for Lasso
    max_iter     : max_iteration time for Lasso
    tolerance    : tolerance value for Lasso
    local        : whether to build the dictionary locally

    Attributes
    ----------
     - - -
    the results and parameters will be reported in a .txt file anyway~
    """

    def __init__(self, picture_path='./res/pictures/outdoor.jpg', mask_path = './res/pictures/outdoor_mask.jpg', save_dir = './results',
                 patch_size=33, step=8, alpha = 0.001,tolerance=0.0001, max_iter = 10000, local = True):
        self.picture_path = picture_path
        self.mask_path = mask_path
        self.save_dir = save_dir
        self.patch_size = patch_size
        self.step = step
        self.alpha = alpha
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.local = local

        self.ave_iter = 0
        self.patch_num = 0
        self.total_time = 0

        self.raw_picture, self.hole_picture, self.mask_picture, self.mask, self.missing_num, self.row, self.col = self.prepare_picture()
        self.clf = Lasso(alpha = alpha, tol=tolerance, max_iter = max_iter)

    def inpainting(self):
        start_time = time.time()

        self.dictionary = self.build_global_dictionary()
        while np.min(self.hole_picture)<-0.5:

            i, j, patch = self.highest_priority()

            fixed_num = np.sum(self.hole_picture[i:i + self.patch_size, j:j + self.patch_size] < -0.5) // 3

            recovered_patch = self.recover_patch(target_i=i,target_j=j,target_patch=patch)
            self.hole_picture[i:i + self.patch_size, j:j + self.patch_size,:] = recovered_patch.copy()

            self.patch_num += 1
            missing_num = np.sum(self.hole_picture < -0.5) // 3
            print('[{}] patches recovered!'.format(self.patch_num))
            print('{} pixels juxt filled,{} pixels left to be fill!'.format(fixed_num, missing_num))
            # cv2.imwrite('mid_'+str(self.patch_num)+'.jpg',self.hole_picture*255)

        end_time = time.time()
        self.total_time = end_time - start_time
        print('>>> Done!')

    def prepare_picture(self):
        # loading and preprocessing the picture
        try:
            picture = cv2.imread(self.picture_path)
            mask = cv2.imread(self.mask_path)
        except:
            raise FileNotFoundError('Invalid Picture or Mask Path!', self.picture_path, self.mask_path)

        if len(picture.shape) != 3:
            picture = np.dstack([picture] * 3)
        if np.max(picture) > 1:
            picture = picture / 255.0
        assert picture.shape[:1] == mask.shape[:1], print('Picture and Mask must have the SAME size!')

        # binaralize mask
        mask = cv2.threshold(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY),127,255,cv2.THRESH_BINARY)[1]/255
        mask= np.dstack([mask] * 3)

        # set the missing pixels as -1
        raw_picture = picture
        hole_picture = picture * mask + ( mask - 1 )
        mask_picture = picture * mask
        row, col, _ = picture.shape
        missing_num = np.sum(mask[:,:,0] ==0 )

        print('>>> Original picture and mask prepared!')
        return raw_picture, hole_picture, mask_picture, mask, missing_num ,row, col


    def highest_priority(self):
        # pixel Confidence, the ratio for filled pixels in a patch, so take only a layer is okay
        self.missing_pixels = np.transpose(np.where(self.hole_picture[:,:,0] < -0.5))  # (len,2) [[x1,y1],[x2,y2],...]
        binary = self.hole_picture[:,:,0] > -0.5
        expand_map = np.zeros((binary.shape[0]+self.patch_size-1,binary.shape[1]+self.patch_size-1))
        half = self.patch_size // 2
        expand_map[half:-half,half:-half] = binary
        binary = expand_map  # expand the picture for the edge points
        C = []
        for i,j in self.missing_pixels:
            i += half
            j += half
            patch = binary[i-half:i+half+1,j-half:j+half+1]
            filled_num = np.sum(patch) # all patched are the same size, so we don't divide it by patch area
            C.append(filled_num)

        inx = np.argmax(C)
        [i,j] = self.missing_pixels[inx]
        start_i = max(0, i - half)
        end_i = min(self.row, i + half + 1)
        start_j = max(0, j - half)
        end_j = min(self.col, j + half + 1)
        return i, j, self.hole_picture[start_i:end_i,start_j:end_j]

    def build_global_dictionary(self):
        # build a global dictionary from the source region of the picture
        patch_size = self.patch_size
        step = self.step
        dictionary = []
        for x in range(0, self.row-patch_size, step):
            for y in range(0, self.col-patch_size, step):
                patch = self.hole_picture[x:x+patch_size,y:y+patch_size,:]
                if np.min(patch) >=0 :
                    dictionary.append(patch)
        return np.array(dictionary)

    def build_local_dictionary(self, target_i,target_j, span_size = 200):
        # build a local dictionary from the source region near the target region to be filled
        # a smaller step is needed compared with the global dictionary
        patch_size = self.patch_size
        step = self.step
        start_i = max(0,target_i-span_size)
        end_i = min(self.row, target_i+span_size)
        start_j = max(0,target_j-span_size)
        end_j = min(self.col, target_j+span_size)

        dictionary = []
        for x in range(start_i, end_i-patch_size, step):
            for y in range(start_j, end_j-patch_size, step):
                patch = self.hole_picture[x:x+patch_size,y:y+patch_size,:]
                if np.min(patch) >= 0:
                    dictionary.append(patch)

        return np.array(dictionary)

    def recover_patch(self, target_i, target_j, target_patch):
        if self.local:
            local_dict = self.build_local_dictionary(target_i, target_j)
        else:
            local_dict = self.dictionary
        recovered_patch = np.zeros((self.patch_size,self.patch_size,3))
        iter_num = 0
        for i in range(3):
            dict_vectors = np.reshape(local_dict[:, :, :, i], (-1, len(local_dict)))
            patch_vector = np.reshape(target_patch[:, :, i], (-1, 1))

            # only calculate the locations where patch_vectors is not -1
            inx_target = np.where(patch_vector < 0)
            fake_dict_vectors = dict_vectors.copy()
            fake_dict_vectors[inx_target, :] = 0
            patch_vector[inx_target] = 0
            self.clf.fit(fake_dict_vectors, patch_vector)
            new_patch_vector = self.clf.predict(dict_vectors)
            recovered_patch[:, :, i] = np.reshape(new_patch_vector.copy(), (self.patch_size, self.patch_size))

            iter_num += self.clf.n_iter_

        self.ave_iter = (self.patch_num*self.ave_iter + iter_num/3)/(self.patch_num + 1)
        # cv2.imwrite('1.jpg',recovered_patch*255)
        inx_source = np.where(target_patch > 0)
        recovered_patch[inx_source] = target_patch[inx_source].copy()
        # cv2.imwrite('2.jpg', recovered_patch*255)
        return recovered_patch

    def measure(self):
        self.mse = compare_mse(self.raw_picture, self.hole_picture)
        self.psnr = compare_psnr(self.raw_picture, self.hole_picture)
        self.ssim = compare_ssim(self.raw_picture, self.hole_picture,multichannel=True)

    def report_results(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        img_name = self.picture_path.split('/')[-1]
        save_inpainted_path = os.path.join(self.save_dir, img_name[:-4] + '_inpaint_'+timestamp + '.jpg').replace('\\', '/')
        save_masked_path = os.path.join(self.save_dir, img_name[:-4] +'_mask_' + timestamp + '.jpg').replace('\\', '/')
        self.hole_picture = cv2.normalize(self.hole_picture*255, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        self.mask_picture = cv2.normalize(self.mask_picture * 255, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(save_inpainted_path, self.hole_picture)
        cv2.imwrite(save_masked_path, self.mask_picture)
        with open(os.path.join(self.save_dir, 'results.txt').replace('\\', '/'), 'a') as f:
            exp_setting = 'patch_size: %d | step: %d | alpha: %.6f | tolerance: %.6f | max_iter: %d | local: %d' % (
                self.patch_size, self.step, self.alpha, self.tolerance, self.max_iter, self.local)
            attributes = 'total patch num: %d | total missing pixel num: %d | average iteration: %d | total time used: %d s' % (
                self.patch_num, self.missing_num, self.ave_iter, self.total_time)
            metrics = 'MSE: %.6f | PSNR: %.6f | SSIM: %.6f ' % (self.mse, self.psnr, self.ssim)

            f.write('>>> Experiment Time: %s \n>>> Experiment Settings: \n%s \n>>>Experimental Attribute: \n%s \n>>>Experimental Metrics: \n%s' % (
            timestamp, exp_setting, attributes, metrics))
            f.write('>>> Inpainted picture save at: \n%s \n \n' % save_inpainted_path)

        print('>>> Experiment Time: %s \n>>> Experiment Settings: \n%s \n>>>Experimental Attribute: \n%s \n>>>Experimental Metrics: \n%s \n' % (
                timestamp, exp_setting, attributes, metrics))
        print('>>> Inpainted picture save at: \n%s \n \n' % save_inpainted_path)
        cv2.imshow('masked picture', self.mask_picture)
        cv2.imshow('inpainted picture', self.hole_picture)
        cv2.waitKey(0)
