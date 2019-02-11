
# coding: utf-8

# In[32]:


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
def merge_all_img(path,x):
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.001, hspace=0.001)
    row=5
    col=4
    plt.figure(figsize=(20,20))

    line=2
    for k in range(3,7):
        print(os.stat(path+'/L2Net_channelwise_max_nonloacl_'+str(k)+'_vis_point.png'))
        CMrelation = cv2.imread(path+'/L2Net_channelwise_max_nonloacl_'+str(k)+'_vis_point.png') #直接读为灰度图像
        print(path+'/L2Net_channelwise_max_nonloacl_'+str(k)+'_vis_point.png')
        # print(CMrelation)
        plt.subplot(row,col,(k-2+col*(line-1))),plt.imshow(CMrelation,interpolation='nearest'),plt.title('CM_point_'+str(k),fontsize=10)
        plt.axis('off')
    line=3
    for k in range(3,7):
        CMrelation = cv2.imread(path+'/L2Net_nonloacl_'+str(k)+'_vis_point.png') #直接读为灰度图像
        plt.subplot(row,col,(k-2+col*(line-1))),plt.imshow(CMrelation,interpolation='nearest'),plt.title('L2_point_'+str(k),fontsize=10)
        plt.axis('off')
    line=4
    for k in range(3,7):
        CMrelation = cv2.imread(path+'/L2Net_channelwise_max_nonloacl_'+str(k)+'_visrelation.png') #直接读为灰度图像
        plt.subplot(row,col,(k-2+col*(line-1))),plt.imshow(CMrelation,interpolation='nearest'),plt.title('CM_relation_'+str(k),fontsize=10)
        plt.axis('off')
    line=5
    for k in range(3,7):
        CMrelation = cv2.imread(path+'/L2Net_nonloacl_'+str(k)+'_visrelation.png') #直接读为灰度图像
        plt.subplot(row,col,(k-2+col*(line-1))),plt.imshow(CMrelation,interpolation='nearest'),plt.title('L2_relation'+str(k),fontsize=10)
        plt.axis('off')
    line=1
    original = cv2.imread(path+'/original.png') #直接读为灰度图像
    plt.subplot(row,col,(1+col*(line-1))),plt.imshow(original,'gray',interpolation='nearest'),plt.title('original',fontsize=10)
    plt.axis('off')
    lowpass = cv2.imread(path+'/Lowpass.png') #直接读为灰度图像
    plt.subplot(row,col,(2+col*(line-1))),plt.imshow(lowpass,'gray',interpolation='nearest'),plt.title('Lowpass',fontsize=10)
    plt.axis('off')
    diff_low = cv2.imread(path+'/diff_between_Lowpass.png') #直接读为灰度图像
    plt.subplot(row,col,(3+col*(line-1))),plt.imshow(diff_low,'gray',interpolation='nearest'),plt.title('diff_between_Lowpass',fontsize=10)
    plt.axis('off')
    plt.savefig(path+'/../combineans/allmerge'+str(x)+'.png')

for i in range(100):
        print os.path.join('/home/yjm/hardnet/data/featmap/non_local_fea/img_{:0>7}'.format(i))
        merge_all_img(os.path.join('/home/yjm/hardnet/data/featmap/non_local_fea/img_{:0>7}'.format(i)),i)
# for root, directories, filenames in os.walk('/home/yjm/hardnet/data/featmap/non_local_fea'):
#     for directory in directories:
#         print os.path.join(root, directory)
#         merge_all_img(os.path.join(root, directory))
#     for filename in filenames: 
#         print os.path.join(root,filename) 


