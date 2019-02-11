#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/26 20:04
# @Author  : yjm
# @Site    : 
# @File    : get_network_model.py
# @Software: PyCharm
import networks
import networks_CDbin_structure_v2 as networks_CDbin_structure


import sys
def get_network_model(name,outnum=128,pretrained=False):
    print("################# network choosing ################################")
    if(name=='L2NET'):
        model = networks.HardNet()
    elif (name == 'CDbin_NET_deep5_1'):
        print("model is CDbin_NET_deep5_1_out"+str(outnum))
        model = networks_CDbin_structure.CDbin_NET_deep5_1(outnum)
    elif (name == 'CDbin_NET_deep4_1'):
        print("model is CDbin_NET_deep4_1_out"+str(outnum))
        model = networks_CDbin_structure.CDbin_NET_deep4_1(outnum)
    elif (name == 'CDbin_NET_deep3'):
        print("model is CDbin_NET_deep3_out"+str(outnum))
        model = networks_CDbin_structure.CDbin_NET_deep3(outnum)
    elif (name == 'CDbin_NET_deep3_2'):
        print("model is CDbin_NET_deep3_2_out"+str(outnum))
        model = networks_CDbin_structure.CDbin_NET_deep3_2(outnum)
    elif (name == 'CDbin_NET_deep2'):
        print("model is CDbin_NET_deep2_out"+str(outnum))
        model = networks_CDbin_structure.CDbin_NET_deep2(outnum)
    else:
        # print ('Unknown batch reduce mode. Try L2NET, L2Net_channelwise_max_iSORT, L2Net_channelwise_max, A_channelwise_max, Ax_channelwise_max, G, H, I, L2Net_channelwise_max_nonloacl_6, L2Net_nonloacl_6')
        print("this network is not defined:",name)
        sys.exit(1)
    print("\n################# network choosing ################################")
    return model