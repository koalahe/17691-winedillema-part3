#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:28:59 2022

@author: conniehe
"""

import numpy as np
import sys
import matplotlib.pyplot as plt 
import streamlit as st


def initialize():
    payout_matrix = np.matrix([6000*5+2000*10+2000*15,
                                5000*5+1000*10+2000*120,
                                5000*5+1000*10,
                                4000*5+2500*10+2000*15+1000*30+500*40,
                                6000*5+2000*10+2000*15,
                                5000*5+1000*10+2500*15+1500*30])
    return payout_matrix * 12

def get_Expected_new(prob_storm, model_sen, payout_matrix, prob_M,prob_high_sugar,prob_low_sugar,prob_regular_sugar):
    P_DNS = model_sen * (prob_storm) + (1-model_sen) * (1-prob_storm)
    P_DS =  1 - P_DNS
    
    P_M_DS = prob_M * (model_sen * (prob_storm))/(P_DS)
    P_NM_DS = (1-prob_M) * (model_sen * (prob_storm))/(P_DS) # add up to 0.2

    E_val_top = []
    E_val_top.append(payout_matrix[0,0])
    E_val_top.append(payout_matrix[0,1] * P_M_DS)
    E_val_top.append(payout_matrix[0,2] * P_NM_DS)

    P_HS_DNS = (prob_high_sugar * model_sen * (1-prob_storm))/(P_DNS)
    P_LS_DNS = (prob_low_sugar * model_sen * (1-prob_storm))/(P_DNS)
    P_RS_DNS = (prob_regular_sugar * model_sen * (1-prob_storm))/(P_DNS) # don't add to 1. add to 0.8

    E_val_bot = []
    E_val_bot.append(P_HS_DNS * payout_matrix[0,3])
    E_val_bot.append(P_LS_DNS * payout_matrix[0,4])
    E_val_bot.append(P_RS_DNS * payout_matrix[0,5])

    result = np.max(E_val_top) * P_DS + np.max(E_val_bot)* (1-P_DS)
    
    cert_equ = []
    cert_equ.append(payout_matrix[0,0])
    cert_equ.append((prob_storm) * payout_matrix[0,1] + (1-prob_storm) * payout_matrix[0,3])
    
    # selection from prev. expected
    result = result - np.max(cert_equ)
    
    return result

def make_plot(prob_storm, model_sen, payout_matrix, prob_M,prob_high_sugar,prob_low_sugar,prob_regular_sugar):
    sen_array = np.arange(0,1,0.01)
    expected_val_list = []
    for i in sen_array:
        #expected_val_list.append(get_Expected_new(prob_storm, i, payout_matrix,0.1,0.1,0.6,0.3))
        expected_val_list.append(get_Expected_new(float(prob_storm), i, payout_matrix, 
                                                  float(prob_M), float(prob_high_sugar), float(prob_low_sugar), float(prob_regular_sugar)))

    plt.plot(sen_array, expected_val_list)
    plt.title('Sensitivity vs. Expected Value')
    plt.xlabel('Sensitivity')
    plt.ylabel('Expected Value')
    
    st.pyplot(plt)


if __name__ == '__main__':
    prob_storm = sys.argv[1]
    model_sen = sys.argv[2]
    prob_M = sys.argv[3]
    prob_high_sugar = sys.argv[4]
    prob_low_sugar = sys.argv[5]
    prob_regular_sugar = sys.argv[6]
    payout_matrix = initialize() #initialize the matrix
    
    st.title('Expected Value for Winemaker')
    st.write("Probability of storm = ",prob_storm)
    st.write("Probability of model sensitivity = ", model_sen)
    st.write("Probability of chance of botrytis = ",prob_M)
    st.write("Probability of high sugar = ",prob_high_sugar)
    st.write("Probability of low sugar = ",prob_low_sugar)
    st.write("Probability of regular sugar = ",prob_regular_sugar)
    st.write("Here is the expected value from the default input")
    st.write(get_Expected_new(float(prob_storm), float(model_sen), payout_matrix, float(prob_M), float(prob_high_sugar), float(prob_low_sugar), float(prob_regular_sugar)))
    
    # streamlit
    prob_high_sugar = st.number_input('Probability of High Sugar')
    prob_low_sugar = st.number_input('Probability of Low Sugar')
    prob_regular_sugar = st.number_input('Probability of Regular Sugar')
    prob_M = st.number_input('Chance of botrytis')

    if(prob_high_sugar + prob_low_sugar + prob_regular_sugar) != 1:
        st.write("Invalid Input")
    else:
        make_plot(float(prob_storm), float(model_sen), payout_matrix, float(prob_M), float(prob_high_sugar), float(prob_low_sugar), float(prob_regular_sugar))
    