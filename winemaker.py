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

def s_branch(payout_matrix,prob_M):
    return(prob_M * payout_matrix[0,1] + (1-prob_M) * payout_matrix[0,2])
    
def ns_branch(payout_matrix,prob_high_sugar,prob_low_sugar,prob_regular_sugar):
    return(prob_high_sugar * payout_matrix[0,3] + prob_low_sugar*payout_matrix[0,4] + prob_regular_sugar*payout_matrix[0,5])    
    
#true positive rate = sensitivity
#true negative rate = specificity
def get_Expected_new(prob_storm, model_sen, model_spe, payout_matrix, prob_M,prob_high_sugar,prob_low_sugar,prob_regular_sugar):
    P_DS = model_sen * (prob_storm) + (1-model_sen) * (1-prob_storm)
    #P_DNS =  1 - P_DS
    P_DNS = model_spe * (1-prob_storm) + (1-model_spe) * prob_storm
    
    P_S_DS = (model_sen * prob_storm)/P_DS
    P_NS_DS = 1 - P_S_DS

    E_val_top = []
    E_val_top.append(payout_matrix[0,0])
    E_val_top.append(s_branch(payout_matrix,prob_M) * P_S_DS + P_NS_DS*ns_branch(payout_matrix,prob_high_sugar,prob_low_sugar,prob_regular_sugar))

    P_NS_DNS = (model_spe * (1-prob_storm))/P_DNS
    P_S_DNS = 1-P_NS_DNS

    E_val_bot = []
    E_val_bot.append(payout_matrix[0,0])
    E_val_bot.append(s_branch(payout_matrix,prob_M) * P_S_DNS + P_NS_DNS*ns_branch(payout_matrix,prob_high_sugar,prob_low_sugar,prob_regular_sugar))

    result = np.max(E_val_top) * P_DS + np.max(E_val_bot)* (1-P_DS)
    
    #Recommend actions
    if E_val_top[1] > E_val_top[0]:
        st.write('When detector predicts storm, do not harvest')
    else:
        st.write('When detector predicts storm, do harvest')
    if E_val_bot[1] > E_val_top[0]:
        st.write('When detector predicts no storm, do not harvest')
    else:
        st.write('When detector predicts no storm, do harvest')
    
    return (result-payout_matrix[0,0])


def make_plot(prob_storm, payout_matrix, prob_M,prob_high_sugar,prob_low_sugar,prob_regular_sugar):
    sen_array = np.arange(0,1,0.01)
    expected_val_list = []
    for i in sen_array:
        expected_val_list.append(get_Expected_new(float(prob_storm), i, i,payout_matrix, 
                                                  float(prob_M), float(prob_high_sugar), float(prob_low_sugar), float(prob_regular_sugar)))

    plt.plot(sen_array, expected_val_list)
    plt.title('Sensitivity vs. Expected Value')
    plt.xlabel('Sensitivity')
    plt.ylabel('Expected Value')
    
    st.pyplot(plt)


if __name__ == '__main__':
    prob_storm = 0.5
    model_sen = 0.138
    model_spe = 0.364
    prob_M = 0.1
    prob_high_sugar = 0.1
    prob_low_sugar = 0.6
    prob_regular_sugar = 0.3
    payout_matrix = initialize() #initialize the matrix
    
    st.title('Expected Value for Winemaker')
    st.write("Probability of storm = ",prob_storm)
    st.write("Probability of model sensitivity = ", model_sen)
    st.write("Probability of model specitivity = ", model_spe)
    st.write("Probability of chance of botrytis = ",prob_M)
    st.write("Probability of high sugar = ",prob_high_sugar)
    st.write("Probability of low sugar = ",prob_low_sugar)
    st.write("Probability of regular sugar = ",prob_regular_sugar)
    st.write("Here is the expected value from the default input")
    st.write(get_Expected_new(float(prob_storm), float(model_sen), float(model_spe), payout_matrix, float(prob_M), float(prob_high_sugar), float(prob_low_sugar), float(prob_regular_sugar)))
    
    # streamlit
    prob_high_sugar = st.number_input('Probability of High Sugar')
    prob_low_sugar = st.number_input('Probability of Low Sugar')
    prob_regular_sugar = st.number_input('Probability of Regular Sugar')
    prob_M = st.number_input('Chance of botrytis')
    
    if(prob_high_sugar + prob_low_sugar + prob_regular_sugar) != 1:
        st.write("Invalid Input")
    else:
        st.write("Below are the recommendations:")
        st.write("Here is the expected value based on your input :", get_Expected_new(float(prob_storm), float(model_sen), float(model_spe), payout_matrix, float(prob_M), float(prob_high_sugar), float(prob_low_sugar), float(prob_regular_sugar)))
       #make_plot(float(prob_storm), payout_matrix, float(prob_M), float(prob_high_sugar), float(prob_low_sugar), float(prob_regular_sugar))
    
    