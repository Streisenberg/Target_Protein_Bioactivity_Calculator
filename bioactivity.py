import streamlit as st
import sys
import os
from chembl_webresource_client.new_client import new_client
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import base64
import os
import json
import pickle
import uuid
import re
from functions import download_button, lipinski, norm_value, pIC50
from SessionState import SessionState
import seaborn as sns
from numpy.random import seed

st.set_option('deprecation.showPyplotGlobalUse', False)

sns.set(style='ticks')

sys.path.append('/usr/local/lib/python3.7/site-packages/')

st.set_option('deprecation.showfileUploaderEncoding', False)

session_state = SessionState.get(name="", button_sent=False)

def main():
    st.title("Target Protein Bioactivity Calculator")

    st.text("")

    arama = st.text_input("Search", "coronavirus")

    target = new_client.target
    target_query = target.search(arama) #Your target protein name that you want to search
    targets = pd.DataFrame.from_dict(target_query)
    st.write("**Chembl Data**:")
    st.write(targets)
    st.text("")

    hedef_protein = st.number_input("Enter a single protein ID that you wanna research", min_value=0 ,value=4, format="%d")
    selected_target = targets.target_chembl_id[hedef_protein]
    st.text("")
    st.write("**_ChEMBL ID_** of your protein: {0}".format(selected_target))
    activity = new_client.activity
    res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
    df = pd.DataFrame.from_dict(res)

    if hedef_protein is not None:
                

        if st.checkbox("Click to see the table where your data is filtered according to IC50"):
            if df.empty:
                st.warning("Please choose a **_Single Protein_**!")
            else:
                    
                st.write(df)
                st.text("")
            
                st.markdown(download_button(df, 'IC50_Data.csv', 'Download the CSV Data'), unsafe_allow_html=True)
        
        
        st.text("")
        st.text("")
        st.markdown("<h3 style='text-align: center; color: red;'>Seçilen Protein için molekül aktivitesini hesaplayan ML programını çalıştırmak isterseniz aşağıda ki butona tıklayınız.</h3>", unsafe_allow_html=True)
        st.text("")
        
        df2 = df[df.standard_value.notna()]
        bioactivity_class = []

        mol_cid = []
        canonical_smiles = []
        standard_value = []
        for unit in df2.standard_value:
            if float(unit) >= 10000:
                bioactivity_class.append("inactive")
            elif float(unit) <= 1000:
                bioactivity_class.append("active")
            else:
                bioactivity_class.append("intermediate")

        for i in df2.molecule_chembl_id:

            mol_cid.append(i)
        
        for i in df2.canonical_smiles:

            canonical_smiles.append(i)
            
        for i in df2.standard_value:

            standard_value.append(i)

        data_tuples = list(zip(mol_cid, canonical_smiles, standard_value, bioactivity_class))
        df3 = pd.DataFrame( data_tuples,  columns=['molecule_chembl_id', 'canonical_smiles', 'standard_value','bioactivity_class' ])
        st.text("")
        if df.empty:
            st.warning("Please choose a **_Single Protein_**!")
        else:
            if st.checkbox("Calculate Molecular Activity"):
                st.text("")
                st.text("")
                st.write(df3)
                st.text("")
                st.markdown(download_button(df3, 'General_Data.csv', 'Download the CSV Data'), unsafe_allow_html=True)

                st.text("")
                if st.selectbox("Just show the active molecules",("Active","")):
                    active_data = (df3.loc[df3['bioactivity_class'] == "active"])
                    st.write(active_data)
                    st.text("")
                    st.markdown(download_button(active_data, 'Active_Data.csv', 'Download the CSV Data'), unsafe_allow_html=True)

        
            st.text("")
            st.text("")
            st.markdown("<h3 style='text-align: center; color: red;'>Click the button below to calculate Lipinski Descriptors</h3>", unsafe_allow_html=True)
            st.text("") 
            
            button_sent = st.checkbox("Lipinski Descriptors")

            if button_sent:
                session_state.button_sent = True

                if session_state.button_sent:
                    st.subheader("Lipinski Data:")
                    st.write("**MW** = Molecular Weight")
                    st.write("**LogP** = Molekül Çözünürlüğü")
                    st.write("**NumHDonors** = Hidrojen Bağı Vericileri")
                    st.write("**NumHAcceptors** = Hidrojen Bağı Alıcıları")
                    exploratory_data = df3
                    df_lipinski = lipinski(exploratory_data.canonical_smiles)
                    #st.write(df_lipinski)
                    df_combined = pd.concat([exploratory_data,df_lipinski], axis=1)
                    st.subheader("Combined Data:")
                    st.write(df_combined)
                    st.markdown(download_button(df_combined, 'Combined_Data.csv', 'Download the CSV Data'), unsafe_allow_html=True)
                    df_norm = norm_value(df_combined)
                    #st.write(df_norm)
                    df_final = pIC50(df_norm)
                    st.subheader("The data with IC50 converted to pIC50:")
                    st.write(df_final)
                    st.markdown(download_button(df_final, 'pIC50_Data.csv', 'Download the CSV Data'), unsafe_allow_html=True)
                    df_class = df_final[df_final.bioactivity_class != "intermediate"]

                    def mannwhitney(descriptor, verbose=False):

                        # seed the random number generator
                        seed(1)

                        # actives and inactives
                        selection = [descriptor, 'bioactivity_class']
                        df = df_class[selection]
                        active = df[df.bioactivity_class == 'active']
                        active = active[descriptor]

                        selection = [descriptor, 'bioactivity_class']
                        df = df_class[selection]
                        inactive = df[df.bioactivity_class == 'inactive']
                        inactive = inactive[descriptor]

                        # compare samples
                        stat, p = mannwhitneyu(active, inactive)
                        #print('Statistics=%.3f, p=%.3f' % (stat, p))

                        # interpret
                        alpha = 0.05
                        if p > alpha:
                            interpretation = 'Same distribution (fail to reject H0)'
                        else:
                            interpretation = 'Different distribution (reject H0)'
                        
                        results = pd.DataFrame({'Descriptor':descriptor,
                                                'Statistics':stat,
                                                'p':p,
                                                'alpha':alpha,
                                                'Interpretation':interpretation}, index=[0])
                        filename = 'mannwhitneyu_' + descriptor + '.csv'
                        results.to_csv(filename)

                        return results

                    st.text("")
                    st.text("")
                    session_state.grafik = st.checkbox("Active/Inactive Molecul Graph")
                    session_state.mw = st.checkbox("Molecular Weight/Solubility Graph")
                    session_state.pic50 = st.checkbox("pIC50/Molecular Weight Graph")
                    session_state.logp = st.checkbox("Solubility/Molecular Weight Graph")
                    session_state.donors = st.checkbox("Hidrogen Bound Donor/Molecular Activity Graph")
                    session_state.acceptors = st.checkbox("Hidrogen Bound Acceptor/Molecular Weight Graph")
                    if session_state.grafik:
                        st.write("**********************************")
                        st.text("")
                        st.subheader("**Active/Inactive Molecule Graphic**")

                        plt.figure(figsize=(5.5, 5.5))

                        sns.countplot(x='bioactivity_class', data=df_class, edgecolor='black')

                        plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
                        plt.ylabel('Frequency', fontsize=14, fontweight='bold')
                        
                        st.pyplot()
                        #st.markdown(get_table_download_link(veri), unsafe_allow_html=True)
                        
                        #Buralara PDF indirici eklenecek

                    if session_state.mw:
                        st.write("**********************************")
                        st.text("")
                        st.subheader("**Molecular Weight/Solubility Graph**")

                        plt.figure(figsize=(5.5, 5.5))
                        sns.scatterplot(x='MW', y='LogP', data=df_class, hue='bioactivity_class', size='pIC50', edgecolor='black', alpha=0.7)

                        plt.xlabel('MW', fontsize=14, fontweight='bold')
                        plt.ylabel('LogP', fontsize=14, fontweight='bold')
                        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
                        st.pyplot()
                        
                        #Buralara PDF indirici eklenecek
                        st.write("**Mann-Whitney U Test Data**:")
                        st.write(mannwhitney("MW"))

                    if session_state.pic50:
                        st.write("**********************************")
                        st.text("")
                        st.subheader("**pIC50/Molecular Weight Graph**")

                        plt.figure(figsize=(5.5, 5.5))

                        sns.boxplot(x = 'bioactivity_class', y = 'pIC50', data = df_class)

                        plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
                        plt.ylabel('pIC50 value', fontsize=14, fontweight='bold')
                        st.pyplot()
                        #Buralara PDF indirici eklenecek

                        st.write("**Mann-Whitney U Test Data**:")
                        st.write(mannwhitney("pIC50"))
                    
                    if session_state.logp:
                        st.write("**********************************")
                        st.text("")
                        st.subheader("**Solubility/Molecular Weight Graph**")

                        plt.figure(figsize=(5.5, 5.5))

                        sns.boxplot(x = 'bioactivity_class', y = 'LogP', data = df_class)

                        plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
                        plt.ylabel('LogP', fontsize=14, fontweight='bold')
                        st.pyplot()
                        #Buralara PDF indirici eklenecek

                        st.write("**Mann-Whitney U Test Data**:")
                        st.write(mannwhitney("LogP"))
                    
                    if session_state.donors:
                        st.write("**********************************")
                        st.text("")
                        st.subheader("**Hidrogen Bound Donor/Molecular Activity Graph**")

                        plt.figure(figsize=(5.5, 5.5))

                        sns.boxplot(x = 'bioactivity_class', y = 'NumHDonors', data = df_class)

                        plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
                        plt.ylabel('NumHDonors', fontsize=14, fontweight='bold')
                        st.pyplot()
                        #Buralara PDF indirici eklenecek

                        st.write("**Mann-Whitney U Test Data**:")
                        st.write(mannwhitney("NumHDonors"))

                    if session_state.acceptors:
                        st.write("**********************************")
                        st.text("")
                        st.subheader("**Hidrogen Bound Acceptor/Molecular Weight Graph**")

                        plt.figure(figsize=(5.5, 5.5))

                        sns.boxplot(x = 'bioactivity_class', y = 'NumHAcceptors', data = df_class)

                        plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
                        plt.ylabel('NumHAcceptors', fontsize=14, fontweight='bold')
                        st.pyplot()
                        #Buralara PDF indirici eklenecek

                        st.write("**Mann-Whitney U Test Data**:")
                        st.write(mannwhitney("NumHAcceptors"))



if __name__ == "__main__":
    main()