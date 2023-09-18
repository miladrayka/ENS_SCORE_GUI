
import os
import shutil
import subprocess
import streamlit as st

import utils

st.set_page_config(
    page_title="ENS-Score GUI",
    page_icon="./title_logo.png",
    layout="centered",
    initial_sidebar_state="collapsed",
)

#Title

st.image("./software_logo.png")
st.title("Binding Affinity Prediction")
st.write("**ENS-Score GUI** is software for predicting binding affinity of protein-ligand complex.")

#Sidebar

st.sidebar.header("About")
st.sidebar.image("./github_logo.png", width=30)
st.sidebar.write("[GitHub](https://github.com/miladrayka), Developed by *[Milad Rayka](https://scholar.google.com/citations?user=NxF2f0cAAAAJ&hl=en)*.")
st.sidebar.divider()
st.sidebar.write("""**ENS-Score GUI** is software for predicting binding affinity of protein-ligand complex. It is developed at 
*Baqiyatallah University of Medical Sciences*.""")
st.sidebar.divider()
st.sidebar.write("""**Reference**:\n
Paper is *under production.*""")

#Body

with st.expander("See Information"):
    st.write("""**ENS-Score** is a machine learning-based scoring function for prediction of binding affinity of a protein-ligand 
                complex. It uses ensemble technique and conformal prediction method to report confidence alongside binding affinity. 
                For more information refer to our published paper.""")


st.warning('**Caution**: Both ligand and protein should have hydrogen atoms.')
st.warning('**Caution**: Change the names of your ligand and protein files to *name_ligand.mol2*, *name_ligand.sdf*, and *name_protein.pdb*.')

#Ligand and Protein Structures Selection
st.divider()
st.subheader("Ligand and Protein Structures Selection")

#CSS code to modify the st.file_uploader graphic
css='''
<style>
[data-testid="stFileUploadDropzone"] div div::before {content:"Drag and drop file here"}
[data-testid="stFileUploadDropzone"] div div span{display:none;}
[data-testid="stFileUploadDropzone"] div div small{display:none;}
[data-testid="stFileUploadDropzone"] svg{display:none;}
</style>
'''
utils.delete_files_and_subdirectories("./Example/")
utils.delete_files_and_subdirectories("./Generated_features/")

name = st.text_input("Provide the name of your protein-ligand complex", placeholder="e.g. 1a1e")
subprocess.run(["wsl", "echo", name, ">", "./Example/list_pdbid.txt"])

st.markdown(css, unsafe_allow_html=True)

mol2_ligand = st.file_uploader("Select a ligand **MOL2** file.", type="mol2")
sdf_ligand = st.file_uploader("Select a ligand **SDF** file.", type="sdf")
pdb_protein = st.file_uploader("Select a protein **PDB** file.", type="pdb")

try:

    with open(f"./Example/{mol2_ligand.name}", "wb") as file :
        file.write(mol2_ligand.getvalue())

    with open(f"./Example/{sdf_ligand.name}", "wb") as file :
        file.write(sdf_ligand.getvalue())
    
    with open(f"./Example/{pdb_protein.name}", "wb") as file :
        file.write(pdb_protein.getvalue())
        
except:
    pass


#Scoring Section
st.divider()
st.subheader("Scoring Section")

start = st.button("Predict Binding Affinity")
if start:
    with st.spinner("Please Wait..."):
        os.system("python generation_and_prediction.py")
        
    with open("./Example/result.txt", "r") as file:
        result = file.readlines()

    st.success(f"Mean of Predicted Binding Affinity: {float(result[0]):.3f}")
    st.success(f"Confidence Region: {float(result[1]):.3f}")
