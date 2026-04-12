import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from hmmlearn import hmm

# ==================================================================================================
# SETUP WARNA BACKGROUND DAN WARNA FONT
# ==================================================================================================

st.markdown("""
            <style>
            .stApp {
                background-color: white;
                color: black;
            }
            </style>
            """, unsafe_allow_html=True)
# ==================================================================================================
# TITLE
# ==================================================================================================

st.title("Simulasi Monte Carlo , Markov Chain, dan Hidden Markov Model pada Sleep & Lifestyle Study")

# ==============================================================================================
# FLOWCHART
# ==============================================================================================

st.header("Flowchart Simulasi Data")
st.image("alur_flowchart_simulasi_data.jpg", caption="Flowchart Simulasi Data Sleep & Lifestyle Study", use_column_width=True)

# ==============================================================================================
# LOAD DATASET
# ==============================================================================================

df = pd.read_excel("sleep_health_and_lifestyle_dataset.xlsx") # Memuat dataset dari file Excel ke dalam DataFrame menggunakan pandas

st.header("Dataset Sleep & Lifestyle Study")
st.dataframe(df.head(11).style.set_properties(**{"background-color": "white", "color": "black"})) # Menampilkan 10 baris pertama dari dataset dengan latar belakang biru muda dan teks hitam menggunakan Streamlit

st.caption("Sumber: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset")
st.markdown("Dataset ini berisi informasi tentang kebiasaan tidur, gaya hidup, dan faktor kesehatan dari berbagai individu. " \
"Data ini dapat digunakan untuk menganalisis hubungan antara pola tidur dan faktor-faktor lain seperti aktivitas fisik, kebiasaan tidur, dan tingkat stres." \
"Dataset ini berisi 400 baris dan 13 kolom (variabel), dengan variabel-variabel tersebut adalah ID, jenis kelamin, usia, durasi tidur, kualitas tidur, aktivitas fisik, tingkat stres, kategori BMI, tekanan darah, detak jantung, langkah harian, dan gangguan tidur." \
"Dataset ini sangat berguna untuk melakukan analisis lebih lanjut, seperti simulasi Monte Carlo, Markov Chain, dan Hidden Markov Model, untuk memahami pola tidur dan faktor-faktor yang mempengaruhinya.")

# ==============================================================================================
# EKSPLORASI DAN VISUALISASI DATA
# ==============================================================================================

st.subheader("Eksplorasi dan Visualisasi Data")
st.markdown("Sebelum melakukan analisis lebih lanjut, penting untuk melakukan eksplorasi dan visualisasi data untuk memahami struktur data, distribusi variabel, missing values, dan statistik deskriptif. " \
"Hal ini akan membantu kita memahami karakteristik data dan mengidentifikasi pola-pola yang mungkin ada dalam dataset, yang akan digunakan dalam simulasi Monte Carlo, Markov Chain, dan Hidden Markov Model untuk menganalisis hubungan antara faktor-faktor tersebut dengan kualitas tidur.")

EDA = '''
# Lakukan Cleaning Data dan EDA 
st.write("Shape: " + str(df.shape)) 

st.write("Missing Values:") 
st.write(df.isnull().sum().to_frame().style.set_properties(**{"background-color": "white", "color": "black"})

st.write("Outliers:") 
for column in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
    st.write(f"{column}: {len(outliers)} outliers")

st.write("Descriptive Statistics:") 
st.write(df.describe().to_frame().style.set_properties(**{"background-color": "white", "color": "black"}))

# Visualisasi Data

fig1, ax1 = plt.subplots() 
df["Sleep Duration"].hist(ax=ax1)
st.write("Distribusi Durasi Tidur:")
st.pyplot(fig1)

fig2, ax2 = plt.subplots() 
df["Quality of Sleep"].hist(ax=ax2)
st.write("Distribusi Kualitas Tidur:")
st.pyplot(fig2)

fig3, ax3 = plt.subplots() 
df["Physical Activity Level"].hist(ax=ax3)
st.write("Distribusi Tingkat Aktivitas Fisik:")
st.pyplot(fig3)

fig4, ax4 = plt.subplots() 
df["Stress Level"].hist(ax=ax4)
st.write("Distribusi Tingkat Stres:")
st.pyplot(fig4)

fig5, ax5 = plt.subplots() 
df["Heart Rate"].hist(ax=ax5)
st.write("Distribusi Detak Jantung:")
st.pyplot(fig5)

fig6, ax6 = plt.subplots() 
df["Daily Steps"].hist(ax=ax6)
st.write("Distribusi Langkah Harian:")
st.pyplot(fig6)
'''
# Tampilkan kode
st.code(EDA, language="python")

# Lakukan Cleaning Data dan EDA
st.write("Shape: " + str(df.shape)) # Menampilkan jumlah baris dan kolom dalam dataset

st.write("Missing Values:") # Menampilkan jumlah missing values untuk setiap kolom dalam dataset
st.write(df.isnull().sum().to_frame().style.set_properties(**{"background-color": "white", "color": "black"}))

st.write("Outliers:") # Menampilkan jumlah outliers untuk setiap kolom numerik dalam dataset menggunakan metode IQR (Interquartile Range)
for column in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
    st.write(f"{column}: {len(outliers)} outliers")

st.write("Descriptive Statistics:") # Menampilkan statistik deskriptif untuk setiap kolom numerik dalam dataset
st.write(df.describe().to_frame().style.set_properties(**{"background-color": "white", "color": "black"}))

# Visualisasi Data

fig1, ax1 = plt.subplots() # Membuat histogram untuk variabel Durasi Tidur dalam dataset
df["Sleep Duration"].hist(ax=ax1)
st.write("Distribusi Durasi Tidur:")
st.pyplot(fig1)

fig2, ax2 = plt.subplots() # Membuat histogram untuk variabel Kualitas Tidur dalam dataset
df["Quality of Sleep"].hist(ax=ax2)
st.write("Distribusi Kualitas Tidur:")
st.pyplot(fig2)

fig3, ax3 = plt.subplots() # Membuat histogram untuk variabel Tingkat Aktivitas Fisik dalam dataset
df["Physical Activity Level"].hist(ax=ax3)
st.write("Distribusi Tingkat Aktivitas Fisik:")
st.pyplot(fig3)

fig4, ax4 = plt.subplots() # Membuat histogram untuk variabel Tingkat Stres dalam dataset
df["Stress Level"].hist(ax=ax4)
st.write("Distribusi Tingkat Stres:")
st.pyplot(fig4)

fig5, ax5 = plt.subplots() # Membuat histogram untuk variabel Detak Jantung dalam dataset
df["Heart Rate"].hist(ax=ax5)
st.write("Distribusi Detak Jantung:")
st.pyplot(fig5)

fig6, ax6 = plt.subplots() # Membuat histogram untuk variabel Langkah Harian dalam dataset
df["Daily Steps"].hist(ax=ax6)
st.write("Distribusi Langkah Harian:")
st.pyplot(fig6)

fig7, ax7 = plt.subplots() # Membuat scatterplot antara variabel Durasi Tidur dan Kualitas Tidur dalam dataset
sns.scatterplot(x="Sleep Duration", y="Quality of Sleep", data=df, ax=ax7)
sns.regplot(x="Sleep Duration", y="Quality of Sleep", data=df, scatter=False, ax=ax7, color="red")
st.write("Scatterplot antara Durasi Tidur dan Kualitas Tidur:")
st.pyplot(fig7)

st.markdown("Visualisasi pada data di atas menunjukkan distribusi durasi tidur dan kualitas tidur dalam dataset. " \
"Distribusi durasi tidur menunjukkan bahwa sebagian besar individu memiliki durasi tidur antara 6 hingga 8 jam, sementara distribusi kualitas tidur menunjukkan bahwa sebagian besar individu memiliki kualitas tidur yang normal hingga baik. " \
"Informasi ini penting untuk memahami pola tidur dan gaya hidup individu dalam studi ini, yang akan digunakan dalam simulasi Monte Carlo, Markov Chain, dan Hidden Markov Model untuk menganalisis hubungan antara faktor-faktor tersebut dengan kualitas tidur.")

# ==============================================================================================
# FEATURE ENGINEERING
# ==============================================================================================

st.subheader("Feature Engineering")
st.markdown("Feature engineering adalah proses mengubah data mentah menjadi fitur (variabel input) yang lebih relevan dan informatif bagi model machine learning. " \
"Tujuannya adalah meningkatkan performa model, akurasi prediksi, dan mempermudah algoritma dalam mempelajari pola data. " \
"Teknik ini krusial dalam siklus hidup proyek data science. " \
"Dalam konteks ini, saya melakukan feature engineering dengan membuat kategori untuk durasi tidur berdasarkan rentang waktu yang telah ditentukan." \
"Feature ini akan membantu kita memahami bagaimana durasi tidur dapat mempengaruhi kualitas tidur dan memberikan dasar yang kuat untuk analisis lebih lanjut (Markov Chain untuk state diskrit dan HMM untuk model probabilistik).")

FE = '''
# Buat Kategori Tidur dari Variabel Durasi Tidur

df["Sleep Category"] = pd.cut(df["Sleep Duration"], bins=[0, 4, 6, 8, 10, np.inf], labels=["Sangat Kurang", "Kurang", "Normal", "Cukup", "Kesiangan"])  
st.dataframe(df[["Sleep Duration", "Sleep Category"]].head(11).to_frame().style.set_properties(**{"background-color": "white", "color": "black"})) 
'''
# Buat Kategori Tidur dari Variabel Durasi Tidur

df["Sleep Category"] = pd.cut(df["Sleep Duration"], bins=[0, 4, 6, 8, 10, np.inf], labels=["Sangat Kurang", "Kurang", "Normal", "Cukup", "Kesiangan"]) # Membuat kategori tidur berdasarkan durasi tidur 
st.dataframe(df[["Sleep Duration", "Sleep Category"]].head(11).to_frame().style.set_properties(**{"background-color": "white", "color": "black"})) # Menampilkan 10 baris pertama dari dataframe yang berisi kolom "Sleep Duration" dan "Sleep Category"

# Tampilkan kode
st.code(FE, language="python")

st.markdown("Kategori durasi tidur yang telah dibuat berdasarkan durasi tidur memberikan wawasan tentang bagaimana durasi tidur dapat mempengaruhi kualitas tidur."
"Feature engineering ini membantu kita memahami bagaimana individu dapat dikategorikan berdasarkan durasi tidur mereka, dan memberikan dasar yang kuat untuk analisis lebih lanjut (Markov Chain untuk state diskrit dan HMM untuk model probabilistik).")

# ==============================================================================================
# MONTE CARLO SIMULATION
# ==============================================================================================

st.subheader("Monte Carlo Simulation")

st.markdown("Simulasi Monte Carlo adalah teknik matematika dan komputasi yang digunakan untuk memprediksi berbagai hasil dari suatu peristiwa yang tidak pasti dengan menggunakan pengambilan sampel acak berulang. " \
"Metode ini memodelkan risiko dan ketidakpastian dalam situasi kompleks, untuk memberikan berbagai kemungkinan hasil beserta probabilitasnya, membantu pengambilan keputusan yang lebih baik. " \
"Dalam konteks ini, saya menggunakan Simulasi Monte Carlo untuk memodelkan distribusi durasi tidur, tingkat aktivitas fisik, dan tingkat stres dalam dataset Sleep & Lifestyle Study. " \
"Simulasi ini akan membantu kita memahami distribusi probabilitas dari durasi tidur, tingkat aktivitas fisik, dan tingkat stres dalam dataset, serta memberikan wawasan tentang bagaimana faktor-faktor tersebut dapat mempengaruhi kualitas tidur berdasarkan distribusi probabilitas yang dihasilkan dari simulasi ini.")

Monte_Carlo_Simulation = '''
# Simulasi Monte Carlo untuk Durasi Tidur, Tingkat Aktivitas Fisik, dan Tingkat Stres
simulasi_durasi_tidur = np.random.normal(loc=df["Sleep Duration"].mean(), scale=df["Sleep Duration"].std(), size=1000) 
simulasi_aktivitas_fisik = np.random.normal(loc=df["Physical Activity Level"].mean(), scale=df["Physical Activity Level"].std(), size=1000) 
simulasi_tingkat_stres = np.random.normal(loc=df["Stress Level"].mean(), scale=df["Stress Level"].std(), size=1000) 

fig8, ax8 = plt.subplots(1, 3, figsize=(15, 5)) 
ax8[0].hist(simulasi_durasi_tidur, bins=30)
ax8[0].set_title("Simulasi Monte Carlo untuk Durasi Tidur")
ax8[0].set_xlabel("Durasi Tidur (jam)")
ax8[0].set_ylabel("Frekuensi")
ax8[1].hist(simulasi_aktivitas_fisik, bins=30)
ax8[1].set_title("Simulasi Monte Carlo untuk Tingkat Aktivitas Fisik")
ax8[1].set_xlabel("Tingkat Aktivitas Fisik")    
ax8[1].set_ylabel("Frekuensi")
ax8[2].hist(simulasi_tingkat_stres, bins=30)
ax8[2].set_title("Simulasi Monte Carlo untuk Tingkat Stres")
ax8[2].set_xlabel("Tingkat Stres")
ax8[2].set_ylabel("Frekuensi")
st.write("Visualisasi Simulasi Monte Carlo untuk Durasi Tidur, Tingkat Aktivitas Fisik, dan Tingkat Stres:")
st.pyplot(fig8)
'''
# Tampilkan kode
st.code(Monte_Carlo_Simulation, language="python")

# Simulasi Monte Carlo untuk Durasi Tidur, Tingkat Aktivitas Fisik, dan Tingkat Stres
simulasi_durasi_tidur = np.random.normal(loc=df["Sleep Duration"].mean(), scale=df["Sleep Duration"].std(), size=1000) # Melakukan simulasi Monte Carlo untuk durasi tidur
simulasi_aktivitas_fisik = np.random.normal(loc=df["Physical Activity Level"].mean(), scale=df["Physical Activity Level"].std(), size=1000) # Melakukan simulasi Monte Carlo untuk tingkat aktivitas fisik
simulasi_tingkat_stres = np.random.normal(loc=df["Stress Level"].mean(), scale=df["Stress Level"].std(), size=1000) # Melakukan simulasi Monte Carlo untuk tingkat stres

fig8, ax8 = plt.subplots(1, 3, figsize=(15, 5)) # Membuat histogram dari hasil simulasi Monte Carlo untuk durasi tidur, tingkat aktivitas fisik, dan tingkat stres
ax8[0].hist(simulasi_durasi_tidur, bins=30)
ax8[0].set_title("Simulasi Monte Carlo untuk Durasi Tidur")
ax8[0].set_xlabel("Durasi Tidur (jam)")
ax8[0].set_ylabel("Frekuensi")
ax8[1].hist(simulasi_aktivitas_fisik, bins=30)
ax8[1].set_title("Simulasi Monte Carlo untuk Tingkat Aktivitas Fisik")
ax8[1].set_xlabel("Tingkat Aktivitas Fisik")    
ax8[1].set_ylabel("Frekuensi")
ax8[2].hist(simulasi_tingkat_stres, bins=30)
ax8[2].set_title("Simulasi Monte Carlo untuk Tingkat Stres")
ax8[2].set_xlabel("Tingkat Stres")
ax8[2].set_ylabel("Frekuensi")
st.write("Visualisasi Simulasi Monte Carlo untuk Durasi Tidur, Tingkat Aktivitas Fisik, dan Tingkat Stres:")
st.pyplot(fig8)

st.markdown("Berdasarkan hasil dari Simulasi Monte Carlo, kita dapat melihat distribusi probabilitas dari durasi tidur, tingkat aktivitas fisik, dan tingkat stres dalam dataset Sleep & Lifestyle Study. " \
"Hasil ini menunjukkan bahwa ketiga variabel tersebut memiliki distribusi kurva normal, menyatakan bahwa  " \
"Simulasi ini memberikan wawasan tentang bagaimana faktor-faktor tersebut dapat mempengaruhi kualitas tidur berdasarkan distribusi probabilitas yang dihasilkan dari simulasi ini. " \
"Informasi ini bisa digunakan dalam pengembangan strategi untuk meningkatkan kualitas tidur berdasarkan faktor-faktor yang terjadi dalam dataset ini, serta memberikan dasar yang kuat untuk penelitian lebih lanjut dalam bidang ini.")

# ==============================================================================================
# MARKOV CHAIN
# ==============================================================================================

st.subheader("Markov Chain")
st.markdown("Markov Chain atau (Rantai Markov) adalah model matematika yang digunakan untuk memodelkan sistem yang berubah dari satu state ke state lainnya, di mana probabilitas transisi hanya bergantung pada state saat ini. "
"Metode ini menggunakan matriks probabilitas transisi untuk menganalisis perubahan sistem seiring waktu, seperti loyalitas merek atau prediksi pasar, hingga mencapai kondisi stabil. "
"Dalam konteks ini, saya menggunakan Markov Chain untuk memodelkan transisi antara berbagai tingkat kategori tidur dalam dataset Sleep & Lifestyle Study. " \
"Model ini akan membantu kita memahami pola transisi antara berbagai tingkat kualitas tidur dalam dataset ini.")

Markov_Chain = '''
# Siapkan Data untuk Markov Chain
states = df["Sleep Category"].unique() 
transition_matrix = np.zeros((len(states), len(states))) 
for i in range(len(df) - 1):
    current_state = df["Sleep Category"].iloc[i] 
    next_state = df["Sleep Category"].iloc[i + 1] 
    transition_matrix[states == current_state, states == next_state] += 1 
transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True) 

# Visualisasi Data Markov Chain
fig9, ax9 = plt.subplots()
sns.heatmap(transition_matrix, annot=True, cmap="Blues", ax=ax9) 
st.write("Visualisasi Matriks Transisi Markov Chain untuk Kualitas Tidur:")
st.pyplot(fig9)
'''
# Tampilkan kode
st.code(Markov_Chain, language="python")

# Siapkan Data untuk Markov Chain
states = df["Sleep Category"].unique() # Mendapatkan kategori tidur yang unik dari kolom "Sleep Category"
transition_matrix = np.zeros((len(states), len(states))) # Membuat matriks transisi dengan ukuran sesuai dengan jumlah kategori tidur
for i in range(len(df) - 1):
    current_state = df["Sleep Category"].iloc[i] # Mendapatkan kategori tidur saat ini
    next_state = df["Sleep Category"].iloc[i + 1] # Mendapatkan kategori tidur berikutnya
    transition_matrix[states == current_state, states == next_state] += 1 # Menghitung jumlah transisi dari kategori tidur saat ini ke kategori tidur berikutnya
transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True) # Normalisasi matriks transisi untuk mendapatkan probabilitas transisi antara kategori tidur

# Visualisasi Data Markov Chain
fig9, ax9 = plt.subplots()
sns.heatmap(transition_matrix, annot=True, cmap="Blues", ax=ax9) # Membuat heatmap untuk matriks transisi menggunakan seaborn
st.write("Visualisasi Matriks Transisi Markov Chain untuk Kualitas Tidur:")
st.pyplot(fig9)

st.markdown("Berdasarkan hasil dari Markov Chain, kita dapat melihat pola transisi antara berbagai tingkat kategori tidur dalam dataset Sleep & Lifestyle Study. " \
"Model ini memberikan wawasan tentang bagaimana individu dapat berpindah antara berbagai tingkat kualitas tidur, serta memberikan informasi tentang probabilitas transisi antara kategori tidur yang berbeda. " \
"Informasi ini sangat membantu untuk mengetahui pola transisi antara berbagai tingkat kualitas tidur dalam dataset ini, serta memberikan dasar yang kuat untuk penelitian lebih lanjut dalam bidang ini, terutama dalam mengembangkan strategi untuk meningkatkan kualitas tidur.")

# ==============================================================================================
# HIDDEN MARKOV MODEL
# ==============================================================================================

st.subheader("Hidden Markov Model (HMM)")
st.markdown("Hidden Markov Model (HMM) adalah model statistik yang digunakan untuk memodelkan data sekuensial atau deret waktu, di mana sistem diasumsikan sebagai proses Markov dengan state (keadaan) yang tidak dapat diamati secara langsung (hidden). " \
"HMM memprediksi state tersembunyi berdasarkan rangkaian observasi yang terlihat. " \
"Dalam konteks ini, saya menggunakan HMM unutk memodelkan pola tersembunyi dalam kualitas tidur dalam dataset Sleep & Lifestyle Study. " \
"Model ini akan membantu kita memahami pola tersembunyi dalam data durasi tidur yang mungkin terkait dengan variabel lainnya, selain dari variabel yang sudah diamati dalam dataset ini (durasi tidur, aktivitas fisik, dan tingkat stres), serta memberikan wawasan tentang bagaimana durasi tidur dapat mempengaruhi kualitas tidur, dan berbagai faktor-faktor lain yang bisa mempengaruhi kualitas tidurnya" \
"berdasarkan pola tersembunyi yang diidentifikasi dalam data durasi tidur.")

HMM = '''
# Siapkan Data untuk HMM
X = df[["Sleep Duration", "Physical Activity Level", "Stress Level"]].values 
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100) 
model.fit(X) # Melatih model HMM dengan data yang telah disiapkan

hidden_states = model.predict(X) 
df["Hidden State"] = hidden_states 
st.write(df[["Sleep Duration", "Physical Activity Level", "Stress Level", "Hidden State"]].head(11).to_frame().style.set_properties(**{"background-color": "white", "color": "black"}))) 

# Visualisasi Data HMM
fig10, ax10 = plt.subplots() 
ax10.plot(df["Sleep Duration"], label="Durasi Tidur") 
ax10.plot(df["Physical Activity Level"], label="Tingkat Aktivitas Fisik") 
ax10.plot(df["Stress Level"], label="Tingkat Stres") # Membuat plot untuk tingkat stres
ax10.set_title("Pola Tersembunyi dalam Data Durasi Tidur berdasarkan Hidden Markov Model (HMM)")
ax10.set_xlabel("Index")  
ax10.set_ylabel("Nilai")
ax10.legend()
st.write("Visualisasi Pola Tersembunyi dalam Data Durasi Tidur berdasarkan Hidden Markov Model (HMM):")
st.pyplot(fig10)  
'''
# Tampilkan kode
st.code(HMM, language="python")

# Siapkan Data untuk HMM
X = df[["Sleep Duration", "Physical Activity Level", "Stress Level"]].values # Ambil kolom "Sleep Duration", "Physical Activity Level", dan "Stress Level" sebagai fitur untuk HMM
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100) # Membuat model HMM dengan 3 hidden states, menggunakan distribusi Gaussian, dan iterasi maksimum sebanyak 100
model.fit(X) # Melatih model HMM dengan data yang telah disiapkan

hidden_states = model.predict(X) # Memprediksi hidden states untuk setiap data point dalam dataset menggunakan model HMM
df["Hidden State"] = hidden_states # Menambahkan kolom "Hidden State" ke dalam dataframenya
st.write(df[["Sleep Duration", "Physical Activity Level", "Stress Level", "Hidden State"]].head(11).to_frame().style.set_properties(**{"background-color": "white", "color": "black"})) # Menampilkan 10 baris pertama dari dataframe yang berisi kolom "Sleep Duration", "Physical Activity Level", "Stress Level", dan "Hidden State"

# Visualisasi Data HMM
fig10, ax10 = plt.subplots() 
ax10.plot(df["Sleep Duration"], label="Durasi Tidur") # Membuat plot untuk durasi tidur
ax10.plot(df["Physical Activity Level"], label="Tingkat Aktivitas Fisik") # Membuat plot untuk tingkat aktivitas fisik
ax10.plot(df["Stress Level"], label="Tingkat Stres") # Membuat plot untuk tingkat stres
ax10.set_title("Pola Tersembunyi dalam Data Durasi Tidur berdasarkan Hidden Markov Model (HMM)")
ax10.set_xlabel("Index")  
ax10.set_ylabel("Nilai")
ax10.legend()
st.write("Visualisasi Pola Tersembunyi dalam Data Durasi Tidur berdasarkan Hidden Markov Model (HMM):")
st.pyplot(fig10) 

st.markdown("Berdasarkan hasil dari Hidden Markov Model (HMM), kita dapat melihat pola tersembunyi dalam data durasi tidur yang mungkin terkait dengan variabel lainnya, selain dari variabel yang sudah diamati dalam dataset ini (durasi tidur, aktivitas fisik, dan tingkat stres), serta memberikan wawasan tentang bagaimana durasi tidur dapat mempengaruhi kualitas tidur, dan berbagai faktor-faktor lain yang bisa mempengaruhi kualitas tidurnya berdasarkan pola tersembunyi yang diidentifikasi dalam data durasi tidur. " \
"Informasi ini sangat berguna untuk mengembangkan strategi untuk meningkatkan kualitas tidur berdasarkan faktor-faktor yang terjadi dalam analisis ini, serta memberikan dasar yang kuat untuk penelitian lebih lanjut dalam bidang ini.")

# ==============================================================================================
# EVALUASI, DISKUSI, DAN KESIMPULAN
# ==============================================================================================

# Evaluasi
st.subheader("Evaluasi")
st.markdown("Evaluasi dari analisis ini menunjukkan bahwa metode yang digunakan (Simulasi Monte Carlo, Markov Chain, dan Hidden Markov Model) memberikan wawasan yang berharga tentang pola tidur dan faktor-faktor yang mempengaruhinya dalam dataset Sleep & Lifestyle Study." \
"Simulasi Monte Carlo memberikan distribusi probabilitas yang realistis untuk durasi tidur, sementara Markov Chain dan Hidden Markov Model berhasil mengidentifikasi pola transisi dan pola tersembunyi dalam data kualitas tidur.")

# Diskusi
st.subheader("Diskusi")
st.markdown("Kelebihan dari analisis ini adalah penggunaan metode ini memberikan wawasan yang lebih komprehensif tentang pola tidur dan faktor-faktor yang mempengaruhinya dalam dataset Sleep & Lifestyle Study. " \
"Lalu, hasil dari analisis ini dapat digunakan untuk mengembangkan strategi untuk meningkatkan kualitas tidur berdasarkan faktor-faktor yang telah diidentifikasi dalam analisis ini. " \
"Namun, keterbatasannya adalah hasil dari analisis ini masih bersifat eksploratif dan tidak dapat digunakan untuk membuat kesimpulan yang pasti tentang hubungan antara faktor-faktor tersebut dengan kualitas tidur. " \
"Selain itu, dataset ini memiliki keterbatasan dalam hal ukuran sampel dan representasi, sehingga hasil dari analisis ini mungkin tidak dapat digeneralisasi ke populasi yang lebih luas. " \
"Jadi, walaupun analisis ini memberikan wawasan yang berharga dan mengetahui faktor-faktor yang paling umum terjadi, perlu ada penelitian lebih lanjut dengan dataset yang lebih besar dan representatif di seluruh dunia untuk mendapatkan pemahaman yang lebih mendalam tentang hubungan antara faktor-faktor tersebut dengan kualitas tidur, " \
"serta untuk mengembangkan strategi yang lebih efektif untuk meningkatkan kualitas tidur berdasarkan faktor-faktor yang telah diidentifikasi dalam analisis ini.")

# Kesimpulan
st.subheader("Kesimpulan")
st.markdown("Kesimpulan yang saya bisa dapatkan dari analisis ini adalah ketiga metode tersebut dapat berikan wawasan yang berharga tentang pengaruh kualitas tidur dan faktor-faktor yang mempengaruhinya dalam dataset ini. " \
"Saya berharap bahwa analisis ini dapat memberikan dasar yang kuat untuk penelitian lebih lanjut dalam bidang ini, dan dapat digunakan untuk mengembangkan strategi untuk meningkatkan kualitas tidur berdasarkan faktor-faktor yang telah diidentifikasi dalam analisis ini.")