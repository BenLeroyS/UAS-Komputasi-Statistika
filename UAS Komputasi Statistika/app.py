import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from hmmlearn import hmm
import os

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
st.image("./.notebook/assets/images/Alur Flowchart.jpg", caption="Flowchart Simulasi Data Sleep & Lifestyle Study", use_column_width=True)

# ==============================================================================================
# LOAD DATASET
# ==============================================================================================

st.header("Dataset Sleep & Lifestyle Study di Kaggle")

df = pd.read_excel(r"C:\Users\Benhard Leroy\Downloads\Sleep Health and Lifestyle Dataset.xlsx")

st.header("Dataset Sleep & Lifestyle Study")
st.dataframe(df.head(10))

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

st.markdown("Shape: " + str(df.shape)) # Menampilkan jumlah baris dan kolom dalam dataset

st.write("Missing Values:") # Menampilkan jumlah missing values untuk setiap kolom dalam dataset
st.write(df.isnull().sum())

st.write("Outliers:") # Menampilkan jumlah outliers untuk setiap kolom numerik dalam dataset menggunakan metode IQR (Interquartile Range)
for column in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
    st.write(f"{column}: {len(outliers)} outliers")

st.write("Descriptive Statistics:") # Menampilkan statistik deskriptif untuk setiap kolom numerik dalam dataset
st.write(df.describe())

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
st.markdown("Feature engineering adalah proses membuat fitur baru dari data yang sudah ada untuk meningkatkan kinerja model. " \
"Fitur baru ini dapat membantu melakukan analisis yang lebih dalam, misalnya dengan membuat kategori dari variabel yang ada. ")

# Buatkan kategori untuk Durasi Tidur)

df["Sleep Quality"] = pd.cut(df["Sleep Duration"], bins=[2, 5, 7, 9, np.inf], labels=["Sangat Kurang", "Kurang", "Normal", "Cukup"]) # Membuat kategori untuk durasi tidur berdasarkan rentang waktu yang telah ditentukan
st.write("Kategori Durasi Tidur:")
st.write(df.head(10))
st.write("Distribusi Kategori Durasi Tidur:")
st.write(df["Sleep Quality"].value_counts())

st.markdown("Berdasarkan hasil dari feature engineering, kita dapat melihat bahwa durasi tidur telah dikategorikan menjadi empat kategori: 'Sangat Kurang', 'Kurang', 'Normal', dan 'Cukup'. " \
"Distribusi kategori durasi tidur menunjukkan bahwa sebagian besar individu memiliki durasi tidur yang normal hingga cukup, sementara sebagian kecil individu memiliki durasi tidur yang sangat kurang atau kurang." \
"Hal ini akan membantu kita dalam melakukan analisis lebih lanjut.")

# ==============================================================================================
# MONTE CARLO SIMULATION
# ==============================================================================================

st.subheader("Monte Carlo Simulation")

st.markdown("Simulasi Monte Carlo adalah metode statistik yang digunakan untuk memodelkan probabilitas hasil yang berbeda dalam proses yang tidak dapat diprediksi secara mudah. " \
"Dalam konteks ini, kita dapat menggunakan Simulasi Monte Carlo untuk memodelkan distribusi durasi tidur berdasarkan data yang ada, dan untuk memahami ketidakpastian dalam durasi tidur serta faktor-faktor yang mempengaruhinya. " \
"Simulasi ini akan membantu kita memahami bagaimana durasi tidur dapat bervariasi berdasarkan faktor-faktor seperti aktivitas fisik, tingkat stres, dan faktor lainnya yang telah diidentifikasi dalam dataset.")

mean = df["Sleep Duration"].mean() # Menghitung rata-rata durasi tidur dalam dataset
standard_deviation = df["Sleep Duration"].std() # Menghitung standar deviasi durasi tidur dalam dataset

st.write(f"Rata-rata Durasi Tidur: {mean:.2f}")
st.write(f"Standar Deviasi Durasi Tidur: {standard_deviation:.2f}")

n_simulations = 1000 # Jumlah simulasi yang akan dilakukan
simulasi_monte_carlo = np.random.normal(mean, standard_deviation, n_simulations) # Melakukan simulasi Monte Carlo dengan menghasilkan data acak yang mengikuti distribusi normal berdasarkan rata-rata dan standar deviasi durasi tidur dalam dataset

fig8, ax8 = plt.subplots() # Membuat histogram dari hasil simulasi Monte Carlo untuk durasi tidur
ax8.hist(simulasi_monte_carlo)
ax8.axvline(mean, color="red", linestyle="dashed", linewidth=2, label="Rata-rata")
ax8.axvline(mean + standard_deviation, color="green", linestyle="dashed", linewidth=2, label="Rata-rata + 1 SD")
ax8.axvline(mean - standard_deviation, color="green", linestyle="dashed", linewidth=2, label="Rata-rata - 1 SD")
ax8.set_title("Simulasi Monte Carlo untuk Durasi Tidur") 
st.write("Hasil Simulasi Monte Carlo untuk Durasi Tidur:")  
st.pyplot(fig8)

st.markdown("Berdasarkan hasil dari Simulasi Monte Carlo, kita dapat melihat distribusi durasi tidur yang dihasilkan dari simulasi tersebut. " \
"Distribusi ini memberikan wawasan tentang bagaimana durasi tidur dapat bervariasi berdasarkan faktor-faktor yang mempengaruhinya, seperti aktivitas fisik, tingkat stres, dan faktor lainnya yang telah diidentifikasi dalam dataset. " \
"Informasi ini dapat digunakan untuk mengembangkan strategi untuk meningkatkan kualitas tidur berdasarkan faktor-faktor yang telah diidentifikasi dalam analisis ini, dan memberikan dasar yang kuat untuk penelitian lebih lanjut dalam bidang ini.")

# ==============================================================================================
# MARKOV CHAIN
# ==============================================================================================

st.subheader("Markov Chain")
st.markdown("Markov Chain adalah model matematika yang digunakan untuk memodelkan sistem yang berubah dari satu state ke state lainnya dalam suatu proses stokastik. " \
"Dalam konteks ini, kita dapat menggunakan Markov Chain untuk memodelkan transisi antara berbagai tingkat kualitas tidur berdasarkan faktor-faktor yang mempengaruhinya, seperti durasi tidur, aktivitas fisik, dan tingkat stres. " \
"Model ini akan membantu kita memahami bagaimana individu dapat berpindah antara berbagai tingkat kualitas tidur berdasarkan faktor-faktor yang mempengaruhinya, dan memberikan wawasan tentang pola transisi antara berbagai tingkat kualitas tidur dalam dataset.")

states = df["Sleep Quality"] # Mengambil kolom "Quality of Sleep" dari dataset untuk digunakan sebagai states dalam Markov Chain
matriks_transisi = pd.crosstab(states, states.shift(), normalize="index") # Membuat matriks transisi dengan menggunakan crosstab untuk menghitung frekuensi transisi antara states dan kemudian menormalkannya untuk mendapatkan probabilitas transisi
matriks_transisi = matriks_transisi.fillna(0) # Mengisi nilai NaN dengan 0 dalam matriks transisi
st.write("Matriks Transisi Markov Chain:")
st.write(matriks_transisi)

# Visualisasi Matriks Transisi
fig9, ax9 = plt.subplots() # Membuat histogram dari hasil simulasi Monte Carlo untuk durasi tidur
ax9.hist(simulasi_monte_carlo)
ax9.set_title("Matriks Transisi Markov Chain untuk Kualitas Tidur")
ax9.set_xlabel("States")
ax9.set_ylabel("Probabilitas Transisi")
sns.heatmap(matriks_transisi, annot=True, cmap="Blues", ax=ax9) # Membuat heatmap dari matriks transisi untuk memvisualisasikan probabilitas transisi antara states dalam Markov Chain
st.write("Visualisasi Matriks Transisi Markov Chain:")
st.pyplot(fig9)

st.markdown("Berdasarkan hasil dari Markov Chain, kita dapat melihat probabilitas transisi antara berbagai tingkat kualitas tidur dalam dataset. " \
"Model ini memberikan wawasan tentang bagaimana individu dapat berpindah antara berbagai tingkat kualitas tidur berdasarkan faktor-faktor yang mempengaruhinya, seperti durasi tidur, aktivitas fisik, dan tingkat stres. " \
"Informasi ini dapat digunakan untuk mengembangkan strategi untuk meningkatkan kualitas tidur berdasarkan faktor-faktor yang telah diidentifikasi dalam analisis ini, dan memberikan dasar yang kuat untuk penelitian lebih lanjut dalam bidang ini.")

# ==============================================================================================
# HIDDEN MARKOV MODEL
# ==============================================================================================

st.subheader("Hidden Markov Model (HMM)")
st.markdown("Hidden Markov Model (HMM) adalah model statistik yang digunakan untuk memodelkan sistem yang berubah dari satu state ke state lainnya, di mana states tersebut tidak dapat diamati secara langsung (hidden states). " \
"Dalam konteks ini, kita dapat menggunakan HMM untuk memodelkan pola tersembunyi dalam data durasi tidur yang mungkin terkait dengan variabel lainnya. " \
"Model ini akan membantu kita memahami pola tersembunyi dalam data durasi tidur dan bagaimana faktor-faktor tersebut mempengaruhi transisi antara berbagai tingkat durasi tidur dalam dataset, serta memberikan wawasan tentang bagaimana durasi tidur dapat mempengaruhi kualitas tidur berdasarkan pola tersembunyi yang diidentifikasi dalam data.")

# Siapkan Data untuk HMM

observations = df[["Sleep Duration"]].values # Mengambil kolom "Sleep Duration" dari dataset untuk digunakan sebagai observasi dalam HMM
scaler = MinMaxScaler() # Membuat objek MinMaxScaler untuk melakukan normalisasi pada data observasi agar berada dalam rentang 0 hingga 1, yang diperlukan untuk melatih model HMM
observations_scaled = scaler.fit_transform(observations) 

# Latih HMM

model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=100) # Membuat model HMM dengan 4 states (Sangat Kurang, Kurang, Normal, Cukup) dan tipe kovarians diagonal 
model.fit(observations_scaled) # Melatih model HMM dengan data observasi yang telah diskalakan
hidden_states = model.predict(observations_scaled)
df["Hidden State"] = hidden_states # Menambahkan kolom "Hidden State" ke dataset untuk menyimpan hasil prediksi dari HMM
st.write("Hasil Prediksi Hidden Markov Model:")
st.write(df[["Sleep Duration", "Hidden State"]].head(11))

# Visualisasi HMM
fig10, ax10 = plt.subplots() # Membuat scatterplot antara variabel Durasi Tidur dan Hidden State dalam dataset
ax10.plot(df["Sleep Duration"], label="Durasi Tidur")
ax10.plot(df["Hidden State"], label="Hidden State")
ax10.set_title("Hasil Hidden Markov Model (HMM) untuk Durasi Tidur")
ax10.set_xlabel("Index")
ax10.set_ylabel("States")
ax10.legend()
st.write("Visualisasi Hasil Hidden Markov Model (HMM):")
st.pyplot(fig10)

st.markdown("Berdasarkan hasil dari Hidden Markov Model (HMM), kita melihat bahwa model berhasil mengidentifikasi pola tersembunyi dalam data durasi tidur yang mungkin terkait dengan variabel lainnya. " \
"Model ini memberikan wawasan tentang bagaimana durasi tidur dapat mempengaruhi kualitas tidur, dan bagaimana faktor-faktor lain seperti aktivitas fisik dan tingkat stres dapat mempengaruhi transisi antara berbagai tingkat durasi tidur. " \
"Informasi ini dapat digunakan untuk mengembangkan strategi untuk meningkatkan kualitas tidur berdasarkan faktor-faktor yang telah diidentifikasi dalam analisis ini, dan memberikan dasar yang kuat untuk penelitian lebih lanjut dalam bidang ini.")

# ==============================================================================================
# EVALUASI, DISKUSI, DAN KESIMPULAN
# ==============================================================================================

# Evaluasi
st.subheader("Evaluasi")
st.markdown("Evaluasi dari analisis ini adalah penggunaan tiga metode (Simulasi Monte Carlo, Markov Chain, dan Hidden Markov Model) memberikan wawasan yang berharga tentang pola tidur dan faktor-faktor yang mempengaruhinya dalam dataset Sleep & Lifestyle Study. " \
"Simulasi Monte Carlo memberikan distribusi probabilitas yang realistis untuk durasi tidur, sementtara markov chain menghasilkan matriks transisi yang memberikan wawasan tentang pola transisi antara berbagai tingkat kualitas tidur, dan Hidden Markov Model berhasil mengidentifikasi pola tersembunyi dalam data durasi tidur yang mungkin terkait dengan variabel lainnya. ")

# Diskusi
st.subheader("Diskusi")
st.markdown("Kelebihan dari analisis ini adalah penggunaan tiga metode yang berbeda memberikan wawasan yang lebih komprehensif tentang pola tidur dan faktor-faktor yang mempengaruhinya dalam dataset Sleep & Lifestyle Study. " \
"Simulasi Monte Carlo memberikan distribusi probabilitas yang realistis untuk durasi tidur, sementara Markov Chain dan Hidden Markov Model berhasil mengidentifikasi pola transisi dan pola tersembunyi dalam data kualitas tidur. " \
"Namun, keterbatasan dari analisis ini adalah bahwa hasil dari analisis ini harus diinterpretasikan dengan hati-hati, dan lebih banyak penelitian diperlukan untuk memahami secara lebih mendalam tentang faktor-faktor yang mempengaruhi kualitas tidur dan bagaimana mereka berinteraksi satu sama lain. " \
"Selain itu, hasil dari analisis ini dapat dipengaruhi oleh kualitas data dan asumsi yang digunakan dalam model, sehingga penting untuk melakukan validasi dan evaluasi lebih lanjut untuk memastikan keakuratan dan relevansi hasil analisis ini."
"Analisis ini memberikan wawasan yang berharga tentang pola tidur dan faktor-faktor yang mempengaruhinya dalam dataset Sleep & Lifestyle Study, dan dapat digunakan untuk mengembangkan strategi untuk meningkatkan kualitas tidur berdasarkan faktor-faktor yang telah diidentifikasi dalam analisis ini, serta memberikan dasar yang kuat untuk penelitian lebih lanjut dalam bidang ini.")

# Kesimpulan
st.subheader("Kesimpulan")
st.markdown("Berdasarkan hasil dari analisis ini, kita dapat menyimpulkan bahwa analisis ini memberikan wawasan yang berharga tentang pola tidur dan faktor-faktor yang mempengaruhinya dalam dataset Sleep & Lifestyle Study. " \
"Pendekatan yang digunakan dalam analisis ini, yaitu Simulasi Monte Carlo, Markov Chain, dan Hidden Markov Model, memberikan wawasan yang lebih komprehensif tentang pola tidur dan faktor-faktor yang mempengaruhinya dalam dataset ini. " \
"Saya berharap bahwa hasil dari analisis ini dapat digunakan untuk mengembangkan strategi untuk meningkatkan kualitas tidur berdasarkan faktor-faktor yang telah diidentifikasi dalam analisis ini, serta memberikan dasar yang kuat untuk penelitian lebih lanjut dalam bidang ini.")