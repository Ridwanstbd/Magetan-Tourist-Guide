import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import time

st.set_page_config(page_title="Magetan Tourist Guide", page_icon="🎒", layout="wide")

st.title("🎒 Magetan Smart Tourist Guide & Priority Matrix")
st.write("Aplikasi AI berbasis Machine Learning (K-Means Clustering) untuk merekomendasikan destinasi wisata di Magetan dan membantu strategi alokasi anggaran daerah.")
st.markdown("")

#  1. MEMUAT DATA (Di-cache agar aplikasi berjalan cepat) 
@st.cache_data
def load_data():
    df = pd.read_csv("kunjungan_wisata_magetan_2021_2025.csv")
    
    # Feature Engineering (Menghindari pembagian dengan nol)
    df['Growth_Rate'] = (df['Kunjungan 2025'] - df['Kunjungan 2021']) / (df['Kunjungan 2021'] + 1)
    return df

df = load_data()

#  2. PREPROCESSING & MACHINE LEARNING 
features = ['Kunjungan 2025', 'Growth_Rate', 'Rating', 'Review Count', 'Akses Mudah']
df_ml = df[features].fillna(0)

# Standarisasi Skala Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_ml)

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster_ID'] = kmeans.fit_predict(scaled_data)

# Logika Penamaan Cluster Dinamis
rata_kunjungan = df['Kunjungan 2025'].mean()
rata_growth = df['Growth_Rate'].mean()
rata_rating = df['Rating'].mean()

kamus_nama_cluster = {}
for i in range(4):
    data_kelompok = df[df['Cluster_ID'] == i]
    avg_kunjungan = data_kelompok['Kunjungan 2025'].mean()
    avg_growth = data_kelompok['Growth_Rate'].mean()
    avg_rating = data_kelompok['Rating'].mean()
    
    if avg_kunjungan >= rata_kunjungan:
        kamus_nama_cluster[i] = "Must-Visit Classics (Cash Cow)"
    elif avg_growth >= rata_growth:
        kamus_nama_cluster[i] = "Trending Spots (Rising Star)"
    elif avg_rating >= rata_rating:
        kamus_nama_cluster[i] = "Hidden Gems (Tenang & Estetik)"
    else:
        kamus_nama_cluster[i] = "Adventure & Anti-Mainstream"

df['Kategori Wisata'] = df['Cluster_ID'].map(kamus_nama_cluster)


#  3. ANTARMUKA PENGGUNA (SIDEBAR REKOMENDASI) 
st.sidebar.header("🔍 Cari Destinasi Impianmu!")
pilihan_turis = st.sidebar.radio(
    "Pengalaman seperti apa yang Anda cari?",
    (
        "Tempat ikonik, fasilitas lengkap, populer", 
        "Tempat yang sedang 'Hits' dan viral", 
        "Tempat yang tenang, view bagus (Hidden Gem)", 
        "Tempat anti-mainstream atau petualangan"
    )
)

st.sidebar.markdown("")
st.sidebar.write("Model *Machine Learning* akan mencocokkan preferensi Anda dengan data historis dan sentimen pengunjung.")

# Mapping input ke Kategori
if "ikonik" in pilihan_turis:
    target_cluster = "Must-Visit Classics (Cash Cow)"
elif "Hits" in pilihan_turis:
    target_cluster = "Trending Spots (Rising Star)"
elif "tenang" in pilihan_turis:
    target_cluster = "Hidden Gems (Tenang & Estetik)"
else:
    target_cluster = "Adventure & Anti-Mainstream"


#  4. MENAMPILKAN HASIL REKOMENDASI 
st.subheader(f"Rekomendasi Terbaik: {target_cluster}")

df_rekomendasi = df[df['Kategori Wisata'] == target_cluster].sort_values(by='Rating', ascending=False)

if df_rekomendasi.empty:
    st.info("Wah, sepertinya belum ada destinasi yang terdeteksi di kategori ini dengan data saat ini.")
else:
    # Menggunakan kolom Streamlit agar tampilannya rapi seperti kartu (cards)
    cols = st.columns(3)
    for index, row in df_rekomendasi.iterrows():
        info_akses = "✅ Akses Kendaraan Besar" if row['Akses Mudah'] == 1 else "⚠️ Akses Jalan Sempit/Menantang"
        pertumbuhan = row['Growth_Rate'] * 100
        
        # Menggilir kolom agar tampil sejajar (grid)
        with cols[index % 3]:
            st.markdown(f"#### 📍 {row['Nama Objek']}")
            st.write(f"⭐ **Rating:** {row['Rating']} ({row['Review Count']} ulasan)")
            st.write(f"📈 **Tren:** Pertumbuhan {pertumbuhan:.1f}%")
            st.caption(info_akses)
            st.markdown("")


#  5. VISUALISASI MATRIKS (Untuk Bagian Laporan/Pemerintah) 
st.markdown("<br><br>", unsafe_allow_html=True)
st.subheader("📊 Analisis Data: Magetan Tourism Priority Matrix")
st.write("Grafik di bawah ini memvisualisasikan bagaimana model algoritma memetakan setiap destinasi berdasarkan kapasitas kunjungan dan kecepatan pertumbuhannya.")

# Membuat Plot
fig, ax = plt.subplots(figsize=(12, 7))
sns.scatterplot(
    data=df, 
    x='Kunjungan 2025', 
    y='Growth_Rate',
    hue='Kategori Wisata', 
    size='Rating', 
    sizes=(100, 800), 
    palette='viridis',   
    alpha=0.8,
    ax=ax
)

# Menambah label teks di grafik
for i in range(len(df)):
    ax.text(
        df['Kunjungan 2025'][i], 
        df['Growth_Rate'][i] + 0.05, 
        df['Nama Objek'][i], 
        fontsize=8,
        ha='center'
    )

ax.set_title('Pemetaan Strategis Pariwisata Magetan (2021-2025)', fontweight='bold')
ax.set_xlabel('Kapasitas Kunjungan (2025)')
ax.set_ylabel('Laju Pertumbuhan')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Menampilkan grafik matplotlib ke dalam Streamlit
st.pyplot(fig)

# Menampilkan Raw Data yang bisa di-expand
with st.expander("Tampilkan Dataset Final"):
    st.dataframe(df[['Nama Objek', 'Kunjungan 2025', 'Growth_Rate', 'Rating', 'Kategori Wisata']])