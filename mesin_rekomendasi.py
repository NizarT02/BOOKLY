"""Mesin rekomendasi buku (content-based filtering: TF-IDF + cosine similarity).

Logikanya sama dengan notebook Colab, tetapi model dibangun ulang dari file CSV
setiap aplikasi dijalankan (hanya beberapa detik). Dengan begitu aplikasi tidak
bergantung pada versi library yang dipakai saat membuat file .joblib.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOPWORD = {'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'ini', 'itu', 'atau', 'oleh',
            'the', 'of', 'and', 'a', 'an', 'in', 'to', 'is', 'or', 'by', 'on', 'at', 'for', 'be', 'with'}

KOLOM_HASIL = ['judul', 'author', 'kategori', 'original_rating', 'rating_count', 'kemiripan']


def cari_csv(folder):
    """Cari file CSV Gramedia di folder yang sama dengan aplikasi."""
    kandidat = sorted(Path(folder).glob('gramedia_goodreads*.csv'))
    return kandidat[0] if kandidat else None


def token_penulis(teks):
    """'Dini W.Tamam;Erby S' -> 'diniwtamam erbys' (tiap penulis jadi satu token)."""
    return ' '.join(re.sub(r'[^a-z0-9]', '', p.lower()) for p in str(teks).split(';') if p.strip())


def bersihkan_judul(teks):
    teks = re.sub(r'[^a-z0-9\s]', ' ', str(teks).lower())
    return ' '.join(t for t in teks.split() if t not in STOPWORD)


def _buat_fitur(baris):
    kat = '' if baris['kategori'] == 'tidak_diketahui' else baris['kategori'].replace(',', ' ')
    pen = token_penulis(baris['author'])
    return ' '.join([kat] * 3 + [pen] * 2 + [bersihkan_judul(baris['judul'])])


class MesinRekomendasi:
    def __init__(self, path_csv):
        df = pd.read_csv(path_csv, sep='|')

        # Hapus duplikat (judul sama + penulis sama); judul sama dengan penulis berbeda dipertahankan
        df['kunci'] = df['judul'].str.lower().str.strip() + '|' + df['author'].apply(token_penulis)
        df = (df.sort_values('rating_count', ascending=False)
                .drop_duplicates('kunci')
                .drop(columns='kunci')
                .reset_index(drop=True))

        df['kategori'] = df['kategori'].fillna('tidak_diketahui').str.lower().str.strip()
        df['jumlah_halaman'] = pd.to_numeric(df['jumlah_halaman'], errors='coerce')

        # Weighted rating (rata-rata Bayesian), lalu diskalakan ke 0-1
        C = df['original_rating'].mean()
        m = df['rating_count'].quantile(0.60)
        v, R = df['rating_count'], df['original_rating']
        df['weighted_rating'] = (v / (v + m)) * R + (m / (v + m)) * C
        wr = df['weighted_rating'].values
        self.wr_norm = (wr - wr.min()) / (wr.max() - wr.min())

        df['fitur'] = df.apply(_buat_fitur, axis=1)
        self.df = df
        self.tfidf = TfidfVectorizer()
        self.matriks = self.tfidf.fit_transform(df['fitur'])

    # ------------------------------------------------------------ daftar pilihan untuk antarmuka
    def daftar_genre(self):
        """Mengembalikan list (token, jumlah_buku), terbanyak lebih dulu."""
        hitung = (self.df['kategori'].str.split(',').explode().value_counts())
        return [(g, int(j)) for g, j in hitung.items() if g != 'tidak_diketahui']

    def daftar_penulis(self):
        """Nama penulis unik (satu nama per orang), urut abjad."""
        unik = {}
        for teks in self.df['author']:
            for nama in str(teks).split(';'):
                nama = nama.strip()
                kunci = re.sub(r'[^a-z0-9]', '', nama.lower())
                if kunci and any(c.isalpha() for c in nama) and kunci not in unik:
                    unik[kunci] = nama
        return sorted(unik.values(), key=str.casefold)

    def daftar_judul(self):
        """List (indeks, label). Label memuat penulis karena ada judul kembar."""
        return [(i, f"{r.judul} oleh {r.author.split(';')[0].strip()}")
                for i, r in self.df[['judul', 'author']].iterrows()]

    # ------------------------------------------------------------ rekomendasi
    def _urutkan(self, sim, n, alpha, kecuali=None):
        skor = (1 - alpha) * sim + alpha * self.wr_norm
        if kecuali is not None:
            skor[kecuali] = -1                      # jangan rekomendasikan buku itu sendiri
        top = np.argsort(skor)[::-1][:n]
        hasil = self.df.iloc[top].copy()
        hasil['kemiripan'] = sim[top]
        return hasil[KOLOM_HASIL].reset_index(drop=True)

    def dari_minat(self, kategori=None, penulis=None, kata_kunci='', n=10, alpha=0.15):
        """Mengembalikan DataFrame kosong bila tidak ada masukan yang dikenali model."""
        bagian = []
        if kategori:
            bagian += [' '.join(kategori)] * 3
        if penulis:
            bagian += [token_penulis(penulis)] * 2
        if kata_kunci:
            bagian.append(bersihkan_judul(kata_kunci))
        vektor = self.tfidf.transform([' '.join(bagian)])
        if vektor.nnz == 0:
            return pd.DataFrame(columns=KOLOM_HASIL)
        sim = cosine_similarity(vektor, self.matriks).ravel()
        return self._urutkan(sim, n, alpha)

    def dari_judul(self, idx, n=10, alpha=0.15):
        sim = cosine_similarity(self.matriks[idx], self.matriks).ravel()
        return self._urutkan(sim, n, alpha, kecuali=idx)
