
import difflib
import html
import json
import re
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st

from mesin_rekomendasi import MesinRekomendasi, cari_csv


# ============================================================
# BOOKLY — Smart Book Recommendation System
# Mesin rekomendasi: TF-IDF + cosine similarity.
# ============================================================


# ----------------------------- PENGATURAN APLIKASI
st.set_page_config(
    page_title="BOOKLY | Smart Book Recommendation",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------- MATERI 1: DESIGN SYSTEM / CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700;800&display=swap');

:root {
  --ink:#18283A; --muted:#718096; --paper:#F5F2EA; --white:#FFFFFF;
  --navy:#14283F; --navy2:#203B58; --gold:#D6AD62; --line:#E8E3D9;
  --mint:#DDEBE4; --rose:#F3E2DC;
}
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--paper); color:var(--ink); }

header[data-testid="stHeader"] {
  background:var(--paper) !important;
  height:3rem;
}
header[data-testid="stHeader"] * { color:#24384E !important; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { background:transparent !important; }
[data-testid="stDecoration"] { display:none; }
.block-container { max-width:1450px; padding-top:4.2rem !important; padding-bottom:3rem; }

[data-testid="stSidebar"] { background:#14283F; border-right:1px solid #29435E; }
[data-testid="stSidebar"] * { color:#F5F2EA !important; }
[data-testid="stSidebar"] .stRadio label { padding:8px 10px; border-radius:10px; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#D4DCE5; }
[data-testid="stSidebar"] div.stButton > button,
[data-testid="stSidebar"] div.stButton > button * { color:#24384E !important; }
[data-testid="stSidebar"] div.stButton > button:hover,
[data-testid="stSidebar"] div.stButton > button:hover * { color:#8B672F !important; }

h1,h2,h3 { color:var(--ink); }
h1,.serif { font-family:'Playfair Display',serif !important; }

.eyebrow {
  text-transform:uppercase; letter-spacing:2.2px; font-size:10px;
  font-weight:700; color:#B88A43;
}
.hero {
  background:linear-gradient(115deg,#14283F 0%,#203B58 64%,#31536A 100%);
  border-radius:24px; padding:34px 38px; color:#F8F5ED; position:relative;
  overflow:hidden; box-shadow:0 18px 38px rgba(20,40,63,.16); margin-bottom:22px;
}
.hero:after {
  content:""; position:absolute; width:290px; height:290px; right:-75px; top:-115px;
  border:1px solid rgba(214,173,98,.35); border-radius:50%;
  box-shadow:0 0 0 25px rgba(214,173,98,.05),0 0 0 55px rgba(214,173,98,.04);
}
.hero h1 {
  color:#FFF9EE; font-size:clamp(30px,4vw,48px);
  margin:7px 0 8px; line-height:1.12;
}
.hero p {
  color:#D9E0E7; max-width:620px; font-size:14px;
  line-height:1.7; margin:0;
}
.hero .eyebrow { color:#E3BD78; }

.metric {
  background:#fff; border:1px solid var(--line); border-radius:17px;
  padding:17px 18px; box-shadow:0 7px 18px rgba(34,48,62,.035);
  min-height:108px;
}
.metric .label {
  color:var(--muted); font-size:11px; font-weight:700;
  letter-spacing:.7px; text-transform:uppercase;
}
.metric .value {
  font-family:'Playfair Display',serif; font-size:27px;
  font-weight:700; color:var(--ink); margin-top:5px;
}
.metric .hint { color:#9AA4AE; font-size:11px; margin-top:2px; }

.section-head {
  display:flex; align-items:end; justify-content:space-between;
  gap:12px; margin:25px 0 13px;
}
.section-head h2 {
  font-family:'Playfair Display',serif; font-size:24px; margin:0;
}
.section-head p { color:var(--muted); font-size:12px; margin:3px 0 0; }

.book-card {
  background:#fff; border:1px solid var(--line); border-radius:17px;
  padding:13px; box-shadow:0 8px 20px rgba(29,43,58,.045);
  height:100%; transition:all .2s ease;
}
.book-card:hover {
  transform:translateY(-4px);
  box-shadow:0 16px 28px rgba(29,43,58,.11);
}
.book-cover {
  height:185px; border-radius:11px; padding:15px 13px; color:#fff;
  display:flex; flex-direction:column; justify-content:space-between;
  position:relative; overflow:hidden;
  box-shadow:8px 10px 15px rgba(22,32,45,.18), inset 5px 0 7px rgba(0,0,0,.13);
  transform:perspective(700px) rotateY(-5deg); transform-origin:left center;
}
.book-cover:after {
  content:""; position:absolute; top:0; left:8px; width:1px;
  height:100%; background:rgba(255,255,255,.25);
}
.book-cover.has-img {
  background-size:cover !important;
  background-position:center !important;
  background-repeat:no-repeat !important;
}
.cover-top { font-size:8px; letter-spacing:2px; text-transform:uppercase; opacity:.8; }
.cover-title {
  font-family:'Playfair Display',serif; font-size:18px;
  font-weight:700; line-height:1.16; z-index:1; overflow-wrap:anywhere;
}
.cover-author { font-size:9px; opacity:.85; z-index:1; }
.book-title {
  font-family:'Playfair Display',serif; font-weight:700; color:var(--ink);
  font-size:15px; line-height:1.3; margin:12px 0 4px; min-height:39px;
}
.book-author { color:#778392; font-size:11px; min-height:18px; }
.book-meta {
  display:flex; justify-content:space-between; align-items:center;
  margin-top:9px; font-size:11px; color:#667486;
}
.book-price { margin-top:9px; font-weight:700; font-size:14px; color:#14283F; }
.book-price small { font-weight:500; color:#8A95A2; font-size:10px; }
.pill {
  display:inline-block; border-radius:30px; padding:4px 9px;
  background:#F1E8D7; color:#87652F; font-size:10px; font-weight:700;
}
.rank {
  display:inline-flex; align-items:center; justify-content:center;
  width:29px; height:29px; background:#F3E9D7; color:#946C2E;
  border-radius:9px; font-weight:800; font-size:12px;
}
.info-panel {
  background:#fff; border:1px solid var(--line);
  border-radius:17px; padding:18px 20px;
}
.info-panel h3 {
  font-family:'Playfair Display',serif; margin:0 0 7px; font-size:18px;
}
.info-panel p { color:#738092; font-size:12px; line-height:1.65; margin:0; }
.detail-box { font-size:12px; line-height:1.7; color:#3A4A5C; }
.detail-box .lbl {
  font-size:10px; font-weight:700; letter-spacing:1px;
  text-transform:uppercase; color:#B88A43; margin-top:8px;
}
.detail-box a {
  color:#203B58; font-weight:700; text-decoration:none;
  border-bottom:1px solid #D6AD62;
}
div[data-testid="stExpander"] {
  border:1px solid var(--line); border-radius:12px; background:#fff;
}
div[data-testid="stExpander"] summary {
  font-size:12px; font-weight:700; color:#24384E;
}
div.stButton > button {
  border-radius:10px; border:1px solid #D9D2C5; background:#fff;
  color:#24384E; font-weight:700; font-size:12px;
  min-height:37px; transition:all .15s ease;
}
div.stButton > button:hover {
  border-color:#C49A55; color:#8B672F; background:#FBF8F1;
}
div.stButton > button[kind="primary"] {
  background:#203B58; color:white; border:0;
}
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div { border-radius:11px; }
hr { border-color:var(--line); }
div[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
.small-note { color:#8A95A2; font-size:11px; line-height:1.6; }

@media(max-width:700px) {
 .hero { padding:25px 22px; }
 .book-cover { height:155px; }
 .book-title { font-size:13px; }
}
</style>
""", unsafe_allow_html=True)


# ----------------------------- MATERI 2: HALAMAN LOGIN
try:
    LOGIN_USERNAME = st.secrets.get("BOOKLY_USERNAME", "bookly")
    LOGIN_PASSWORD = st.secrets.get("BOOKLY_PASSWORD", "bookly123")
except Exception:
    LOGIN_USERNAME = "bookly"
    LOGIN_PASSWORD = "bookly123"

if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False

if not st.session_state.is_logged_in:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {display:none;}
    .block-container {max-width:1100px !important; padding-top:3rem !important;}
    .login-shell {
      background:#fff;border:1px solid #E8E3D9;border-radius:26px;
      overflow:hidden;box-shadow:0 22px 55px rgba(20,40,63,.10);
    }
    .login-left {
      background:linear-gradient(145deg,#14283F,#203B58 68%,#31536A);
      min-height:510px;padding:52px 42px;color:#F8F5ED;
      display:flex;flex-direction:column;justify-content:space-between;
    }
    .login-brand {
      font-size:13px;letter-spacing:3px;font-weight:700;color:#E3BD78;
    }
    .login-heading {
      font-family:'Playfair Display',serif;font-size:clamp(32px,4vw,48px);
      line-height:1.12;color:#FFF9EE;margin:28px 0 14px;
    }
    .login-desc {
      color:#D9E0E7;font-size:14px;line-height:1.8;max-width:390px;
    }
    .login-foot {font-size:11px;letter-spacing:1.4px;color:#C4D0DB;}
    .login-right {padding:42px 42px 34px;}
    .login-title {
      font-family:'Playfair Display',serif;font-size:30px;
      color:#18283A;margin:0 0 7px;
    }
    .login-subtitle {font-size:13px;color:#718096;margin-bottom:24px;}
    @media(max-width:700px){
      .login-left{min-height:260px;padding:30px 25px}
      .login-right{padding:30px 24px}
    }
    </style>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown("""
        <div class="login-left">
          <div>
            <div class="login-brand">📚 BOOKLY</div>
            <div class="login-heading">Temukan cerita,<br>temukan duniamu.</div>
            <div class="login-desc">
              Satu ruang untuk menjelajahi buku, menemukan bacaan baru,
              dan mendapatkan rekomendasi yang sesuai dengan minatmu.
            </div>
          </div>
          <div class="login-foot">YOUR PERSONAL DIGITAL LIBRARY</div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="login-title">Selamat datang</div>'
            '<div class="login-subtitle">Masuk untuk melanjutkan ke perpustakaan BOOKLY.</div>',
            unsafe_allow_html=True
        )

        show_password = st.checkbox("Tampilkan password", key="show_login_password")

        with st.form("bookly_login_form", clear_on_submit=False):
            username = st.text_input(
                "Username",
                placeholder="Masukkan username",
                autocomplete="username"
            )
            password = st.text_input(
                "Password",
                type="default" if show_password else "password",
                placeholder="Masukkan password",
                autocomplete="current-password"
            )
            submitted = st.form_submit_button(
                "Masuk ke BOOKLY  →",
                use_container_width=True,
                type="primary"
            )

        if submitted:
            import hmac

            valid_user = hmac.compare_digest(
                str(username).strip(), str(LOGIN_USERNAME)
            )
            valid_pass = hmac.compare_digest(
                str(password), str(LOGIN_PASSWORD)
            )

            if valid_user and valid_pass:
                st.session_state.is_logged_in = True
                st.session_state.login_username = str(username).strip()
                st.rerun()
            else:
                st.error("Username atau password belum sesuai. Silakan coba lagi.")

        st.markdown(
            '<div style="margin-top:22px;color:#8A95A2;font-size:11px;text-align:center">'
            'Akses pribadi · BOOKLY Smart Book Recommendation</div>',
            unsafe_allow_html=True
        )

    st.stop()


# ----------------------------- MATERI 3: FUNGSI PENDUKUNG APLIKASI
@st.cache_resource(show_spinner="Menyiapkan mesin rekomendasi...")
def load_engine(csv_path):
    return MesinRekomendasi(csv_path)


# Penyimpanan lokal untuk buku favorit dan riwayat pencarian.
FAVORITES_FILE = Path(__file__).parent / "bookly_favorites.json"
HISTORY_FILE = Path(__file__).parent / "bookly_search_history.json"


def load_saved_list(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_saved_list(path, data):
    try:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except OSError as error:
        st.warning(f"Data tidak dapat disimpan: {error}")


def add_search_history(query):
    query = str(query).strip()
    if not query:
        return

    history = st.session_state.search_history
    history = [
        item for item in history
        if str(item).casefold() != query.casefold()
    ]

    history.insert(0, query)
    st.session_state.search_history = history[:30]
    save_saved_list(HISTORY_FILE, st.session_state.search_history)


def select_history_query(query):
    st.session_state.global_search = query
    st.session_state.navigation = "Katalog Buku"


# ----------------------------- MATERI 4: DESKRIPSI GENRE BUKU
DESKRIPSI_GENRE = {
    "romance": (
        "Buku ini mengangkat kisah tentang hubungan dan perjalanan emosional "
        "yang dialami oleh para tokohnya. Cerita dapat menggambarkan berbagai "
        "perasaan, seperti cinta, kebahagiaan, kesedihan, pengorbanan, hingga "
        "perjuangan dalam mempertahankan sebuah hubungan. Pembaca diajak "
        "memahami bagaimana perasaan dan pilihan tokoh memengaruhi jalannya cerita."
    ),

    "fantasy": (
        "Buku ini membawa pembaca memasuki dunia imajinatif yang dipenuhi "
        "keajaiban, kekuatan khusus, dan berbagai hal yang tidak selalu ditemukan "
        "dalam kehidupan nyata. Ceritanya dapat melibatkan makhluk unik, dunia "
        "khayalan, perjalanan luar biasa, serta pertarungan antara berbagai "
        "kekuatan. Genre ini cocok bagi pembaca yang menyukai petualangan dan "
        "cerita dengan imajinasi yang luas."
    ),

    "mystery": (
        "Buku ini menyajikan cerita yang berpusat pada teka-teki, rahasia, "
        "atau kejadian yang belum diketahui penyebabnya. Pembaca diajak "
        "mengikuti berbagai petunjuk, mengamati kejadian, dan memahami hubungan "
        "antarperistiwa untuk menemukan jawaban. Alur ceritanya dirancang untuk "
        "membangkitkan rasa penasaran hingga misteri yang ada mulai terungkap."
    ),

    "thriller": (
        "Buku ini menghadirkan cerita yang dipenuhi ketegangan, konflik, "
        "dan situasi yang tidak terduga. Tokoh-tokohnya sering dihadapkan pada "
        "masalah, ancaman, atau keadaan berbahaya yang membutuhkan keputusan "
        "penting. Perkembangan peristiwa yang cepat membuat pembaca terus "
        "penasaran terhadap apa yang akan terjadi selanjutnya."
    ),

    "horror": (
        "Buku ini menghadirkan suasana mencekam melalui cerita tentang "
        "ketakutan, misteri, dan berbagai kejadian menyeramkan. Ceritanya dapat "
        "melibatkan hal-hal gaib, tempat yang dianggap angker, makhluk misterius, "
        "atau ancaman yang sulit dijelaskan. Penggambaran suasana dan peristiwa "
        "dibuat untuk membangun rasa tegang serta ketakutan bagi pembaca."
    ),

    "self_help": (
        "Buku ini berisi pengetahuan, motivasi, dan panduan praktis yang "
        "dapat membantu pembaca mengembangkan kemampuan diri. Pembahasannya "
        "dapat mencakup pengelolaan waktu, pembentukan kebiasaan positif, "
        "kepercayaan diri, pengendalian emosi, hingga cara menghadapi tantangan "
        "kehidupan. Pembaca dapat memperoleh wawasan dan menerapkan hal-hal "
        "yang dipelajari dalam kehidupan sehari-hari."
    ),

    "business": (
        "Buku ini membahas berbagai pengetahuan dan strategi yang berkaitan "
        "dengan dunia bisnis dan pengembangan usaha. Materinya dapat mencakup "
        "pengelolaan keuangan, pemasaran, kepemimpinan, manajemen, serta "
        "pengambilan keputusan dalam menjalankan bisnis. Pembaca dapat "
        "memperoleh pemahaman mengenai cara mengelola usaha dan menghadapi "
        "berbagai tantangan dalam dunia kerja maupun bisnis."
    ),

    "history": (
        "Buku ini mengajak pembaca mengenal berbagai peristiwa penting yang "
        "terjadi pada masa lampau. Pembahasannya dapat mencakup tokoh bersejarah, "
        "perjalanan suatu bangsa, perkembangan peradaban, hingga perubahan "
        "sosial dan budaya. Melalui informasi yang disajikan, pembaca dapat "
        "memahami latar belakang suatu peristiwa serta pengaruhnya terhadap "
        "kehidupan pada masa sekarang."
    ),

    "science": (
        "Buku ini menyajikan pengetahuan ilmiah yang membantu pembaca memahami "
        "berbagai fenomena di sekitar kehidupan. Pembahasannya dapat mencakup "
        "alam, teknologi, lingkungan, tubuh manusia, hingga perkembangan ilmu "
        "pengetahuan. Informasi disampaikan untuk menambah wawasan, mendorong "
        "rasa ingin tahu, dan membantu pembaca memahami suatu hal berdasarkan "
        "penjelasan ilmiah."
    ),

    "novel": (
        "Buku ini menyajikan cerita yang berfokus pada perjalanan tokoh dalam "
        "menghadapi berbagai peristiwa dan konflik kehidupan. Pembaca diajak "
        "mengikuti perkembangan karakter, hubungan antartokoh, serta tantangan "
        "yang muncul sepanjang cerita. Alurnya dapat menghadirkan berbagai emosi "
        "dan memberikan sudut pandang baru tentang kehidupan."
    ),
}


def deskripsi_genre(value):
    """Mengambil deskripsi berdasarkan genre utama buku."""
    if value is None or pd.isna(value):
        return DESKRIPSI_GENRE["novel"]

    genre = str(value).strip().lower()

    # Mengambil genre pertama jika terdapat lebih dari satu genre.
    genre = re.split(r"[,;/|]", genre)[0].strip()

    # Menyamakan format nama genre.
    genre = genre.replace(" ", "_").replace("-", "_")

    # Menyesuaikan beberapa kemungkinan penamaan genre.
    alias_genre = {
        "romantis": "romance",
        "romance": "romance",
        "fantasi": "fantasy",
        "fantasy": "fantasy",
        "misteri": "mystery",
        "mystery": "mystery",
        "horor": "horror",
        "horror": "horror",
        "pengembangan_diri": "self_help",
        "self_help": "self_help",
        "bisnis": "business",
        "business": "business",
        "sejarah": "history",
        "history": "history",
        "sains": "science",
        "science": "science",
        "novel": "novel",
    }

    genre = alias_genre.get(genre, genre)

    return DESKRIPSI_GENRE.get(
        genre,
        DESKRIPSI_GENRE["novel"]
    )


# ----------------------------- MATERI 5: DATA TAMBAHAN BUKU
SYNOPSIS_COLS = [
    "sinopsis", "synopsis", "deskripsi", "description", "desc",
    "ringkasan", "blurb", "summary", "detail", "isi"
]
PRICE_COLS = [
    "harga", "price", "harga_buku", "harga_normal", "harga_jual",
    "harga_rp", "harga_asli", "final_price", "current_price"
]
IMAGE_COLS = [
    "gambar", "image", "image_url", "img", "img_url", "cover",
    "cover_url", "url_gambar", "link_gambar", "foto", "thumbnail", "poster"
]
LINK_COLS = [
    "url", "link", "link_buku", "url_buku", "product_url", "link_produk"
]


def _find_col(columns, candidates):
    lower = {str(c).strip().lower(): c for c in columns}

    for cand in candidates:
        if cand in lower:
            return lower[cand]

    for cand in candidates:
        for low, orig in lower.items():
            if cand in low:
                return orig

    return None


def _norm_title(value):
    return re.sub(r"\s+", " ", str(value)).strip().lower()


@st.cache_data(show_spinner=False)
def load_extra(csv_path):
    raw = None

    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            raw = pd.read_csv(
                csv_path,
                sep=None,
                engine="python",
                encoding=enc
            )
            break
        except Exception:
            continue

    if raw is None:
        return {}, {}

    title_col = _find_col(
        raw.columns,
        ["judul", "title", "nama", "nama_buku", "book_title"]
    )

    cols = {
        "sinopsis": _find_col(raw.columns, SYNOPSIS_COLS),
        "harga": _find_col(raw.columns, PRICE_COLS),
        "gambar": _find_col(raw.columns, IMAGE_COLS),
        "link": _find_col(raw.columns, LINK_COLS),
    }

    lookup = {}

    if title_col is not None:
        for _, r in raw.iterrows():
            key = _norm_title(r[title_col])

            if key in lookup:
                continue

            lookup[key] = {
                k: (r[c] if c is not None else None)
                for k, c in cols.items()
            }

    return lookup, {k: v for k, v in cols.items() if v is not None}


# ----------------------------- MATERI 6: PENGOLAHAN HARGA DAN GAMBAR
def parse_price(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None

    try:
        num = float(value)
    except (TypeError, ValueError):
        digits = re.sub(r"[^\d]", "", str(value))

        if not digits:
            return None

        num = float(digits)

    return int(num) if num > 0 else None


def fmt_rupiah(value):
    price = parse_price(value)

    if price is None:
        return None

    return "Rp " + f"{price:,}".replace(",", ".")


def clean_image_url(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None

    url = str(value).strip().strip('"').strip("'")

    if url.startswith("//"):
        url = "https:" + url

    if url.lower().startswith(("http://", "https://")):
        return url

    return None


def clean_synopsis(value):
    """Membersihkan sinopsis dari tag HTML, spasi berlebih, dan nilai kosong."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None

    synopsis = html.unescape(str(value))
    synopsis = re.sub(r"<[^>]+>", " ", synopsis)
    synopsis = re.sub(r"\s+", " ", synopsis).strip()

    return synopsis if synopsis else None


# ----------------------------- MATERI 7: GOOGLE BOOKS API
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def fetch_online_info(title, author):
    """Mengambil sinopsis, foto sampul, dan harga bila tersedia."""
    result = {"gambar": None, "sinopsis": None, "harga": None}

    a = author if author and author != "Belum tersedia" else ""
    queries = []

    if a:
        queries.append(f"intitle:{title} inauthor:{a}")

    queries.append(f"intitle:{title}")

    for q in queries:
        try:
            r = requests.get(
                "https://www.googleapis.com/books/v1/volumes",
                params={
                    "q": q,
                    "maxResults": 5,
                    "country": "ID",
                    "printType": "books"
                },
                timeout=4,
            )

            if r.status_code != 200:
                continue

            items = r.json().get("items", [])

        except Exception:
            continue

        for it in items:
            info = it.get("volumeInfo", {})

            ratio = difflib.SequenceMatcher(
                None,
                _norm_title(title),
                _norm_title(info.get("title", ""))
            ).ratio()

            if ratio < 0.6:
                continue

            if not result["gambar"]:
                links = info.get("imageLinks") or {}
                img = links.get("thumbnail") or links.get("smallThumbnail")

                if img:
                    result["gambar"] = (
                        img.replace("http://", "https://")
                        .replace("&edge=curl", "")
                    )

            if not result["sinopsis"] and info.get("description"):
                result["sinopsis"] = clean_synopsis(
                    info.get("description")
                )

            if not result["harga"]:
                sale = it.get("saleInfo", {}) or {}

                for key in ("retailPrice", "listPrice"):
                    p = sale.get(key)

                    if (
                        p
                        and p.get("currencyCode") == "IDR"
                        and p.get("amount")
                    ):
                        result["harga"] = p["amount"]
                        break

        if result["gambar"] and result["sinopsis"]:
            break

    return result


def get_book_info(title, author):
    """Mengambil informasi buku dari dataset tanpa pencarian online."""
    info = dict(EXTRA.get(_norm_title(title), {}))
    info["sinopsis"] = clean_synopsis(info.get("sinopsis"))
    info["gambar"] = clean_image_url(info.get("gambar"))
    info["harga"] = parse_price(info.get("harga"))
    return info


# ----------------------------- MATERI 8: FUNGSI PENGOLAHAN DATA BUKU
def fmt_num(value):
    try:
        return f"{int(value):,}".replace(",", ".")
    except (TypeError, ValueError):
        return "—"


def clean_genre(value):
    if not value or value == "tidak_diketahui":
        return "Genre belum tersedia"

    return ", ".join(
        x.replace("_", " ").title()
        for x in str(value).split(",")
    )


def primary_genre(value):
    if not value or value == "tidak_diketahui":
        return "novel"

    return str(value).split(",")[0].strip().lower()


def safe_text(value, fallback="Belum tersedia"):
    if pd.isna(value) or str(value).strip() == "":
        return fallback

    return str(value).strip()


def cover_html(title, author, index=0, image_url=None):
    palettes = [
        ("#263E55", "#54758A"),
        ("#694B48", "#B77C60"),
        ("#344F48", "#75947B"),
        ("#443A62", "#8574A8"),
        ("#7B5B32", "#C49A55"),
        ("#354B67", "#7795B2"),
        ("#684A5C", "#AD8295"),
        ("#3E5659", "#789A91"),
    ]

    c1, c2 = palettes[index % len(palettes)]
    title = html.escape(str(title))
    author = html.escape(str(author))
    gradient = f"linear-gradient(145deg,{c1},{c2})"

    if image_url:
        safe_url = (
            html.escape(image_url, quote=True)
            .replace("'", "%27")
            .replace("(", "%28")
            .replace(")", "%29")
        )

        return f"""
        <div class="book-cover has-img"
             style="background:url('{safe_url}'),{gradient};"></div>
        """

    return f"""
    <div class="book-cover" style="background:{gradient};">
      <div class="cover-top">BOOKLY • LIBRARY EDITION</div>
      <div class="cover-title">{title[:72]}</div>
      <div class="cover-author">{author[:42]}</div>
    </div>
    """


# ----------------------------- MATERI 9: TAMPILAN KARTU BUKU

def render_book_card(row, index, key_prefix="book"):
    title = safe_text(row.get("judul"))
    author = safe_text(row.get("author"))
    categories = str(row.get("kategori", ""))

    current_preference = st.session_state.get("active_genre", "semua")

    if (
        current_preference != "semua"
        and current_preference in categories.split(",")
    ):
        genre = current_preference
    else:
        genre = primary_genre(row.get("kategori"))

    rating = row.get("original_rating", 0)

    # Data tambahan dari CSV dan Google Books.
    info = get_book_info(title, author)
    image_url = clean_image_url(info.get("gambar"))
    price_text = fmt_rupiah(info.get("harga"))

    # Deskripsi ditampilkan berdasarkan genre buku.
    genre_description = deskripsi_genre(row.get("kategori"))

    gramedia_link = (
        clean_image_url(info.get("link"))
        or f"https://www.gramedia.com/search?q={quote_plus(title)}"
    )

    with st.container():
        price_block = (
            f'<div class="book-price">{html.escape(price_text)} '
            f'<small>· info harga</small></div>'
            if price_text
            else
            '<div class="book-price">'
            '<small>Cek harga terbaru di Gramedia ↓</small></div>'
        )

        st.markdown(
            f'<div class="book-card">'
            f'{cover_html(title, author, index, image_url)}'
            f'<div class="book-title">{html.escape(title)}</div>'
            f'<div class="book-author">{html.escape(author)}</div>'
            f'<div style="margin-top:8px">'
            f'<span class="pill">'
            f'{html.escape(genre.replace("_", " ").title())}'
            f'</span></div>'
            f'<div class="book-meta">'
            f'<span>★ {float(rating):.2f}</span>'
            f'<span>{fmt_num(row.get("rating_count", 0))} penilai</span>'
            f'</div>'
            f'{price_block}'
            f'</div>',
            unsafe_allow_html=True
        )

        with st.expander("📖 Deskripsi Genre & harga"):
            genre_html = html.escape(genre_description)
            price_html = (
                html.escape(price_text)
                if price_text
                else "Belum tersedia di data"
            )

            st.markdown(
                f'<div class="detail-box">'
                f'<div class="lbl">Deskripsi Genre</div>{genre_html}'
                f'<div class="lbl">Harga (sesuai data)</div>{price_html}'
                f'<div style="margin-top:6px">'
                f'<a href="{html.escape(gramedia_link, quote=True)}" '
                f'target="_blank">Cek harga terbaru di Gramedia ↗</a>'
                f'</div>'
                f'<div style="font-size:10px;color:#8A95A2;margin-top:6px">'
                f'Harga hanya sebagai informasi dan dapat berubah sewaktu-waktu.'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Status buku favorit
        liked = st.session_state.get("liked_books", [])
        is_liked = title in liked

        if st.button(
            "❤️  Buku disukai" if is_liked else "♡  Suka buku ini",
            key=f"{key_prefix}_{index}_{re.sub('[^a-zA-Z0-9]','',title)[:18]}",
            use_container_width=True
        ):
            liked = st.session_state.get("liked_books", [])

            if title in liked:
                # Menghapus buku dari favorit jika diklik kembali
                liked.remove(title)
                st.toast(f"{title} dihapus dari favorit")
            else:
                # Menambahkan buku ke favorit
                liked.append(title)
                st.session_state.active_genre = genre
                st.session_state.last_action = f"Kamu menyukai buku {title}"
                st.toast(
                    f"Preferensi diperbarui: {genre.replace('_', ' ').title()}"
                )

            st.session_state.liked_books = liked
            save_saved_list(FAVORITES_FILE, liked)
            st.rerun()

# ----------------------------- MATERI 10: TOP 10 DAN FILTER GENRE
def get_top_books(engine, n=10):
    data = engine.df.copy()

    if "weighted_rating" in data.columns:
        return data.sort_values(
            ["weighted_rating", "rating_count"],
            ascending=False
        ).head(n)

    return data.sort_values(
        ["original_rating", "rating_count"],
        ascending=False
    ).head(n)


def genre_filter(data, genre):
    if not genre or genre == "semua":
        return data

    return data[
        data["kategori"].fillna("").str.lower().str.contains(
            re.escape(genre),
            regex=True
        )
    ]


# ----------------------------- MATERI 11: MEMUAT DATASET DAN MESIN ML
csv_path = cari_csv(Path(__file__).parent)

if csv_path is None:
    st.error(
        "Dataset tidak ditemukan. Pastikan file CSV Gramedia berada satu folder dengan app.py."
    )
    st.stop()

engine = load_engine(str(csv_path))
df = engine.df.copy()
EXTRA, EXTRA_COLS = load_extra(str(csv_path))


# ----------------------------- MATERI 12: SESSION STATE APLIKASI
if "active_genre" not in st.session_state:
    st.session_state.active_genre = "semua"

if "liked_books" not in st.session_state:
    st.session_state.liked_books = load_saved_list(FAVORITES_FILE)

if "search_history" not in st.session_state:
    st.session_state.search_history = load_saved_list(HISTORY_FILE)

if "last_action" not in st.session_state:
    st.session_state.last_action = ""


# ----------------------------- MATERI 13: PENCARIAN DAN DETEKSI GENRE
all_genres = [g for g, _ in engine.daftar_genre()]

genre_lookup = {
    g.replace("_", " ").lower(): g
    for g in all_genres
}
genre_lookup.update({
    g.lower(): g
    for g in all_genres
})


def detect_genre(text):
    q = text.lower().strip()

    for label, raw in sorted(
        genre_lookup.items(),
        key=lambda x: len(x[0]),
        reverse=True
    ):
        if label in q:
            return raw

    return None


# ----------------------------- MATERI 14: NAVIGASI SIDEBAR
with st.sidebar:
    st.markdown("## BOOKLY")

    st.markdown(
        "<div style='font-size:10px;letter-spacing:2px;"
        "color:#D6AD62!important;margin-top:-12px'>"
        "SMART BOOK RECOMMENDATION</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "NAVIGASI",
        [
            "Dashboard",
            "Katalog Buku",
            "Rekomendasi AI",
            "Buku Favorit",
            "Riwayat Pencarian",
            "Top 10 Buku",
            "Tentang Sistem"
        ],
        label_visibility="collapsed",
        key="navigation",
    )

    st.markdown("---")
    st.caption(
        f"Masuk sebagai: {st.session_state.get('login_username', 'Pengguna')}"
    )

    if st.button("↪  Keluar", use_container_width=True):
        st.session_state.is_logged_in = False
        st.session_state.pop("login_username", None)
        st.rerun()

    st.markdown("---")
    st.markdown("**PREFERENSI KAMU**")

    st.caption(
        f"Genre aktif: "
        f"{st.session_state.active_genre.replace('_', ' ').title() if st.session_state.active_genre != 'semua' else 'Semua genre'}"
    )
    st.caption(f"Buku disukai: {len(st.session_state.liked_books)}")

    if st.button("Reset preferensi", use_container_width=True):
        st.session_state.active_genre = "semua"
        st.session_state.liked_books = []
        save_saved_list(FAVORITES_FILE, [])
        st.session_state.last_action = ""
        st.rerun()

    st.markdown("---")

    st.markdown(
        "<div style='font-size:10px;color:#AAB8C6!important'>"
        "Powered by TF-IDF · Cosine Similarity</div>",
        unsafe_allow_html=True
    )


# ----------------------------- MATERI 15: PENCARIAN BUKU GLOBAL
search_col, search_btn_col = st.columns([5, 1])

with search_col:
    global_query = st.text_input(
        "Cari buku atau genre",
        placeholder="Coba ketik: horor, novel, romance, atau judul buku...",
        label_visibility="collapsed",
        key="global_search"
    )

with search_btn_col:
    search_clicked = st.button(
        "⌕  Cari",
        use_container_width=True,
        type="primary"
    )

if search_clicked and global_query.strip():
    add_search_history(global_query)

    found_genre = detect_genre(global_query)

    if found_genre:
        st.session_state.active_genre = found_genre
        st.session_state.last_action = (
            f"Kamu mencari genre {found_genre.replace('_', ' ').title()}"
        )
        st.toast(
            f"Dashboard disesuaikan dengan genre "
            f"{found_genre.replace('_', ' ').title()}"
        )
    else:
        st.session_state.last_action = (
            f"Hasil pencarian: {global_query.strip()}"
        )

    st.rerun()


# ----------------------------- MATERI 16: HEADER UTAMA BOOKLY
active = st.session_state.active_genre

active_label = (
    active.replace("_", " ").title()
    if active != "semua"
    else "Semua Genre"
)

st.markdown(
    f'<div class="hero">'
    f'<div class="eyebrow">YOUR PERSONAL DIGITAL LIBRARY</div>'
    f'<h1>Temukan cerita,<br>temukan duniamu.</h1>'
    f'<p>Jelajahi ribuan koleksi buku dan dapatkan rekomendasi '
    f'yang menyesuaikan minat bacamu. '
    f'Dashboard kamu saat ini berfokus pada '
    f'<b>{html.escape(active_label)}</b>.</p></div>',
    unsafe_allow_html=True
)


# ----------------------------- MATERI 17: HALAMAN DASHBOARD
if page == "Dashboard":
    st.markdown(
        '<div class="section-head"><div>'
        '<h2>Dashboard personal</h2>'
        '<p>Ringkasan koleksi dan aktivitas membaca kamu.</p>'
        '</div></div>',
        unsafe_allow_html=True
    )

    active_data = genre_filter(df, active)

    total_books = (
        len(active_data)
        if active != "semua"
        else len(df)
    )

    avg_rating = (
        active_data["original_rating"].mean()
        if total_books
        else 0
    )

    bestseller_count = (
        int(active_data["isBestSeller"].sum())
        if total_books and "isBestSeller" in active_data
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        (c1, "KOLEKSI BUKU", fmt_num(total_books), "Sesuai genre aktif"),
        (c2, "RATA-RATA RATING", f"{avg_rating:.2f} ★" if total_books else "—", "Dari data penilaian"),
        (c3, "BESTSELLER", fmt_num(bestseller_count), "Dalam koleksi terpilih"),
        (c4, "BUKU DISUKAI", fmt_num(len(st.session_state.liked_books)), "Preferensi kamu"),
    ]

    for col, label, value, hint in metrics:
        with col:
            st.markdown(
                f'<div class="metric">'
                f'<div class="label">{label}</div>'
                f'<div class="value">{value}</div>'
                f'<div class="hint">{hint}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-head"><div>'
        '<h2>Jelajahi berdasarkan genre</h2>'
        '<p>Pilih genre untuk mengubah isi dashboard dan rekomendasi.</p>'
        '</div></div>',
        unsafe_allow_html=True
    )

    genre_options = ["semua"] + all_genres
    genre_cols = st.columns(5)

    for i, genre in enumerate(genre_options[:15]):
        with genre_cols[i % 5]:
            label = (
                "Semua"
                if genre == "semua"
                else genre.replace("_", " ").title()
            )

            if st.button(
                ("✓ " if active == genre else "＋ ") + label,
                key=f"genre_pick_{genre}",
                use_container_width=True
            ):
                st.session_state.active_genre = genre
                st.rerun()

    st.markdown(
        '<div class="section-head"><div>'
        '<h2>Rekomendasi untukmu</h2>'
        '<p>Buku pilihan dari genre yang sedang kamu minati.</p>'
        '</div></div>',
        unsafe_allow_html=True
    )

    if active == "semua":
        featured = get_top_books(engine, 8)
    else:
        try:
            featured = engine.dari_minat(kategori=[active], n=8)

            if featured.empty:
                featured = genre_filter(df, active).head(8)

        except Exception:
            featured = genre_filter(df, active).head(8)

    if len(featured):
        cols = st.columns(4)

        for i, (_, row) in enumerate(featured.head(8).iterrows()):
            with cols[i % 4]:
                render_book_card(row, i, "dashboard")
    else:
        st.info("Belum ada buku yang cocok dengan genre ini.")

    st.markdown(
        '<div class="section-head"><div>'
        '<h2>Top 10 buku pilihan</h2>'
        '<p>Peringkat berdasarkan weighted rating dan jumlah penilai.</p>'
        '</div></div>',
        unsafe_allow_html=True
    )

    top10 = get_top_books(engine, 10)

    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        a, b, c = st.columns([0.5, 5, 1.2])

        with a:
            st.markdown(
                f'<div class="rank">{rank:02d}</div>',
                unsafe_allow_html=True
            )

        with b:
            st.markdown(
                f"**{safe_text(row.get('judul'))}**  \n"
                f"<small style='color:#8290A0'>"
                f"{safe_text(row.get('author'))} · "
                f"{clean_genre(row.get('kategori'))}</small>",
                unsafe_allow_html=True
            )

        with c:
            st.markdown(
                f"**★ {float(row.get('original_rating', 0)):.2f}**"
            )

    st.caption(
        "Catatan: urutan Top 10 menggunakan weighted rating "
        "yang dihitung oleh mesin rekomendasi, bukan sekadar rating mentah."
    )


# ----------------------------- MATERI 18: HALAMAN KATALOG BUKU
elif page == "Katalog Buku":
    st.markdown(
        '<div class="section-head"><div>'
        '<h2>Katalog buku</h2>'
        '<p>Telusuri koleksi dan simpan buku yang menarik perhatianmu.</p>'
        '</div></div>',
        unsafe_allow_html=True
    )

    f1, f2, f3 = st.columns([2, 1, 1])

    with f1:
        q = st.text_input(
            "Cari judul atau penulis",
            value=global_query,
            key="catalog_query"
        )

    with f2:
        selected_genre = st.selectbox(
            "Genre",
            ["Semua"] + [
                g.replace("_", " ").title()
                for g in all_genres
            ]
        )

    with f3:
        sort_by = st.selectbox(
            "Urutkan",
            ["Rating tertinggi", "Paling populer", "Judul A–Z"]
        )

    catalog = df.copy()

    if q.strip():
        mask = (
            catalog["judul"].str.contains(q, case=False, na=False)
            | catalog["author"].str.contains(q, case=False, na=False)
            | catalog["kategori"].str.contains(q, case=False, na=False)
        )

        catalog = catalog[mask]
        detected = detect_genre(q)

        if detected:
            st.session_state.active_genre = detected

    if selected_genre != "Semua":
        raw = next(
            (
                g for g in all_genres
                if g.replace("_", " ").title() == selected_genre
            ),
            selected_genre.lower()
        )

        catalog = genre_filter(catalog, raw)

    if sort_by == "Rating tertinggi":
        catalog = catalog.sort_values(
            ["original_rating", "rating_count"],
            ascending=False
        )
    elif sort_by == "Paling populer":
        catalog = catalog.sort_values(
            "rating_count",
            ascending=False
        )
    else:
        catalog = catalog.sort_values("judul")

    st.caption(f"Menampilkan {fmt_num(len(catalog))} buku.")

    if len(catalog):
        per_page = 12
        pages = max(1, (len(catalog) + per_page - 1) // per_page)

        page_no = st.number_input(
            "Halaman",
            min_value=1,
            max_value=pages,
            value=1,
            step=1
        )

        shown = catalog.iloc[
            (page_no - 1) * per_page:page_no * per_page
        ]

        cols = st.columns(4)

        for i, (_, row) in enumerate(shown.iterrows()):
            with cols[i % 4]:
                render_book_card(row, int(row.name), "catalog")

    else:
        st.warning("Buku tidak ditemukan. Coba kata kunci atau genre lain.")


# ----------------------------- MATERI 19: HALAMAN REKOMENDASI AI
elif page == "Rekomendasi AI":
    st.markdown(
        '<div class="section-head"><div>'
        '<h2>Smart recommendation</h2>'
        '<p>Temukan buku yang sesuai dengan selera dan buku favoritmu.</p>'
        '</div></div>',
        unsafe_allow_html=True
    )

    tab_interest, tab_book = st.tabs([
        "Berdasarkan minat",
        "Berdasarkan buku favorit"
    ])

    with tab_interest:
        genre_names = {
            g.replace("_", " ").title(): g
            for g in all_genres
        }

        chosen_genres = st.multiselect(
            "Genre yang kamu sukai",
            list(genre_names.keys()),
            default=[
                active.replace("_", " ").title()
            ] if active in all_genres else []
        )

        author_choice = st.selectbox(
            "Penulis favorit (opsional)",
            ["Tidak memilih"] + engine.daftar_penulis()
        )

        keyword = st.text_input(
            "Kata kunci buku (opsional)",
            placeholder="Contoh: misteri, cinta, petualangan"
        )

        if st.button("✨  Buat rekomendasi", type="primary"):
            if (
                not chosen_genres
                and author_choice == "Tidak memilih"
                and not keyword.strip()
            ):
                st.warning(
                    "Pilih setidaknya satu genre, penulis, atau kata kunci."
                )

            else:
                raw_genres = [
                    genre_names[x]
                    for x in chosen_genres
                ]

                result = engine.dari_minat(
                    kategori=raw_genres,
                    penulis=(
                        None
                        if author_choice == "Tidak memilih"
                        else author_choice
                    ),
                    kata_kunci=keyword.strip(),
                    n=12
                )

                if result.empty:
                    st.warning(
                        "Belum ditemukan kecocokan. Coba genre atau kata kunci lain."
                    )
                else:
                    if raw_genres:
                        st.session_state.active_genre = raw_genres[0]

                    st.success(
                        f"Ditemukan {len(result)} rekomendasi untuk preferensimu."
                    )

                    cols = st.columns(4)

                    for i, (_, row) in enumerate(result.iterrows()):
                        with cols[i % 4]:
                            render_book_card(
                                row,
                                i,
                                "recommend_interest"
                            )

                            st.caption(
                                f"Kesamaan profil: "
                                f"{float(row.get('kemiripan', 0)) * 100:.0f}%"
                            )

    with tab_book:
        titles = engine.daftar_judul()
        labels = [label for _, label in titles]

        selected = st.selectbox(
            "Pilih buku yang kamu sukai",
            labels,
            index=None,
            placeholder="Ketik judul buku..."
        )

        if st.button(
            "Cari buku serupa",
            type="primary",
            disabled=selected is None
        ):
            index = dict(
                (label, idx)
                for idx, label in titles
            )[selected]

            source = engine.df.loc[index]
            source_genre = primary_genre(source["kategori"])

            st.session_state.active_genre = source_genre

            result = engine.dari_judul(index, n=12)

            st.success(
                f"Rekomendasi berdasarkan buku: {source['judul']}"
            )

            if not result.empty:
                cols = st.columns(4)

                for i, (_, row) in enumerate(result.iterrows()):
                    with cols[i % 4]:
                        render_book_card(
                            row,
                            i,
                            "recommend_book"
                        )

                        st.caption(
                            f"Kesamaan profil: "
                            f"{float(row.get('kemiripan', 0)) * 100:.0f}%"
                        )


# ----------------------------- MATERI 20: HALAMAN BUKU FAVORIT
elif page == "Buku Favorit":
    st.markdown(
        '<div class="section-head"><div>'
        '<h2>Buku yang kamu sukai</h2>'
        '<p>Setiap tombol suka membantu BOOKLY memahami preferensi genre kamu.</p>'
        '</div></div>',
        unsafe_allow_html=True
    )

    if not st.session_state.liked_books:
        st.info(
            "Belum ada buku favorit. Buka Katalog Buku, "
            "lalu tekan tombol ♡ Suka buku ini."
        )

    else:
        for title in st.session_state.liked_books:
            match = df[df["judul"] == title]

            if len(match):
                row = match.iloc[0]

                st.markdown(
                    f"**{title}** · "
                    f"{safe_text(row.get('author'))} · "
                    f"{clean_genre(row.get('kategori'))}"
                )

                if st.button(
                    "Hapus dari favorit",
                    key=f"remove_{re.sub('[^a-zA-Z0-9]', '', title)[:25]}"
                ):
                    st.session_state.liked_books.remove(title)

                    save_saved_list(
                        FAVORITES_FILE,
                        st.session_state.liked_books
                    )

                    st.rerun()

        st.markdown("---")

        liked_genres = []

        for title in st.session_state.liked_books:
            match = df[df["judul"] == title]

            if len(match):
                liked_genres.append(
                    primary_genre(match.iloc[0]["kategori"])
                )

        if liked_genres:
            preferred = pd.Series(liked_genres).value_counts().index[0]

            st.session_state.active_genre = preferred

            st.markdown(
                f"**Genre yang paling sering kamu sukai:** "
                f"{preferred.replace('_', ' ').title()}"
            )

            suggestions = engine.dari_minat(
                kategori=[preferred],
                n=8
            )

            if not suggestions.empty:
                cols = st.columns(4)

                for i, (_, row) in enumerate(suggestions.iterrows()):
                    with cols[i % 4]:
                        render_book_card(
                            row,
                            i,
                            "favorite_suggest"
                        )


# ----------------------------- MATERI 21: RIWAYAT PENCARIAN
elif page == "Riwayat Pencarian":
    st.markdown(
        '<div class="section-head"><div>'
        '<h2>Riwayat Pencarian</h2>'
        '<p>Daftar buku atau genre yang pernah kamu cari.</p>'
        '</div></div>',
        unsafe_allow_html=True
    )

    if not st.session_state.search_history:
        st.info("Belum ada riwayat pencarian.")

    else:
        clear_col, _ = st.columns([1, 5])

        with clear_col:
            if st.button(
                "Hapus semua riwayat",
                use_container_width=True
            ):
                st.session_state.search_history = []
                save_saved_list(HISTORY_FILE, [])
                st.rerun()

        for i, query in enumerate(st.session_state.search_history):
            icon_col, query_col = st.columns([0.55, 6])

            with icon_col:
                st.markdown(
                    "<div style='font-size:22px;color:#8993A0;padding:5px 0'>◷</div>",
                    unsafe_allow_html=True
                )

            with query_col:
                st.button(
                    query,
                    key=f"history_query_{i}",
                    on_click=select_history_query,
                    args=(query,),
                    use_container_width=True
                )


# ----------------------------- MATERI 22: HALAMAN TOP 10 BUKU

elif page == "Top 10 Buku":
    st.markdown(
        '<div class="section-head"><div>'
        '<h2>Top 10 buku terbaik</h2>'
        '<p>Peringkat berdasarkan weighted rating dan jumlah penilai.</p>'
        '</div></div>',
        unsafe_allow_html=True
    )

    top = get_top_books(engine, 10)

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        left, mid, right = st.columns([0.7, 4.5, 1.2])

        title = safe_text(row.get("judul"))
        author = safe_text(row.get("author"))

        info = get_book_info(title, author)
        price_text = fmt_rupiah(info.get("harga"))
        genre_description = deskripsi_genre(row.get("kategori"))

        link = (
            info.get("link")
            or f"https://www.gramedia.com/search?q={quote_plus(title)}"
        )

        with left:
            st.markdown(
                f'<div class="rank">{rank:02d}</div>',
                unsafe_allow_html=True
            )

        with mid:
            st.markdown(
                f"**{title}**  \n"
                f"{author} · {clean_genre(row.get('kategori'))}"
            )

            st.caption(
                f"{fmt_num(row.get('rating_count', 0))} penilai · "
                f"{fmt_num(row.get('review_count', 0))} ulasan · "
                f"{fmt_num(row.get('jumlah_halaman'))} halaman"
            )

            with st.expander("📖 Deskripsi Genre & harga"):
                st.markdown(
                    f'<div class="detail-box">'
                    f'<div class="lbl">Deskripsi Genre</div>'
                    f'{html.escape(genre_description)}'
                    f'<div class="lbl">Harga (sesuai data)</div>'
                    f'{html.escape(str(price_text)) if price_text else "Belum tersedia di data"}'
                    f'<div style="margin-top:6px">'
                    f'<a href="{html.escape(link, quote=True)}" target="_blank">'
                    f'Cek harga terbaru di Gramedia ↗</a></div>'
                    f'<div style="font-size:10px;color:#8A95A2;margin-top:6px">'
                    f'Harga hanya sebagai informasi dan dapat berubah sewaktu-waktu.'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

        with right:
            st.markdown(
                f"### ★ {float(row.get('original_rating', 0)):.2f}"
            )

        st.divider()


# ----------------------------- MATERI 23: TENTANG SISTEM
elif page == "Tentang Sistem":
    st.markdown(
        '<div class="section-head"><div>'
        '<h2>Di balik rekomendasi BOOKLY</h2>'
        '<p>Penjelasan singkat teknologi yang digunakan dalam sistem.</p>'
        '</div></div>',
        unsafe_allow_html=True
    )

    info1, info2 = st.columns(2)

    with info1:
        st.markdown(
            '<div class="info-panel">'
            '<h3>Content-Based Filtering</h3>'
            '<p>Sistem membandingkan karakteristik buku seperti genre, '
            'penulis, dan kata pada judul. Rekomendasi diberikan berdasarkan '
            'kemiripan profil buku dengan minat pengguna atau buku yang dipilih.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    with info2:
        st.markdown(
            '<div class="info-panel">'
            '<h3>TF-IDF</h3>'
            '<p>TF-IDF mengubah informasi teks menjadi bobot numerik agar '
            'karakteristik buku dapat diproses dan dibandingkan oleh model.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    st.write("")

    info3, info4 = st.columns(2)

    with info3:
        st.markdown(
            '<div class="info-panel">'
            '<h3>Cosine Similarity</h3>'
            '<p>Mengukur tingkat kemiripan dua profil dalam bentuk vektor. '
            'Nilai yang lebih tinggi menunjukkan profil yang lebih mirip.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    with info4:
        st.markdown(
            '<div class="info-panel">'
            '<h3>Weighted Rating</h3>'
            '<p>Menggabungkan rating buku dengan jumlah penilai sehingga '
            'peringkat tidak hanya ditentukan oleh buku yang memiliki '
            'sedikit penilaian.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    st.write("")
    st.markdown("**Alur kerja sistem**")

    st.code(
        "Dataset buku → Pembersihan data → Pembentukan fitur TF-IDF "
        "→ Cosine Similarity → Weighted Rating → Rekomendasi buku",
        language="text"
    )

    st.caption(
        "Model rekomendasi menggunakan implementasi dari file "
        "mesin_rekomendasi.py. Buku favorit dan riwayat pencarian disimpan "
        "secara lokal agar tetap tersedia saat aplikasi dibuka kembali. "
        "Sinopsis dan foto sampul diambil dari Google Books API bila "
        "tidak tersedia di dataset."
    )

    st.markdown("**Kolom data tambahan yang terdeteksi di CSV**")

    if EXTRA_COLS:
        st.write({
            k: str(v)
            for k, v in EXTRA_COLS.items()
        })
    else:
        st.caption(
            "Tidak ada kolom sinopsis / harga / gambar di CSV, "
            "jadi data diambil dari Google Books API."
        )


# ----------------------------- MATERI 24: FOOTER APLIKASI
st.markdown("---")

st.markdown(
    "<div style='text-align:center;color:#929BA5;font-size:10px;"
    "letter-spacing:1px'>"
    "BOOKLY · SMART BOOK RECOMMENDATION SYSTEM · "
    "BUILT WITH PYTHON & MACHINE LEARNING"
    "</div>",
    unsafe_allow_html=True,
)
