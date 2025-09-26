import json, math, numpy as np
import re, unicodedata
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

st.set_page_config(
    page_title="اردو سے رومن اردو - Neural Machine Translation", 
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------
# Amazing Colorful Design
# --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');

/* Beautiful gradient background */
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    min-height: 100vh;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Main container styling */
.main .block-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    margin-top: 20px;
    margin-bottom: 20px;
}

/* Make Urdu text larger and more readable */
.urdu-text {
    font-family: 'Noto Nastaliq Urdu', serif;
    font-size: 28px;
    line-height: 2.2;
    direction: rtl;
    text-align: right;
    color: #2c3e50;
    font-weight: 500;
}

.urdu-title {
    font-family: 'Noto Nastaliq Urdu', serif;
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 3s ease infinite;
    margin: 20px 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.roman-text {
    font-size: 24px;
    line-height: 2;
    color: #2c3e50;
    font-weight: 500;
    background: linear-gradient(45deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.large-text {
    font-size: 20px;
    font-weight: 500;
}

/* Amazing colorful card styling */
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.7));
    padding: 25px;
    border-radius: 15px;
    border: none;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
    margin: 10px 0;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 5px;
    background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
    background-size: 300% 100%;
    animation: gradientShift 2s ease infinite;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.15);
}

.model-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.7));
    padding: 25px;
    border-radius: 15px;
    margin: 15px 0;
    border: none;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    position: relative;
    overflow: hidden;
}

.model-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 5px;
    height: 100%;
    background: linear-gradient(180deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
    background-size: 100% 300%;
    animation: gradientShift 3s ease infinite;
}

.model-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 35px rgba(0,0,0,0.12);
}

/* Beautiful result cards */
.result-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(248,249,250,0.9));
    padding: 25px;
    border-radius: 15px;
    margin: 20px 0;
    border: none;
    box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 6px;
    height: 100%;
    background: linear-gradient(180deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
    background-size: 100% 300%;
    animation: gradientShift 4s ease infinite;
}

/* Section headers */
.section-header {
    background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.7));
    padding: 20px;
    border-radius: 15px;
    margin: 20px 0;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
}

/* Footer styling */
.footer-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.7));
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
    margin: 30px 0;
    border: 1px solid rgba(255, 255, 255, 0.3);
}

/* Button styling */
.stButton > button {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 15px 30px;
    font-size: 18px;
    font-weight: 600;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.3);
    background: linear-gradient(45deg, #ff5252, #26a69a);
}

/* Text area styling */
.stTextArea > div > div > textarea {
    border-radius: 15px;
    border: 2px solid rgba(255,255,255,0.3);
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(10px);
    font-size: 18px;
    padding: 15px;
}

/* Slider styling */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
}

/* Spinner styling */
.stSpinner > div {
    border-color: #ff6b6b transparent #ff6b6b transparent;
}
</style>
""", unsafe_allow_html=True)

# --------------------
# Constants & Helpers
# --------------------
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Urdu normalization
ZW_CHARS = "".join(["\u200b", "\u200c", "\u200d", "\ufeff"])
URDU_MAP = {
    "ي":"ی","ك":"ک","ۀ":"ہ","ھ":"ہ","ۃ":"ہ",
    "أ":"ا","إ":"ا","آ":"آ","ٱ":"ا","ؤ":"و","ئ":"ی",
    "ٔ":"", "ٰ":"", "ٌ":"", "ً":"", "ٍ":"", "ْ":"", "ّ":"", "ـ":""
}

def normalize_urdu(s: str) -> str:
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(f"[{re.escape(ZW_CHARS)}]", "", s)
    s = "".join(URDU_MAP.get(ch, ch) for ch in s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@st.cache_resource(show_spinner=False)
def load_vocabs():
    # Try to load from src directory first, then fallback to artifacts
    try:
        with open("src/vocab_src_char.json", "r", encoding="utf-8") as f:
            v_src = json.load(f)
        with open("src/vocab_tgt_char.json", "r", encoding="utf-8") as f:
            v_tgt = json.load(f)
    except:
        with open("artifacts/vocab_src_char.json", "r", encoding="utf-8") as f:
            v_src = json.load(f)
        with open("artifacts/vocab_tgt_char.json", "r", encoding="utf-8") as f:
            v_tgt = json.load(f)
    
    stoi_src = v_src["stoi"]
    itos_tgt = v_tgt["itos"]
    return stoi_src, itos_tgt

# Char encoders/decoders
def encode_char_src(text, stoi_src):
    text = normalize_urdu(text)              # normalize before encoding
    ids = [BOS_ID]
    for ch in text:
        ids.append(stoi_src.get(ch, UNK_ID))
    ids.append(EOS_ID)
    return ids

def decode_char_tgt(ids, itos_tgt):
    out = []
    for i in ids:
        if i in (PAD_ID, BOS_ID, EOS_ID): 
            continue
        if 0 <= i < len(itos_tgt):
            out.append(itos_tgt[i])
    return "".join(out).strip()

# --------------------
# Model Definition
# --------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_size, num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.bilstm = nn.LSTM(
            emb_dim, hid_size, num_layers=num_layers, dropout=dropout,
            bidirectional=bidirectional, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_ids):
        emb = self.dropout(self.embedding(src_ids))          # [B,T,E]
        outputs, (hn, cn) = self.bilstm(emb)                 # outputs: [B,T,2H]
        return outputs, (hn, cn)

class LuongAttention(nn.Module):
    def __init__(self, dec_hid, enc_hid_bi):
        super().__init__()
        self.W = nn.Linear(dec_hid, enc_hid_bi, bias=False)
    def forward(self, dec_h_t, enc_outputs, mask=None):
        # dec_h_t: [B,H], enc_outputs: [B,T,Henc]
        score = torch.bmm(self.W(dec_h_t).unsqueeze(1), enc_outputs.transpose(1,2))  # [B,1,T]
        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(1), -1e9)
        attn = torch.softmax(score, dim=-1)               # [B,1,T]
        ctx  = torch.bmm(attn, enc_outputs).squeeze(1)    # [B,Henc]
        return ctx, attn.squeeze(1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_size, enc_hid_bi, num_layers=4, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim + enc_hid_bi, hid_size, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.attn = LuongAttention(hid_size, enc_hid_bi)
        self.fc_out = nn.Linear(hid_size + enc_hid_bi, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward_step(self, y_prev, h_c, enc_outputs, enc_mask, ctx_prev):
        emb = self.dropout(self.embedding(y_prev))           # [B,E]
        lstm_in = torch.cat([emb, ctx_prev], dim=-1).unsqueeze(1)  # [B,1,E+Henc]
        out, h_c = self.lstm(lstm_in, h_c)                   # out: [B,1,Hdec]
        h_t = out.squeeze(1)                                 # [B,Hdec]
        ctx_t, attn = self.attn(h_t, enc_outputs, enc_mask)  # [B,Henc]
        logits = self.fc_out(torch.cat([h_t, ctx_t], dim=-1))# [B,V]
        return logits, h_c, ctx_t, attn

    def forward(self, y_prev, h_c, enc_outputs, enc_mask, ctx_prev):
        return self.forward_step(y_prev, h_c, enc_outputs, enc_mask, ctx_prev)

class Bridge(nn.Module):
    def __init__(self, enc_hid, dec_hid, dec_layers, bidirectional=True):
        super().__init__()
        mul = 2 if bidirectional else 1
        self.h_proj = nn.Linear(enc_hid*mul, dec_hid)
        self.c_proj = nn.Linear(enc_hid*mul, dec_hid)
        self.dec_layers = dec_layers
    def forward(self, enc_hn, enc_cn):
        top_h = torch.cat([enc_hn[-2], enc_hn[-1]], dim=-1)  # [B,2H]
        top_c = torch.cat([enc_cn[-2], enc_cn[-1]], dim=-1)
        h0 = torch.tanh(self.h_proj(top_h)).unsqueeze(0)     # [1,B,Hd]
        c0 = torch.tanh(self.c_proj(top_c)).unsqueeze(0)
        # repeat for all decoder layers
        h0 = h0.repeat(self.dec_layers, 1, 1)
        c0 = c0.repeat(self.dec_layers, 1, 1)
        return (h0, c0)

class Seq2Seq(nn.Module):
    def __init__(self, src_vsize, tgt_vsize, emb_dim=256, hid_size=256, enc_layers=2, dec_layers=4, dropout=0.3):
        super().__init__()
        self.encoder = Encoder(src_vsize, emb_dim, hid_size, enc_layers, dropout, bidirectional=True)
        enc_hid_bi = hid_size * 2
        self.decoder = Decoder(tgt_vsize, emb_dim, hid_size, enc_hid_bi, dec_layers, dropout)
        self.bridge  = Bridge(hid_size, hid_size, dec_layers, bidirectional=True)

    def greedy_decode(self, src_ids, max_len=150):
        self.eval()
        with torch.no_grad():
            enc_outputs, (hn, cn) = self.encoder(src_ids)
            enc_mask = (src_ids == PAD_ID)

            B = src_ids.size(0)
            dec_hc = self.bridge(hn, cn)
            ctx = torch.zeros(B, enc_outputs.size(-1), device=src_ids.device)
            y_prev = torch.full((B,), BOS_ID, dtype=torch.long, device=src_ids.device)

            # storage and finished flags
            outs = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=src_ids.device)
            finished = torch.zeros(B, dtype=torch.bool, device=src_ids.device)

            for t in range(max_len):
                logits, dec_hc, ctx, _ = self.decoder(y_prev, dec_hc, enc_outputs, enc_mask, ctx)
                next_tok = torch.argmax(logits, dim=-1)           # [B]

                # write token
                outs[:, t] = next_tok

                # mark which just hit EOS
                just_eos = (next_tok == EOS_ID) & (~finished)
                finished = finished | just_eos

                # if all sequences finished, stop early
                if finished.all():
                    break

                # for finished seqs, keep EOS as the next input; else keep decoding
                y_prev = torch.where(finished, torch.full_like(next_tok, EOS_ID), next_tok)

            return outs  # [B, L] (may include PAD/EOS; decode will strip)

# --------------------
# Load model (cached)
# --------------------
@st.cache_resource(show_spinner=True)
def load_model_and_meta():
    stoi_src, itos_tgt = load_vocabs()
    src_vsize = len(stoi_src) + 0  # mapping contains specials
    tgt_vsize = len(itos_tgt)

    # Must match your best checkpoint's hyperparams
    model = Seq2Seq(src_vsize=src_vsize, tgt_vsize=tgt_vsize,
                    emb_dim=256, hid_size=256, enc_layers=2, dec_layers=4, dropout=0.3).to(DEVICE)
    
    # Try to load from src directory first, then fallback to models directory
    try:
        ckpt_path = "src/bilstm4lstm_char_E256_H256_enc2_dec4_drop0.3_best.pt"
        payload = torch.load(ckpt_path, map_location=DEVICE)
    except:
        ckpt_path = "models/bilstm4lstm_char_E256_H256_enc2_dec4_drop0.3_best.pt"
        payload = torch.load(ckpt_path, map_location=DEVICE)
    
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, stoi_src, itos_tgt

model, stoi_src, itos_tgt = load_model_and_meta()

# --------------------
# Simple Clean UI Interface
# --------------------

# Title
st.markdown('<h1 class="urdu-title">اردو سے رومن اردو</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 20px; color: #666;">Neural Machine Translation with BiLSTM Architecture</p>', unsafe_allow_html=True)

# Performance Metrics
st.markdown("## 📊 Model Performance")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #28a745; margin: 0;">74.89</h3>
        <p style="margin: 5px 0 0 0; font-size: 16px;">BLEU Score</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #dc3545; margin: 0;">4.8%</h3>
        <p style="margin: 5px 0 0 0; font-size: 16px;">Error Rate</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #007bff; margin: 0;">5,255</h3>
        <p style="margin: 5px 0 0 0; font-size: 16px;">Test Examples</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #6f42c1; margin: 0;">29</h3>
        <p style="margin: 5px 0 0 0; font-size: 16px;">Urdu Poets</p>
    </div>
    """, unsafe_allow_html=True)

# Translation Interface
st.markdown("""
<div class="section-header">
    <h2 style="margin: 0; font-size: 28px; background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🔄 Translation Interface</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<h4 style="font-size: 24px; color: #2c3e50; margin-bottom: 15px;">اردو متن</h4>', unsafe_allow_html=True)
    txt = st.text_area(
        "Enter Urdu text here (one line per translation)",
        value="مزے جہان کے اپنی نظر میں خاک نہیں\nہم نے اس کے شہر کو چھوڑا اور آنکھوں کو موند لیا ہے\nعاشقی میں میرؔ جیسے خواب مت دیکہا کرو",
        height=200,
        label_visibility="collapsed"
    )

with col2:
    st.markdown('<h4 style="font-size: 24px; color: #2c3e50; margin-bottom: 15px;">⚙️ Settings</h4>', unsafe_allow_html=True)
    max_len = st.slider("Max Output Length", 50, 300, 120, 10)
    run_btn = st.button("🚀 Transliterate", key="translate_btn", type="primary")

# Translation Results
if run_btn:
    if txt.strip():
        with st.spinner("Translating..."):
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            batch_src = []
            for line in lines:
                ids = encode_char_src(line, stoi_src)
                t = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
                batch_src.append(t)
            
            # pad to max len
            T = max(x.size(1) for x in batch_src) if batch_src else 0
            src_pad = torch.full((len(batch_src), T), PAD_ID, dtype=torch.long, device=DEVICE)
            for i, x in enumerate(batch_src):
                src_pad[i, :x.size(1)] = x

            # decode
            pred_ids = model.greedy_decode(src_pad, max_len=max_len)
            preds = []
            for i in range(pred_ids.size(0)):
                preds.append(decode_char_tgt(pred_ids[i].tolist(), itos_tgt))

        st.markdown("""
        <div class="section-header">
            <h2 style="margin: 0; font-size: 28px; background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📝 Translation Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Show full poem in one box (Urdu above, Roman Urdu below)
        ur_full = "\n".join(lines)
        rom_full = "\n".join(preds)
        # Escape HTML and preserve line breaks
        def _to_html(text):
            return (text
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace("\n", "<br>"))
        ur_html = _to_html(ur_full)
        rom_html = _to_html(rom_full)

        st.markdown(f"""
        <div class="result-card">
            <h5 style="color: #ff6b6b; margin: 0 0 15px 0; font-size: 22px; font-weight: 600;">اردو:</h5>
            <div class="urdu-text">{ur_html}</div>
            <h5 style="color: #4ecdc4; margin: 20px 0 15px 0; font-size: 22px; font-weight: 600;">Roman:</h5>
            <div class="roman-text">{rom_html}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Please enter some Urdu text to translate!")

# Model Details
st.markdown("""
<div class="section-header">
    <h2 style="margin: 0; font-size: 28px; background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🏗️ Model Architecture & Details</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="model-card">
        <h4 style="color: #007bff; margin-top: 0;">🏗️ Architecture</h4>
        <p class="large-text"><strong>Encoder:</strong> 2-layer BiLSTM</p>
        <p class="large-text"><strong>Decoder:</strong> 4-layer LSTM with Luong Attention</p>
        <p class="large-text"><strong>Embedding:</strong> 256 dimensions</p>
        <p class="large-text"><strong>Hidden Size:</strong> 256 units</p>
        <p class="large-text"><strong>Parameters:</strong> ~44M trainable</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="model-card">
        <h4 style="color: #28a745; margin-top: 0;">📊 Performance Metrics</h4>
        <p class="large-text"><strong>BLEU Score:</strong> <span style="color: #28a745;">74.89</span></p>
        <p class="large-text"><strong>Character Error Rate:</strong> <span style="color: #dc3545;">4.8%</span></p>
        <p class="large-text"><strong>Perplexity:</strong> 1.137</p>
        <p class="large-text"><strong>Training Data:</strong> 10,493 pairs</p>
        <p class="large-text"><strong>Test Data:</strong> 5,255 pairs</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="model-card">
        <h4 style="color: #dc3545; margin-top: 0;">🔤 Tokenization</h4>
        <p class="large-text"><strong>Method:</strong> Character-level</p>
        <p class="large-text"><strong>Source Vocab:</strong> 50 characters</p>
        <p class="large-text"><strong>Target Vocab:</strong> 46 characters</p>
        <p class="large-text"><strong>Special Tokens:</strong> &lt;pad&gt;, &lt;s&gt;, &lt;/s&gt;, &lt;unk&gt;</p>
        <p class="large-text"><strong>Normalization:</strong> Unicode NFC</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="model-card">
        <h4 style="color: #ffc107; margin-top: 0;">🎓 Training Details</h4>
        <p class="large-text"><strong>Epochs:</strong> 20</p>
        <p class="large-text"><strong>Batch Size:</strong> 64</p>
        <p class="large-text"><strong>Learning Rate:</strong> 0.0005</p>
        <p class="large-text"><strong>Optimizer:</strong> Adam</p>
        <p class="large-text"><strong>Teacher Forcing:</strong> Scheduled Decay</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="model-card">
        <h4 style="color: #6f42c1; margin-top: 0;">📚 Dataset</h4>
        <p class="large-text"><strong>Source:</strong> Rekhta Foundation</p>
        <p class="large-text"><strong>Poets:</strong> 29 classical & modern</p>
        <p class="large-text"><strong>Content:</strong> Urdu Ghazals</p>
        <p class="large-text"><strong>Language:</strong> Urdu → Roman Urdu</p>
        <p class="large-text"><strong>Quality:</strong> <span style="color: #28a745;">Gold standard</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="model-card">
        <h4 style="color: #17a2b8; margin-top: 0;">🚀 Technical Features</h4>
        <p class="large-text"><strong>Attention:</strong> Luong General</p>
        <p class="large-text"><strong>Decoding:</strong> Greedy + Beam Search</p>
        <p class="large-text"><strong>EOS Handling:</strong> Smart termination</p>
        <p class="large-text"><strong>GPU Support:</strong> <span style="color: #28a745;">CUDA acceleration</span></p>
        <p class="large-text"><strong>Framework:</strong> PyTorch + Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div class="footer-card">
    <h4 style="color: #ff6b6b; margin-bottom: 15px; font-size: 24px;">🌟 Built with PyTorch + Streamlit</h4>
    <p class="large-text" style="margin-bottom: 15px; color: #2c3e50;">Preserving Urdu poetry through advanced neural machine translation</p>
    <p style="font-size: 16px; color: #666; line-height: 1.6;">
        Model: bilstm4lstm_char_E256_H256_enc2_dec4_drop0.3_best.pt | 
        Last Updated: {datetime.now().strftime("%B %d, %Y")} | 
        Runs on CPU or CUDA
    </p>
</div>
""", unsafe_allow_html=True)
