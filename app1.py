# REGEN — Improved Streamlit UI (single-file)
# Save as: REGEN_streamlit_app.py
# Notes: This is a UX-first rewrite of your original app.py. Functionality is preserved
# but the UI is modernized: glass cards, gradients, two-panel layout, search/filter,
# advanced generation controls, download button, and neat debugging expander.

import streamlit as st
import pandas as pd
import torch
import pickle
import numpy as np
import time
import traceback
from tqdm import tqdm

# --- ML libs used by your pipeline ---
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------------
# Model classes (unchanged logic)
# ------------------------------
class TrainableCFEncoder(nn.Module):
    def __init__(self, num_items, cf_dim=128):
        super(TrainableCFEncoder, self).__init__()
        self.item_embeddings = nn.Embedding(num_items, cf_dim)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
    def forward(self, item_indices):
        return self.item_embeddings(item_indices)

class ItemEncoder(nn.Module):
    def __init__(self, input_dim, cf_dim):
        super(ItemEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, cf_dim)
        )
    def forward(self, x):
        return self.fc(x)

class AdvancedFusionAdapter(nn.Module):
    def __init__(self, cf_dim, semantic_dim, model_dim=1024, num_layers=2, num_heads=4, dropout=0.1):
        super(AdvancedFusionAdapter, self).__init__()
        self.input_proj_cf = nn.Linear(cf_dim, model_dim)
        self.input_proj_semantic = nn.Linear(semantic_dim, model_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
    def forward(self, cf_emb, semantic_emb):
        cf_emb = cf_emb.unsqueeze(1)
        semantic_emb = semantic_emb.unsqueeze(1)
        cf_proj = self.input_proj_cf(cf_emb)
        sem_proj = self.input_proj_semantic(semantic_emb)
        fused = cf_proj + sem_proj
        for layer in self.layers:
            fused = layer(fused)
        return fused.squeeze(1)

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, query, key, value):
        attn_output, _ = self.cross_attn(query, key, value)
        query = self.norm1(query + attn_output)
        ff_output = self.ff(query)
        out = self.norm2(query + ff_output)
        return out

class SoftPromptGenerator(nn.Module):
    def __init__(self, num_soft_tokens, d_model, num_heads, num_layers, dropout=0.1):
        super(SoftPromptGenerator, self).__init__()
        self.soft_tokens = nn.Parameter(torch.randn(num_soft_tokens, d_model))
        self.blocks = nn.ModuleList([ CrossAttentionBlock(d_model, num_heads, dropout) for _ in range(num_layers) ])
    def forward(self, fused_embeddings):
        batch_size = fused_embeddings.size(0)
        soft_prompts = self.soft_tokens.unsqueeze(1).expand(-1, batch_size, -1)
        fused_emb_exp = fused_embeddings.unsqueeze(0)
        for block in self.blocks:
            soft_prompts = block(soft_prompts, fused_emb_exp, fused_emb_exp)
        return soft_prompts

# ------------------------------
# Utility functions
# ------------------------------

def preprocess_dataset(df):
    expected_columns = ['title', 'category', 'price', 'reviewer_id', 'asin', 'rating', 'review', 'purchase_date']
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame is missing expected column: {col}")
    df = df.copy()
    df.fillna("", inplace=True)
    df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')
    df['final_item'] = False
    for user, group in df.groupby('reviewer_id'):
        max_date = group['purchase_date'].max()
        group_max = group[group['purchase_date'] == max_date]
        final_idx = group_max.index[-1]
        df.at[final_idx, 'final_item'] = True
    df['metadata'] = (df['title'] + " " + df['category'] + " Price: " + df['price'].astype(str) + " Date: " + df['purchase_date'].astype(str) + " Final Item: " + df['final_item'].astype(str))
    def reorder_group(group):
        return group.sort_values(by=['final_item', 'purchase_date'], ascending=[False, False])
    df = df.groupby('reviewer_id', group_keys=False).apply(reorder_group).reset_index(drop=True)
    user_interactions = {}
    for _, row in df.iterrows():
        user_id = row['reviewer_id']
        asin = row['asin']
        try:
            rating = float(row['rating'])
        except Exception:
            rating = 0.0
        review = row['review']
        metadata = row['metadata']
        interaction = (asin, metadata, rating, review)
        user_interactions.setdefault(user_id, []).append(interaction)
    user_ids = list(user_interactions.keys())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item_ids = df['asin'].unique().tolist()
    item2idx = {asin: i for i, asin in enumerate(item_ids)}
    return user_interactions, user2idx, item2idx, df


def generate_interaction_summary(user_interactions, user_id):
    interactions = user_interactions[user_id]
    summary_lines = []
    for asin, metadata, rating, review in interactions:
        summary_lines.append(f"Item: {asin}. Metadata: {metadata}. Rating: {rating}. Review: {review}.")
    return "\n".join(summary_lines)


def prepare_llm_inputs(soft_prompts, prompt_text, tokenizer, llm, device, max_length=512):
    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=max_length, truncation=True).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    if hasattr(llm, "get_encoder"):
        input_embeds = llm.get_encoder().embed_tokens(input_ids)
    else:
        input_embeds = llm.encoder.embed_tokens(input_ids)
    soft_prompt_embeds = soft_prompts.squeeze(1).unsqueeze(0)
    inputs_embeds = torch.cat([soft_prompt_embeds, input_embeds], dim=1)
    soft_prompt_attention = torch.ones((inputs_embeds.size(0), soft_prompt_embeds.size(1)), device=device)
    new_attention_mask = torch.cat([soft_prompt_attention, attention_mask], dim=1)
    return inputs_embeds, new_attention_mask

# ------------------------------
# CACHED LOADERS
# ------------------------------
@st.cache_data(show_spinner=False)
def load_dataframe(csv_path: str):
    df = pd.read_csv(csv_path)
    return df

@st.cache_resource(show_spinner=False)
def load_models_and_assets(item2idx_pkl='item2idx.pkl', cf_encoder_path='trainable_cf_encoder.pth', fusion_adapter_path='fusion_adapter_model.pth', soft_prompt_path='soft_prompt_generator.pth', llm_dir='fine_tuned_llm_peft', tokenizer_name='google/flan-t5-large'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(item2idx_pkl, 'rb') as f:
        item2idx_local = pickle.load(f)
    cf_dim = 128
    num_items = len(item2idx_local)
    cf_encoder_local = TrainableCFEncoder(num_items, cf_dim=cf_dim).to(device)
    cf_encoder_local.load_state_dict(torch.load(cf_encoder_path, map_location=device))
    cf_encoder_local.eval()
    semantic_model_local = SentenceTransformer('all-MiniLM-L6-v2', device=str(device))
    semantic_dim = semantic_model_local.get_sentence_embedding_dimension()
    item_encoder_local = ItemEncoder(semantic_dim, cf_dim).to(device)
    fusion_adapter_local = AdvancedFusionAdapter(cf_dim=cf_dim, semantic_dim=semantic_dim, model_dim=1024, num_layers=2, num_heads=4, dropout=0.1).to(device)
    fusion_adapter_local.load_state_dict(torch.load(fusion_adapter_path, map_location=device))
    fusion_adapter_local.eval()
    soft_prompt_generator_local = SoftPromptGenerator(num_soft_tokens=20, d_model=1024, num_heads=4, num_layers=2, dropout=0.1).to(device)
    soft_prompt_generator_local.load_state_dict(torch.load(soft_prompt_path, map_location=device))
    soft_prompt_generator_local.eval()
    tokenizer_local = AutoTokenizer.from_pretrained(tokenizer_name)
    llm_local = AutoModelForSeq2SeqLM.from_pretrained(llm_dir).to(device)
    llm_local.eval()
    return {
        "device": device,
        "item2idx": item2idx_local,
        "cf_encoder": cf_encoder_local,
        "semantic_model": semantic_model_local,
        "item_encoder": item_encoder_local,
        "fusion_adapter": fusion_adapter_local,
        "soft_prompt_generator": soft_prompt_generator_local,
        "tokenizer": tokenizer_local,
        "llm": llm_local
    }

# ------------------------------
# Fixed prompt
# ------------------------------
fixed_prompt = (
    "As a customer analyst, synthesize a user narrative in about 200 words from this customer's purchase history, reviews, and interaction data. "
    "Focus on identifying statistically significant patterns and trends that reveal their core preferences—especially in colors, brands, sizes, and styles—while also capturing broader behavioral signals across the purchase history. \n"
    "Your narrative should include:\n"
    " - Group related products to expose recurring patterns and underlying purposes of purchases\n"
    " - Highlight seasonal or cyclical buying behaviors\n"
    " - Analyze purchase frequency, order sizes, and brand loyalties\n"
    " - Detail category preferences, including any product avoidances\n"
    " - Evaluate price range preferences\n"
    " - Pinpoint preferences in color, size, and style across different categories\n"
    " - Extract common themes, including satisfaction drivers and pain points\n"
    "\nDeliverable Format:\n"
    "- Present your analysis as a single, focused paragraph.\n"
    "- Emphasize overarching patterns and insights rather than isolated examples.\n"
    "- Use specific examples only when they clearly illustrate significant trends (e.g., key colors, brands, sizes).\n"
    "- Conclude with the customer's overall profile.\n"
    "\nGuidelines:\n"
    "- Cover all purchase history.\n"
    "- Maintain analytical objectivity and clarity.\n"
    "- Avoid extraneous commentary that is not part of the narrative."
)

# ------------------------------
# UX / Visuals
# ------------------------------
st.set_page_config(page_title="REGEN — Narrative Generator", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for glassy modern UI
st.markdown(
    "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap' rel='stylesheet'>",
    unsafe_allow_html=True,
)

CUSTOM_CSS = '''
<style>
:root{--accent:#7c3aed;--accent2:#06b6d4}
body {font-family: Inter, sans-serif}
.header {
  background: linear-gradient(90deg, rgba(124,58,237,0.95), rgba(6,182,212,0.9));
  padding: 28px 40px; border-radius: 12px; color: white; margin-bottom: 18px;
}
.header h1{margin:0; font-weight:800}
.header p{margin:0; opacity:0.95}
.card {background: rgba(255,255,255,0.06); border-radius: 12px; padding: 18px; box-shadow: 0 8px 30px rgba(2,6,23,0.35); border: 1px solid rgba(255,255,255,0.04);}
.small-muted{color:rgba(255,255,255,0.7); font-size:13px}
.user-card{background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:12px; border-radius:10px}
.btn-primary{background:linear-gradient(90deg,var(--accent),var(--accent2)); color:white; padding:8px 18px; border-radius:8px}
.metadata{font-size:13px; color:rgba(255,255,255,0.9)}
.code{background:#0b1220; padding:8px; border-radius:8px}
</style>
'''
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Top header
st.markdown('<div class="header"><h1>Data-to-Narrative — Customer Narrative Studio</h1><p>Generate concise, insight-rich user narratives from purchase history and reviews. Export-ready output and advanced controls.</p></div>', unsafe_allow_html=True)

# Sidebar: asset paths & quick settings
with st.sidebar:
    st.markdown("## ⚙️ Assets & Settings")
    csv_path = st.text_input("CSV path", value="REGEN_preprocessed_office.csv")
    item2idx_pkl = st.text_input("item2idx .pkl path", value="item2idx.pkl")
    cf_encoder_path = st.text_input("cf encoder .pth", value="trainable_cf_encoder.pth")
    fusion_adapter_path = st.text_input("fusion adapter .pth", value="fusion_adapter_model.pth")
    soft_prompt_path = st.text_input("soft prompt .pth", value="soft_prompt_generator.pth")
    llm_dir = st.text_input("LLM folder / HF repo", value="fine_tuned_llm_peft")
    tokenizer_name = st.text_input("tokenizer (HF name)", value="google/flan-t5-large")
    st.divider()
    st.markdown("**Computation device**:  ``{}``".format("GPU" if torch.cuda.is_available() else "CPU"))
    st.markdown("---")
    st.markdown("### Quick UX tweaks")
    compact_mode = st.checkbox("Compact UI (less spacing)", value=False)
    show_refs = st.checkbox("Show reference narrative when available", value=True)
    st.markdown("---")
    st.markdown("Made with ❤️ — REGEN UI")

# Load dataframe
try:
    df = load_dataframe(csv_path)
except Exception as e:
    st.error(f"Failed to load CSV at '{csv_path}': {e}")
    st.stop()

# Split dataset and prepare test set
from sklearn.model_selection import train_test_split

if df.shape[0] < 2:
    st.error("Loaded CSV seems too small or empty. Provide a valid REGEN CSV.")
    st.stop()

df_train, df_temp = train_test_split(df, test_size=0.3, shuffle=False)
df_val, df_test = train_test_split(df_temp, test_size=0.5, shuffle=False)

try:
    user_interactions_test, _, _, df_test_proc = preprocess_dataset(df_test)
except Exception as e:
    st.error(f"Error during dataset preprocessing: {e}")
    st.stop()

# Load models and assets (cached)
with st.spinner("Loading models and assets — this may take a while (cached)..."):
    try:
        assets = load_models_and_assets(item2idx_pkl=item2idx_pkl, cf_encoder_path=cf_encoder_path, fusion_adapter_path=fusion_adapter_path, soft_prompt_path=soft_prompt_path, llm_dir=llm_dir, tokenizer_name=tokenizer_name)
    except Exception as e:
        st.error(f"Failed to load models/assets: {e}")
        st.stop()

device = assets["device"]
item2idx = assets["item2idx"]
cf_encoder = assets["cf_encoder"]
semantic_model = assets["semantic_model"]
item_encoder = assets["item_encoder"]
fusion_adapter = assets["fusion_adapter"]
soft_prompt_generator = assets["soft_prompt_generator"]
tokenizer = assets["tokenizer"]
llm = assets["llm"]

# Main layout: left = selector, right = generator/result
left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Find a user")
    search_q = st.text_input("Search reviewer_id (or paste one)")
    all_users = sorted(list(user_interactions_test.keys()))
    if search_q:
        filtered = [u for u in all_users if search_q.lower() in str(u).lower()]
        if not filtered:
            st.warning("No users matched your search — try a different query.")
            filtered = all_users[:200]
    else:
        filtered = all_users[:200]
    sample_user_id = st.selectbox("Choose reviewer_id", options=filtered, index=0)
    st.markdown("---")
    st.markdown("**Interaction preview (top 8)**")
    inter_preview = user_interactions_test[sample_user_id][:8]
    for i, (asin, metadata, rating, review) in enumerate(inter_preview):
        st.markdown(f"<div class='user-card'><b>{asin}</b> — <span class='small-muted'>Rating: {rating}</span><div class='metadata'>{metadata.split(' Date: ')[0]}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("\n")
    st.markdown("<div class='card'><b>Generation controls</b>", unsafe_allow_html=True)
    max_tokens = st.slider("Max generated length", min_value=80, max_value=600, value=250, step=10)
    min_tokens = st.slider("Min generated length", min_value=40, max_value=200, value=120, step=10)
    beams = st.slider("Beam size", min_value=1, max_value=8, value=5)
    temp = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    rep_penalty = st.slider("Repetition penalty", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
    st.markdown("</div>", unsafe_allow_html=True)

    gen_button = st.button("✨ Generate Narrative")

with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Result preview")
    result_area = st.empty()
    download_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# Helper to build reference map quickly
def build_user_reference_map(df_proc):
    user_reference = {}
    for _, row in df_proc.iterrows():
        user_id = row['reviewer_id']
        long_summary = row.get('long_summary', "")
        if user_id not in user_reference and isinstance(long_summary, str) and long_summary.strip():
            user_reference[user_id] = long_summary
    return user_reference

user_reference_map_test = build_user_reference_map(df_test_proc)

# Generation logic triggered by button
if gen_button:
    try:
        t0 = time.time()
        st.toast("Starting generation — steps will be shown below")
        # Step 1: prepare interaction texts and cf embeddings
        with st.spinner("Computing embeddings & soft prompts..."):
            interactions = user_interactions_test[sample_user_id]
            cf_emb_list = []
            texts = []
            for interaction in interactions:
                asin, metadata, rating, review = interaction
                texts.append(metadata + " " + review)
                if asin in item2idx:
                    idx = item2idx[asin]
                    idx_tensor = torch.tensor([idx], dtype=torch.long, device=device)
                    cf_emb = cf_encoder(idx_tensor)
                else:
                    meta_emb = semantic_model.encode([metadata], convert_to_tensor=True, device=str(device))
                    cf_emb = item_encoder(meta_emb)
                cf_emb_list.append(cf_emb)
            cf_emb_tensor = torch.cat(cf_emb_list, dim=0)
            semantic_emb_tensor = semantic_model.encode(texts, convert_to_tensor=True, device=str(device))
            fused_embeddings = fusion_adapter(cf_emb_tensor, semantic_emb_tensor)
            user_fused = fused_embeddings.mean(dim=0, keepdim=True)
            soft_prompts = soft_prompt_generator(user_fused)

        # Step 2: prepare prompt
        interaction_summary = generate_interaction_summary(user_interactions_test, sample_user_id)
        full_prompt = fixed_prompt + "\n\n" + interaction_summary

        # Step 3: call LLM
        with st.spinner("Calling LLM to generate narrative..."):
            inputs_embeds, new_attention_mask = prepare_llm_inputs(soft_prompts, full_prompt, tokenizer, llm, device)
            max_len = soft_prompts.size(0) + max_tokens
            generated_ids = llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=new_attention_mask,
                max_length=max_len,
                min_length=min_tokens,
                repetition_penalty=rep_penalty,
                temperature=temp,
                no_repeat_ngram_size=3,
                num_beams=beams,
                num_return_sequences=1
            )
            narrative = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
            if narrative and not narrative[0].isupper():
                narrative = narrative[0].upper() + narrative[1:]

        elapsed = time.time() - t0
        #result_area.markdown(f"**Generated ({elapsed:.1f}s)**\n\n{narrative}")
        result_area.markdown(f"Generated Narrative:\n\n{narrative}")
        #(f"**Generated ({elapsed:.1f}s)**\n\n{narrative}")


        # Download button
        download_placeholder.download_button("Download narrative (TXT)", data=narrative, file_name=f"narrative_{sample_user_id}.txt")

        # Show ref if configured
        if show_refs:
            ref = user_reference_map_test.get(sample_user_id, "").strip()
            if ref:
                st.markdown("---")
                st.markdown("### Reference Narrative (from dataset)")
                st.write(ref)

        # Debug expander
        with st.expander("Debug / internals"):
            st.write(f"Elapsed (sec): {elapsed:.2f}")
            st.write(f"Number interactions: {len(interactions)}")
            st.write(f"Soft prompt shape: {tuple(soft_prompts.shape)} (num_soft_tokens, batch, d_model)")
            st.write("Sample interaction summary (first 500 chars):")
            st.write(interaction_summary[:500])

    except Exception as e:
        st.error("Generation failed. See details below.")
        st.text(traceback.format_exc())

# Footer with tips
st.markdown("<div style='margin-top:18px; color: #9aa8b2'>Tip: You can tweak generation hyperparameters in the left panel for shorter, more creative, or more conservative outputs.</div>", unsafe_allow_html=True)

# End of file
