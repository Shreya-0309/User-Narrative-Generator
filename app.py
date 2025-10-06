# app.py
import streamlit as st
import pandas as pd
import torch
import pickle
import numpy as np
from tqdm import tqdm
import time
import traceback

# --- ML libs used by your pipeline ---
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------------
# Your model / helper class definitions (copy/paste from your script)
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
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
    def forward(self, fused_embeddings):
        batch_size = fused_embeddings.size(0)
        soft_prompts = self.soft_tokens.unsqueeze(1).expand(-1, batch_size, -1)
        fused_emb_exp = fused_embeddings.unsqueeze(0)
        for block in self.blocks:
            soft_prompts = block(soft_prompts, fused_emb_exp, fused_emb_exp)
        return soft_prompts

# ------------------------------
# Utility functions adapted from your script
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

    df['metadata'] = (df['title'] + " " + df['category'] + " Price: " +
                      df['price'].astype(str) + " Date: " +
                      df['purchase_date'].astype(str) + " Final Item: " +
                      df['final_item'].astype(str))

    def reorder_group(group):
        return group.sort_values(by=['final_item', 'purchase_date'], ascending=[False, False])

    df = df.groupby('reviewer_id', group_keys=False).apply(reorder_group).reset_index(drop=True)

    user_interactions = {}
    for _, row in df.iterrows():
        user_id = row['reviewer_id']
        asin = row['asin']
        try:
            rating = float(row['rating'])
        except Exception as e:
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

# tokenization & input prep (kept simple & aligned with your script)
def prepare_llm_inputs(soft_prompts, prompt_text, tokenizer, llm, device, max_length=512):
    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=max_length, truncation=True).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    # embed tokens using model's encoder embedding (works for HF seq2seq)
    if hasattr(llm, "get_encoder"):
        input_embeds = llm.get_encoder().embed_tokens(input_ids)
    else:
        # fallback
        input_embeds = llm.encoder.embed_tokens(input_ids)
    # soft_prompts: (num_soft_tokens, batch, d_model)
    # normalize shapes: make (batch, num_soft_tokens, d_model)
    soft_prompt_embeds = soft_prompts.squeeze(1).unsqueeze(0)
    # concat
    inputs_embeds = torch.cat([soft_prompt_embeds, input_embeds], dim=1)
    soft_prompt_attention = torch.ones((inputs_embeds.size(0), soft_prompt_embeds.size(1)), device=device)
    new_attention_mask = torch.cat([soft_prompt_attention, attention_mask], dim=1)
    return inputs_embeds, new_attention_mask

# ------------------------------
# CACHED LOADERS (use Streamlit caching)
# ------------------------------
@st.cache_data(show_spinner=False)
def load_dataframe(csv_path: str):
    df = pd.read_csv(csv_path)
    return df

@st.cache_resource(show_spinner=False)
def load_models_and_assets(item2idx_pkl='item2idx.pkl',
                           cf_encoder_path='trainable_cf_encoder.pth',
                           fusion_adapter_path='fusion_adapter_model.pth',
                           soft_prompt_path='soft_prompt_generator.pth',
                           llm_dir='fine_tuned_llm_peft',
                           tokenizer_name='google/flan-t5-large'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load item2idx
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
    # (Don't load state for item_encoder unless you have a saved checkpoint for it)

    fusion_adapter_local = AdvancedFusionAdapter(cf_dim=cf_dim, semantic_dim=semantic_dim,
                                         model_dim=1024, num_layers=2, num_heads=4, dropout=0.1).to(device)
    fusion_adapter_local.load_state_dict(torch.load(fusion_adapter_path, map_location=device))
    fusion_adapter_local.eval()

    soft_prompt_generator_local = SoftPromptGenerator(num_soft_tokens=20, d_model=1024, num_heads=4, num_layers=2, dropout=0.1).to(device)
    soft_prompt_generator_local.load_state_dict(torch.load(soft_prompt_path, map_location=device))
    soft_prompt_generator_local.eval()

    # Use AutoTokenizer / AutoModelForSeq2SeqLM so it's robust to T5 variants
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
# Fixed analytical prompt (same as your script)
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
# App UI
# ------------------------------
st.set_page_config(page_title="REGEN — Narrative Generator", layout="wide")

st.title("Customer Narrative Generator (REGEN)")
st.markdown(
    "Select a sample user from the dropdown, then click **Generate Narrative**. "
    "Models and heavy assets are cached — first load may take a while depending on your machine."
)

# Sidebar: asset locations (editable)
st.sidebar.header("Assets & Settings")
csv_path = st.sidebar.text_input("CSV path", value="REGEN_preprocessed_office.csv")
item2idx_pkl = st.sidebar.text_input("item2idx .pkl path", value="item2idx.pkl")
cf_encoder_path = st.sidebar.text_input("cf encoder .pth", value="trainable_cf_encoder.pth")
fusion_adapter_path = st.sidebar.text_input("fusion adapter .pth", value="fusion_adapter_model.pth")
soft_prompt_path = st.sidebar.text_input("soft prompt .pth", value="soft_prompt_generator.pth")
llm_dir = st.sidebar.text_input("LLM folder / HF repo", value="fine_tuned_llm_peft")
tokenizer_name = st.sidebar.text_input("tokenizer (HF name)", value="google/flan-t5-large")

st.sidebar.write("Device:", "GPU" if torch.cuda.is_available() else "CPU")

# Load dataframe
try:
    df = load_dataframe(csv_path)
except Exception as e:
    st.error(f"Failed to load CSV at '{csv_path}': {e}")
    st.stop()

# Split dataset like your script and get test subset
from sklearn.model_selection import train_test_split
df_train, df_temp = train_test_split(df, test_size=0.3, shuffle=False)
df_val, df_test = train_test_split(df_temp, test_size=0.5, shuffle=False)

# Preprocess test
try:
    user_interactions_test, _, _, df_test_proc = preprocess_dataset(df_test)
except Exception as e:
    st.error(f"Error during dataset preprocessing: {e}")
    st.stop()

# Load models & assets (cached)
with st.spinner("Loading models and assets (cached after first load)..."):
    try:
        assets = load_models_and_assets(item2idx_pkl=item2idx_pkl,
                                        cf_encoder_path=cf_encoder_path,
                                        fusion_adapter_path=fusion_adapter_path,
                                        soft_prompt_path=soft_prompt_path,
                                        llm_dir=llm_dir,
                                        tokenizer_name=tokenizer_name)
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

# User selection dropdown
st.subheader("Select a sample user")
sample_user_id = st.selectbox("Choose reviewer_id", options=sorted(list(user_interactions_test.keys())), index=0)

# show a sample of interactions for the chosen user
st.markdown("**Interaction preview (top 8 entries)**")
inter_preview = user_interactions_test[sample_user_id][:8]
for i, (asin, metadata, rating, review) in enumerate(inter_preview):
    st.markdown(f"- **{asin}** — Rating: {rating} — {metadata.split(' Date: ')[0]}")
    if i >= 7:
        break

# Controls
col1, col2 = st.columns([1, 1])
with col1:
    generate_btn = st.button("Generate Narrative")
with col2:
    show_reference = st.checkbox("Show reference narrative (if available)", value=True)

# Generation process
if generate_btn:
    try:
        t0 = time.time()
        st.info("Computing embeddings & soft prompts...")
        # Prepare inputs (similar to your code)
        interactions = user_interactions_test[sample_user_id]
        cf_emb_list = []
        texts = []
        for interaction in interactions:
            asin, metadata, rating, review = interaction
            texts.append(metadata + " " + review)
            if asin in item2idx:
                idx = item2idx[asin]
                idx_tensor = torch.tensor([idx], dtype=torch.long, device=device)
                cf_emb = cf_encoder(idx_tensor)  # returns (1, cf_dim)
            else:
                meta_emb = semantic_model.encode([metadata], convert_to_tensor=True, device=str(device))
                cf_emb = item_encoder(meta_emb)
            cf_emb_list.append(cf_emb)

        cf_emb_tensor = torch.cat(cf_emb_list, dim=0)  # (n_interactions, cf_dim)
        # semantic embeddings
        semantic_emb_tensor = semantic_model.encode(texts, convert_to_tensor=True, device=str(device))
        # run fusion
        fused_embeddings = fusion_adapter(cf_emb_tensor, semantic_emb_tensor)
        user_fused = fused_embeddings.mean(dim=0, keepdim=True)
        soft_prompts = soft_prompt_generator(user_fused)  # (num_soft_tokens, batch, d_model)
        # Prepare prompt
        interaction_summary = generate_interaction_summary(user_interactions_test, sample_user_id)
        full_prompt = fixed_prompt + "\n\n" + interaction_summary

        st.info("Calling LLM to generate narrative...")
        inputs_embeds, new_attention_mask = prepare_llm_inputs(soft_prompts, full_prompt, tokenizer, llm, device)
        max_len = soft_prompts.size(0) + 250
        generated_ids = llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            max_length=max_len,
            min_length=120,
            repetition_penalty=1.5,
            temperature=1.0,
            no_repeat_ngram_size=3,
            num_beams=5,
            num_return_sequences=1
        )
        narrative = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        if narrative and not narrative[0].isupper():
            narrative = narrative[0].upper() + narrative[1:]
        elapsed = time.time() - t0

        st.success("Narrative generated.")
        st.markdown("### Generated Narrative")
        st.write(narrative)

        # Optionally show reference
        if show_reference:
            # build reference map quickly
            def build_user_reference_map(df_proc):
                user_reference = {}
                for _, row in df_proc.iterrows():
                    user_id = row['reviewer_id']
                    long_summary = row.get('long_summary', "")
                    if user_id not in user_reference and isinstance(long_summary, str) and long_summary.strip():
                        user_reference[user_id] = long_summary
                return user_reference
            user_reference_map_test = build_user_reference_map(df_test_proc)
            ref = user_reference_map_test.get(sample_user_id, "").strip()
            if ref:
                st.markdown("### Reference Narrative")
                st.write(ref)
            else:
                st.info("No reference narrative available for this user.")

        # Show some internals (for debugging)
        with st.expander("Debug / internals"):
            st.write(f"Elapsed (sec): {elapsed:.2f}")
            st.write(f"Number interactions: {len(interactions)}")
            st.write(f"Soft prompt shape: {tuple(soft_prompts.shape)} (num_soft_tokens, batch, d_model)")
            st.write("Sample interaction summary (first 500 chars):")
            st.write(interaction_summary[:500])

    except Exception as e:
        st.error("Generation failed. See details below.")
        st.text(traceback.format_exc())
