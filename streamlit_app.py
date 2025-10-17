import streamlit as st
import torch
import torch.nn.functional as F
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import re
import math
import time
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Empathetic Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
        color: #333;
        font-size: 14px;
        line-height: 1.5;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left-color: #1976D2;
        color: #0D47A1;
    }
    .bot-message {
        background-color: #F3E5F5;
        border-left-color: #7B1FA2;
        color: #4A148C;
    }
    .emotion-tag {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .metrics-box {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vocab' not in st.session_state:
    st.session_state.vocab = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Emotion categories
EMOTIONS = [
    "afraid", "angry", "annoyed", "anticipating", "anxious", "apprehensive", "ashamed",
    "caring", "confident", "content", "devastated", "disappointed", "disgusted", "embarrassed",
    "excited", "faithful", "furious", "grateful", "guilty", "hopeful", "impressed", "jealous",
    "joyful", "lonely", "nostalgic", "prepared", "proud", "sad", "sentimental", "surprised",
    "terrified", "trusting"
]

# Load model components (copied from your notebook)
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
        # Add special tokens
        self.PAD_TOKEN = '<PAD>'
        self.BOS_TOKEN = '<BOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'
        
        self.add_word(self.PAD_TOKEN)
        self.add_word(self.BOS_TOKEN)  
        self.add_word(self.EOS_TOKEN)
        self.add_word(self.UNK_TOKEN)
        
        self.PAD_IDX = 0
        self.BOS_IDX = 1
        self.EOS_IDX = 2
        self.UNK_IDX = 3
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def get_idx(self, word):
        return self.word2idx.get(word, self.UNK_IDX)
    
    def get_word(self, idx):
        return self.idx2word.get(idx, self.UNK_TOKEN)
    
    def __len__(self):
        return len(self.word2idx)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        return self.W_o(attn_output), attn_weights

class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.linear2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x, attn_weights

class TransformerEncoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        attn_weights = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attn_weights.append(attn)
        return x, attn_weights

class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        cross_attn_output, cross_attn_weights = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, self_attn_weights, cross_attn_weights

class TransformerDecoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        self_attn_weights = []
        cross_attn_weights = []
        
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, enc_output, src_mask, tgt_mask)
            self_attn_weights.append(self_attn)
            cross_attn_weights.append(cross_attn)
            
        return x, self_attn_weights, cross_attn_weights

class Transformer(torch.nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_encoder_layers=6, 
                 num_decoder_layers=6, d_ff=2048, max_seq_len=1000, dropout=0.1, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        self.src_embedding = torch.nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = torch.nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        
        # Use fc_out to match the trained model
        self.fc_out = torch.nn.Linear(d_model, vocab_size)
        self.dropout = torch.nn.Dropout(dropout)
        
    def create_padding_mask(self, x):
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
    
    def create_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0
        
    def forward(self, src, tgt):
        src_mask = self.create_padding_mask(src)
        tgt_mask = self.create_padding_mask(tgt)
        
        look_ahead_mask = self.create_look_ahead_mask(tgt.size(1)).to(tgt.device)
        combined_mask = tgt_mask & look_ahead_mask.unsqueeze(0).unsqueeze(0)
        
        src_emb = self.dropout(self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model)))
        tgt_emb = self.dropout(self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))
        
        enc_output, enc_attn_weights = self.encoder(src_emb, src_mask)
        dec_output, dec_self_attn, dec_cross_attn = self.decoder(tgt_emb, enc_output, src_mask, combined_mask)
        
        output = self.fc_out(dec_output)
        
        return output, {
            'encoder_attention': enc_attn_weights,
            'decoder_self_attention': dec_self_attn,
            'decoder_cross_attention': dec_cross_attn
        }

def normalize_text(text):
    """Normalize text similar to preprocessing"""
    text = text.lower()
    text = re.sub(r'[^\w\s\.\!\?\,]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    # Add spaces around punctuation
    text = re.sub(r'([.!?,:;])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_text(text, vocab):
    """Convert text to token indices"""
    normalized = normalize_text(text)
    tokens = normalized.split()
    indices = [vocab.get_idx(token) for token in tokens]
    return indices

def create_input_sequence(emotion, situation, customer_utterance, vocab):
    """Create formatted input sequence"""
    input_text = f"emotion : {emotion} | situation : {situation} | customer : {customer_utterance} agent :"
    return tokenize_text(input_text, vocab)

def ensure_vocab_compatibility(vocab):
    """Ensure vocabulary has all required attributes for compatibility"""
    # Add missing attributes if they don't exist
    if not hasattr(vocab, 'PAD_TOKEN'):
        vocab.PAD_TOKEN = '<PAD>'
    if not hasattr(vocab, 'BOS_TOKEN'):
        vocab.BOS_TOKEN = '<BOS>'
    if not hasattr(vocab, 'EOS_TOKEN'):
        vocab.EOS_TOKEN = '<EOS>'
    if not hasattr(vocab, 'UNK_TOKEN'):
        vocab.UNK_TOKEN = '<UNK>'
    
    # Add missing indices if they don't exist
    if not hasattr(vocab, 'PAD_IDX'):
        vocab.PAD_IDX = vocab.word2idx.get('<PAD>', 0)
    if not hasattr(vocab, 'BOS_IDX'):
        vocab.BOS_IDX = vocab.word2idx.get('<BOS>', 1)
    if not hasattr(vocab, 'EOS_IDX'):
        vocab.EOS_IDX = vocab.word2idx.get('<EOS>', 2)
    if not hasattr(vocab, 'UNK_IDX'):
        vocab.UNK_IDX = vocab.word2idx.get('<UNK>', 3)
    
    # Ensure get_idx method exists
    if not hasattr(vocab, 'get_idx'):
        def get_idx(word):
            return vocab.word2idx.get(word, vocab.UNK_IDX)
        vocab.get_idx = get_idx
    
    # Ensure get_word method exists
    if not hasattr(vocab, 'get_word'):
        def get_word(idx):
            return vocab.idx2word.get(idx, vocab.UNK_TOKEN)
        vocab.get_word = get_word
    
    return vocab

def download_model_from_github():
    """Download model from GitHub releases if not present"""
    model_path = 'saved_models/best_model.pkl'
    
    if not os.path.exists(model_path):
        st.info("📥 Downloading model from GitHub (first time only, ~545MB)...")
        
        # Replace with your actual release URL after uploading to GitHub releases
        model_url = "https://github.com/Hasnain-rdj/Empathetic_Chatbot_Project/releases/download/v1.0.0/best_model.pkl"
        
        # Create directory if it doesn't exist
        os.makedirs('saved_models', exist_ok=True)
        
        try:
            import requests
            with st.spinner("Downloading model... Please wait."):
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                progress_bar = st.progress(0)
                
                with open(model_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
                
                st.success("✅ Model downloaded successfully!")
                
        except Exception as e:
            st.error(f"❌ Error downloading model: {str(e)}")
            st.error("Please download manually from: https://github.com/Hasnain-rdj/Empathetic_Chatbot_Project/releases")
            return False
    
    return True

@st.cache_resource
def load_model_and_vocab():
    """Load the trained model and vocabulary"""
    try:
        # Download model if not present
        if not download_model_from_github():
            return None, None, "Failed to download model from GitHub releases."
        
        # Try different possible paths
        possible_paths = [
            'saved_models/best_model.pkl',
            './saved_models/best_model.pkl',
            'best_model.pkl'
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            return None, None, "Model file not found. Please ensure 'best_model.pkl' exists."
        
        # Load checkpoint - compatible with PyTorch 2.0+
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Load vocabulary from checkpoint
        if 'vocab' in checkpoint and checkpoint['vocab'] is not None:
            vocab = checkpoint['vocab']
        else:
            # Try to load from separate vocab file
            vocab_path = 'saved_vocab/vocabulary.pkl'
            if os.path.exists(vocab_path):
                with open(vocab_path, 'rb') as f:
                    vocab = pickle.load(f)
            else:
                return None, None, "Vocabulary not found in checkpoint or separate file."
        
        # Ensure vocabulary compatibility
        vocab = ensure_vocab_compatibility(vocab)
        
        # Get model config - use exact config from checkpoint
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config'].copy()
            # Update vocab size to match actual vocabulary
            model_config['vocab_size'] = len(vocab)
        else:
            # Fallback config based on inspection
            model_config = {
                'vocab_size': len(vocab),
                'd_model': 512,
                'num_heads': 2,
                'num_encoder_layers': 2,
                'num_decoder_layers': 2,
                'd_ff': 2048,
                'max_seq_len': 1000,
                'dropout': 0.1,  # Use 0.1 as seen in saved config
                'pad_idx': 0
            }
        
        # Create model
        model = Transformer(**model_config)
        
        # Load model state dict with error handling
        if 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            except RuntimeError as e:
                # If strict loading fails, try with strict=False and provide info
                if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                    st.warning(f"Model architecture mismatch detected. Attempting flexible loading...")
                    try:
                        # Try loading with strict=False
                        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        if missing_keys:
                            st.warning(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
                        if unexpected_keys:
                            st.warning(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
                        
                        # If critical keys are missing, show error
                        critical_keys = ['src_embedding.weight', 'tgt_embedding.weight']
                        missing_critical = [k for k in missing_keys if any(ck in k for ck in critical_keys)]
                        if missing_critical:
                            return None, None, f"Critical model components missing: {missing_critical}"
                            
                    except Exception as e2:
                        return None, None, f"Failed to load model weights: {str(e2)}"
                else:
                    return None, None, f"Model loading error: {str(e)}"
        else:
            return None, None, "Model state dict not found in checkpoint"
        
        model.eval()
        
        return model, vocab, None
        
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

def generate_response_greedy(model, input_indices, vocab, device, max_length=50):
    """Generate response using greedy decoding"""
    model.eval()
    
    with torch.no_grad():
        # Prepare input
        src = torch.tensor([input_indices], device=device)
        
        # Start with BOS token
        tgt = torch.tensor([[vocab.BOS_IDX]], device=device)
        
        attention_weights = None
        
        for _ in range(max_length):
            # Forward pass
            output, attn_dict = model(src, tgt)
            attention_weights = attn_dict
            
            # Get next token
            next_token_logits = output[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            
            # Stop if EOS token
            if next_token == vocab.EOS_IDX:
                break
                
            # Add token to sequence
            tgt = torch.cat([tgt, torch.tensor([[next_token]], device=device)], dim=1)
        
        # Convert to text
        generated_tokens = tgt[0][1:].tolist()  # Remove BOS token
        response_tokens = []
        
        for idx in generated_tokens:
            if idx == vocab.EOS_IDX:
                break
            word = vocab.get_word(idx)
            if word not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                response_tokens.append(word)
        
        return ' '.join(response_tokens), attention_weights

def generate_response_beam_search(model, input_indices, vocab, device, beam_width=3, max_length=50):
    """Generate response using beam search"""
    model.eval()
    
    with torch.no_grad():
        src = torch.tensor([input_indices], device=device)
        
        # Initialize beams
        beams = [([vocab.BOS_IDX], 0.0)]  # (sequence, score)
        
        for step in range(max_length):
            candidates = []
            
            for sequence, score in beams:
                if sequence[-1] == vocab.EOS_IDX:
                    candidates.append((sequence, score))
                    continue
                
                tgt = torch.tensor([sequence], device=device)
                output, _ = model(src, tgt)
                
                # Get probabilities for next token
                next_token_probs = F.softmax(output[0, -1, :], dim=0)
                
                # Get top-k tokens
                top_probs, top_indices = torch.topk(next_token_probs, beam_width)
                
                for prob, idx in zip(top_probs, top_indices):
                    new_sequence = sequence + [idx.item()]
                    new_score = score + torch.log(prob).item()
                    candidates.append((new_sequence, new_score))
            
            # Select top beams
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]
            
            # Check if all beams ended
            if all(seq[-1] == vocab.EOS_IDX for seq, _ in beams):
                break
        
        # Get best sequence
        best_sequence = beams[0][0][1:]  # Remove BOS token
        
        # Convert to text
        response_tokens = []
        for idx in best_sequence:
            if idx == vocab.EOS_IDX:
                break
            word = vocab.get_word(idx)
            if word not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                response_tokens.append(word)
        
        return ' '.join(response_tokens), None

def plot_attention_heatmap(attention_weights, input_tokens, output_tokens, layer_idx=0, head_idx=0):
    """Create attention heatmap visualization"""
    try:
        # Get attention weights for specified layer and head
        if 'decoder_cross_attention' in attention_weights:
            attn = attention_weights['decoder_cross_attention'][layer_idx][0, head_idx].cpu().numpy()
        else:
            return None
        
        # Limit tokens for visualization
        max_tokens = 20
        input_tokens = input_tokens[:max_tokens]
        output_tokens = output_tokens[:max_tokens]
        
        # Crop attention matrix
        attn = attn[:len(output_tokens), :len(input_tokens)]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use a custom colormap
        cmap = LinearSegmentedColormap.from_list("custom", ["white", "#1E88E5"])
        
        im = ax.imshow(attn, cmap=cmap, aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(input_tokens)))
        ax.set_yticks(range(len(output_tokens)))
        ax.set_xticklabels(input_tokens, rotation=45, ha='right')
        ax.set_yticklabels(output_tokens)
        
        # Labels and title
        ax.set_xlabel('Input Tokens')
        ax.set_ylabel('Output Tokens')
        ax.set_title(f'Cross-Attention Heatmap (Layer {layer_idx}, Head {head_idx})')
        
        # Colorbar
        plt.colorbar(im, ax=ax)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating attention heatmap: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">🤖 Empathetic Chatbot</div>', unsafe_allow_html=True)
    
    # Model info
    if not os.path.exists('saved_models/best_model.pkl'):
        st.info("ℹ️ **First-time setup**: The trained model (~545MB) will be automatically downloaded from GitHub releases on first run.")
    
    # Auto-load model on startup
    if not st.session_state.model_loaded:
        with st.spinner("🔄 Loading model and vocabulary..."):
            model, vocab, error = load_model_and_vocab()
            
            if error:
                st.error(f"❌ {error}")
                st.stop()  # Stop execution if model loading fails
            else:
                st.session_state.model = model
                st.session_state.vocab = vocab
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
                time.sleep(1)  # Brief pause to show success message
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model status
        st.success("✅ Model loaded and ready")
        
        if st.button("🔄 Reload Model"):
            st.session_state.model_loaded = False
            st.session_state.model = None
            st.session_state.vocab = None
            st.rerun()
        
        # Decoding strategy
        st.subheader("🎯 Decoding Strategy")
        decoding_method = st.selectbox(
            "Choose method:",
            ["Greedy Search", "Beam Search"],
            key="decoding_method"
        )
        
        if decoding_method == "Beam Search":
            beam_width = st.slider("Beam Width", 2, 5, 3)
        else:
            beam_width = 1
        
        # Generation parameters
        st.subheader("🎛️ Generation Parameters")
        max_length = st.slider("Max Response Length", 10, 100, 50)
        
        # Visualization options
        st.subheader("📊 Visualization")
        show_attention = st.checkbox("Show Attention Heatmap", value=False)
        
        if show_attention:
            layer_idx = st.selectbox("Attention Layer", [0, 1], key="layer_idx")
            head_idx = st.selectbox("Attention Head", [0, 1], key="head_idx")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()
    
    # Main interface - Model is guaranteed to be loaded at this point
    
    # Input section
    st.header("💬 Chat Interface")
    
    # Use session state to track if we need to clear input
    if 'clear_input' not in st.session_state:
        st.session_state.clear_input = False
    
    # Input form
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_input = st.text_area(
                "Enter your message:",
                placeholder="Type your message here...",
                height=100,
                key="user_input_form"
            )
        
        with col2:
            st.subheader("😊 Emotion (Optional)")
            selected_emotion = st.selectbox(
                "Choose emotion:",
                ["auto-detect"] + EMOTIONS,
                key="emotion_select_form"
            )
            
            situation = st.text_input(
                "Situation (Optional):",
                placeholder="Describe the situation...",
                key="situation_input_form"
            )
        
        # Generate button (submit button for form)
        submitted = st.form_submit_button("🚀 Generate Response", type="primary")
    
    # Handle form submission
    if submitted and user_input.strip():
        if not st.session_state.model or not st.session_state.vocab:
            st.error("Model not loaded!")
            return
        
        with st.spinner("Generating response..."):
            try:
                # Prepare inputs
                emotion = selected_emotion if selected_emotion != "auto-detect" else "caring"
                situation_text = situation if situation.strip() else user_input
                
                # Create input sequence
                input_indices = create_input_sequence(
                    emotion, situation_text, user_input, st.session_state.vocab
                )
                
                # Generate response
                start_time = time.time()
                
                if decoding_method == "Beam Search":
                    response, attention_weights = generate_response_beam_search(
                        st.session_state.model, input_indices, st.session_state.vocab,
                        st.session_state.device, beam_width, max_length
                    )
                else:
                    response, attention_weights = generate_response_greedy(
                        st.session_state.model, input_indices, st.session_state.vocab,
                        st.session_state.device, max_length
                    )
                
                generation_time = time.time() - start_time
                
                # Add to conversation history
                st.session_state.conversation_history.append({
                    'timestamp': datetime.now(),
                    'user_input': user_input,
                    'emotion': emotion,
                    'situation': situation_text,
                    'bot_response': response,
                    'generation_time': generation_time,
                    'decoding_method': decoding_method,
                    'attention_weights': attention_weights
                })
                
                # Success message - input will be cleared on rerun
                st.success("✅ Response generated successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    
    # Display conversation
    st.header("💭 Conversation History")
    
    if st.session_state.conversation_history:
        for i, conv in enumerate(reversed(st.session_state.conversation_history)):
            with st.expander(f"💬 Conversation {len(st.session_state.conversation_history) - i}", expanded=(i == 0)):
                # User message
                import html
                user_text = html.escape(conv['user_input'])
                st.markdown(
                    f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong> {user_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Emotion and situation tags
                if conv['emotion'] or conv['situation']:
                    tags_html = ""
                    if conv['emotion']:
                        tags_html += f'<span class="emotion-tag" style="background-color: #E1F5FE; color: #01579B;">😊 {conv["emotion"]}</span>'
                    if conv['situation'] and conv['situation'] != conv['user_input']:
                        tags_html += f'<span class="emotion-tag" style="background-color: #FFF3E0; color: #E65100;">📝 Situation</span>'
                    
                    st.markdown(tags_html, unsafe_allow_html=True)
                
                # Bot response
                bot_text = html.escape(conv['bot_response'])
                st.markdown(
                    f"""
                    <div class="chat-message bot-message">
                        <strong>🤖 Empathetic Bot:</strong> {bot_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"⏱️ Generated in {conv['generation_time']:.2f}s")
                with col2:
                    st.caption(f"🔍 Method: {conv['decoding_method']}")
                with col3:
                    st.caption(f"🕐 {conv['timestamp'].strftime('%H:%M:%S')}")
                
                # Attention visualization
                if (show_attention and conv['attention_weights'] and 
                    conv['decoding_method'] == "Greedy Search"):
                    
                    st.subheader("🔍 Attention Heatmap")
                    
                    # Prepare tokens for visualization
                    input_tokens = normalize_text(
                        f"emotion : {conv['emotion']} | situation : {conv['situation']} | customer : {conv['user_input']} agent :"
                    ).split()
                    output_tokens = conv['bot_response'].split()
                    
                    fig = plot_attention_heatmap(
                        conv['attention_weights'], input_tokens, output_tokens,
                        layer_idx, head_idx
                    )
                    
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("Attention visualization not available for this response.")
    else:
        st.info("No conversations yet. Start by entering a message above! 👆")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <p>🤖 Empathetic Chatbot | Built with Streamlit & PyTorch Transformer</p>
            <p>💡 This chatbot uses a custom-trained transformer model to provide empathetic responses</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()