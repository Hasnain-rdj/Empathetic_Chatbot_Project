# Empathetic Chatbot with Transformer Architecture

A sophisticated AI-powered chatbot that generates empathetic responses based on user emotions and situational context. Built using PyTorch Transformer architecture with attention mechanisms and deployed via Streamlit for interactive web interface.

## 🎯 Project Overview

This project implements an empathetic conversational AI that can understand human emotions and respond with appropriate empathy. The system uses a custom Transformer model trained on emotional dialogue data to generate contextually relevant and emotionally intelligent responses.

## ✨ Key Features

- **🤖 Transformer-based Architecture**: Custom encoder-decoder model with multi-head attention
- **😊 Emotion Recognition**: Supports 32+ different emotions (joy, sadness, anger, fear, etc.)
- **🎯 Context Awareness**: Considers both user input and situational context
- **🔍 Multiple Decoding Strategies**: Greedy search and beam search options
- **📊 Attention Visualization**: Interactive heatmaps showing model attention patterns
- **💬 Interactive Web Interface**: User-friendly Streamlit application
- **📱 Real-time Chat**: Live conversation with response generation
- **🕒 Performance Metrics**: Generation time and method tracking

## 🏗 Architecture

### Model Specifications
- **Architecture**: Transformer (Encoder-Decoder)
- **Model Dimension**: 512
- **Attention Heads**: 2
- **Encoder Layers**: 2
- **Decoder Layers**: 2
- **Feed-forward Dimension**: 2048
- **Vocabulary Size**: 21,295 tokens
- **Parameters**: ~47M parameters

### Technical Components
- **Positional Encoding**: Sinusoidal embeddings for sequence position
- **Multi-Head Attention**: Self and cross-attention mechanisms
- **Layer Normalization**: Stabilized training with residual connections
- **Masked Attention**: Prevents information leakage during decoding
- **Custom Vocabulary**: Specialized tokenization for emotional dialogue

## 📁 Project Structure

```
├── empathetic_chatbot.ipynb      # Main training notebook with model development
├── streamlit_app.py              # Interactive web application
├── requirements.txt              # Python dependencies
├── saved_models/
│   └── best_model.pkl           # Trained model checkpoint (545MB)
├── saved_data/                  # Processed datasets
│   ├── train_df.pkl            # Training data
│   ├── val_df.pkl              # Validation data
│   └── test_df.pkl             # Test data
└── saved_vocab/                 # Vocabulary files
    └── vocabulary.pkl          # Custom tokenizer vocabulary
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- Streamlit 1.28+
- 4GB+ RAM (for model loading)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Hasnain-rdj/Empathetic_Chatbot_Project.git
   cd empathetic-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the web interface**
   - Open your browser to `http://localhost:8501`
   - Start chatting with the empathetic AI!

## 💻 Usage

### Web Interface
1. **Enter Message**: Type your message in the text area
2. **Select Emotion**: Choose from 32+ emotions or use auto-detect
3. **Add Context**: Optionally describe the situation
4. **Choose Strategy**: Select Greedy or Beam Search decoding
5. **Generate Response**: Click to get empathetic AI response
6. **View Attention**: Explore attention visualization heatmaps

### Supported Emotions
- **Positive**: joyful, excited, grateful, hopeful, confident, proud
- **Negative**: sad, angry, frustrated, disappointed, anxious, afraid
- **Complex**: nostalgic, sentimental, surprised, embarrassed, guilty
- **And 20+ more nuanced emotional states**

## 📊 Model Performance

- **Training Dataset**: EmpatheticDialogues (69k conversations)
- **Training Time**: ~10 epochs with early stopping
- **Vocabulary Coverage**: 21,295 unique tokens
- **Response Generation**: <1 second average latency
- **Context Understanding**: Multi-turn conversation support

## 🔧 Technical Implementation

### Training Process
1. **Data Preprocessing**: Text normalization and tokenization
2. **Model Architecture**: Custom Transformer implementation
3. **Training Strategy**: Teacher forcing with attention supervision
4. **Optimization**: Adam optimizer with learning rate scheduling
5. **Evaluation**: Perplexity and human evaluation metrics

### Inference Pipeline
1. **Input Processing**: Text normalization and emotion encoding
2. **Context Integration**: Situation and emotion context embedding
3. **Sequence Generation**: Autoregressive decoding with attention
4. **Response Post-processing**: Token-to-text conversion
5. **Attention Extraction**: Visualization-ready attention weights

## 🎨 Features Showcase

### Interactive Chat Interface
- Real-time conversation with AI
- Message history with timestamps
- Emotion-based response customization
- Situational context integration

### Attention Visualization
- Multi-head attention heatmaps
- Layer-wise attention analysis
- Interactive exploration tools
- Cross-attention pattern insights

### Decoding Options
- **Greedy Search**: Fast, deterministic responses
- **Beam Search**: Higher quality, diverse outputs
- **Customizable Parameters**: Length control, repetition handling

## 📚 Research Foundation

This project builds upon:
- **Transformer Architecture**: "Attention Is All You Need" (Vaswani et al.)
- **Empathetic Dialogues**: EmpatheticDialogues dataset
- **Emotional AI**: Research in computational empathy
- **Conversational AI**: Modern dialogue generation techniques

## 🔬 Future Enhancements

- **Multi-modal Input**: Image and voice emotion recognition
- **Personality Modeling**: Consistent character traits
- **Memory Integration**: Long-term conversation context
- **Fine-tuning Options**: Domain-specific customization
- **Performance Optimization**: Model compression and acceleration

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or collaboration opportunities, please open an issue or contact the repository maintainer.

---

**Built with ❤️ using PyTorch, Transformers, and Streamlit**
