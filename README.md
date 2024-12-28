# Multi-Label Emotion Classification with Fine-Tuned RoBERTa

This project fine-tunes a pretrained RoBERTa model to classify text into seven distinct emotions (admiration, amusement, gratitude, love, pride, relief, remorse). By leveraging GPU acceleration, dropout regularization, and early stopping, it achieves a significant improvement over a baseline F1 score, providing reliable multi-label classification performance for emotion detection tasks.

**Files & Structure**  
- `train.csv` / `dev.csv`: Training & Validation data  
- `nn.py`: Main Python script (training & prediction)  
- `test.ipynb`: Notebook for experimentation  
- `requirements.txt`: Dependencies  

**Key Steps**  
1. **Clone & Install**  
   ```bash
   git clone https://github.com/Anshkumardev/Multi-Label-Emotion-Classification-with-Fine-Tuned-RoBERTa.git
   cd Multi-Label-Emotion-Classification-with-Fine-Tuned-RoBERTa
   pip install -r requirements.txt

2. **Train & Predict**
   ```
   python nn.py
