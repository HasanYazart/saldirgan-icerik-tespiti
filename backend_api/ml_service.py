import os
import torch
from .nlp_service import preprocess_text, dictionary_check

try:
    from transformers import BertTokenizer, BertForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

MODEL_PATH = "bert_best.pt"
MODEL_NAME = "dbmdz/bert-base-turkish-cased"

class MLService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_mock = True
        self.tokenizer = None
        self.model = None
        
        self._load_model()

    def _load_model(self):
        if TRANSFORMERS_AVAILABLE and os.path.exists(MODEL_PATH):
            print("Gerçek BERT modeli yükleniyor...")
            try:
                self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
                self.model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.is_mock = False
                print("✅ Model başarıyla yüklendi!")
            except Exception as e:
                print(f"Model yüklenirken hata oluştu: {e}")
                self.is_mock = True
        else:
            self.is_mock = True

    def analyze_text(self, text: str) -> dict:
        \"\"\"
        Metni analiz eder ve ne kadar saldırgan olduğunu döndürür.
        \"\"\"
        # 1. NLP ÖN İŞLEME (Stemming, Typo Correction, Emoji Translation)
        clean_text = preprocess_text(text)
        
        # 2. SÖZLÜK (KURAL BAZLI) KONTROL (Evasion/Boşluk Hilelerini Engeller)
        if dictionary_check(clean_text):
            return {"is_toxic": True, "toxicity_score": 0.99}

        # 3. YAPAY ZEKA (BERT) ANALİZİ
        if self.is_mock:
            # Mock mod: Zaten sözlük yakalamadıysa %95 güvenle temiz sayılır
            return {"is_toxic": False, "toxicity_score": 0.05}

        # GERÇEK MODEL ÇALIŞMASI
        with torch.no_grad():
            encoding = self.tokenizer(
                clean_text, add_special_tokens=True, max_length=128,
                padding='max_length', truncation=True,
                return_attention_mask=True, return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits.float(), dim=1)
            
            toxic_prob = probs[0][1].item()
            is_toxic = toxic_prob > 0.5
            
            return {"is_toxic": is_toxic, "toxicity_score": round(toxic_prob, 4)}

ml_service = MLService()
