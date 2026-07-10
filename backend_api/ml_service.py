import os
import torch

try:
    from transformers import BertTokenizer, BertForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Eğitilmiş modelin yolu (Şu an sahte, Colab'dan gelince buraya koyulacak)
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
                print("Mock (Sahte) model moduna geçiliyor.")
                self.is_mock = True
        else:
            print("Model dosyası (bert_best.pt) bulunamadı veya transformers yüklü değil.")
            print("Mock (Sahte) model modunda çalışılıyor...")
            self.is_mock = True

    def analyze_text(self, text: str) -> dict:
        \"\"\"
        Metni analiz eder ve ne kadar saldırgan olduğunu döndürür.
        \"\"\"
        if self.is_mock:
            # Sahte model mantığı: İçinde argo bir kelime geçiyorsa direkt %95 saldırgan say.
            bad_words = ['s***', 'aptal', 'salak', 'gerizekalı', 'lan']
            is_toxic = any(bw in text.lower() for bw in bad_words)
            score = 0.95 if is_toxic else 0.05
            return {"is_toxic": is_toxic, "toxicity_score": score}

        # GERÇEK MODEL ÇALIŞMASI
        with torch.no_grad():
            encoding = self.tokenizer(
                text, add_special_tokens=True, max_length=128,
                padding='max_length', truncation=True,
                return_attention_mask=True, return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits.float(), dim=1)
            
            # Label 1 = Saldırgan, Label 0 = Normal
            toxic_prob = probs[0][1].item()
            is_toxic = toxic_prob > 0.5
            
            return {"is_toxic": is_toxic, "toxicity_score": round(toxic_prob, 4)}

# Singleton instance
ml_service = MLService()
