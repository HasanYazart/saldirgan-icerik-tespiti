# Model Kartı — Turkish Toxic BERT v2

## Amaç

Türkçe sosyal medya metinlerinde saldırgan veya toksik içerik olasılığı üretmek. Model kararı yüksek kesinlikli kaçınma kurallarıyla birleştirilir; eşik yakınındaki sonuçlar insan incelemesine gönderilir.

## Model ve giriş

- Taban model: `dbmdz/bert-base-turkish-cased`
- Görev: İkili sınıflandırma (`0=clean`, `1=toxic`)
- En uzun dizi: varsayılan 160 token (manifestte sürümlenir)
- Belge deneyi ön işlemesi: `belge3-word2vec-v1`
- Ana modeller: train-only Word2Vec + LSTM, CNN ve CNN→LSTM
- Olasılık kalibrasyonu: validation NLL ile temperature scaling
- Karar eşiği: yalnızca validation bölümünde hedef yanlış pozitif oranı altında en yüksek F1'e göre seçilir

Gerçek model sürümü, eşik, SHA-256 ve validation ölçümleri dağıtılan `model_manifest.json` içinde bulunur. Manifest yoksa veya ön işleme sürümü eşleşmiyorsa model hazır kabul edilmez.

## Uygun kullanım

- İçerik sıralama ve moderatör önceliklendirme
- Kullanıcıya gönderim öncesi uyarı
- İnsan incelemesine yardımcı sinyal

## Uygun olmayan kullanım

- Tek başına kalıcı ban veya hukuki karar
- Kullanıcının kişiliği, niyeti veya tehlikeliliği hakkında çıkarım
- Türkçe dışındaki dillerde güvenilir sınıflandırma
- Özel nitelik, siyasi görüş veya sağlık bilgisi çıkarımı

## Riskler

- Alıntı, haber, eğitim, mizah ve karşı söylem yanlış pozitif üretebilir.
- Yeni argo ve kaçınma teknikleri yanlış negatif üretebilir.
- Lehçe, bölgesel dil ve kimlik ifadelerinde gruplar arası hata oranı değişebilir.
- 128 token sonrasındaki içerik modele girmez.

## Zorunlu değerlendirme

Her model sürümünde accuracy dışında precision, recall, F1, ROC-AUC, PR-AUC, false-positive rate ve false-negative rate raporlanmalıdır. Kısa mesaj, tehdit, alıntı, emoji, leetspeak, lehçe ve kimlik terimi dilimleri ayrıca ölçülmelidir. Üretim drift'i ve itiraz kabul oranı model sürümüne göre izlenmelidir.
