# Veri Kartı

## Kaynaklar

- Hugging Face: `Overfit-GM/turkish-toxic-language`
- Kaggle: `toygarr/turkish-offensive-language-detection`
- Çalışma alanına eklenen, kaçınma varyasyonları ve kısa mesajlar içeren
  `train_1/test_1/train_2/test_2.csv`

Ham envanter sekiz CSV'de 141.320 satırdır. `belge3-word2vec-v1` doğrulamasında
temizlik ve tekilleştirme sonrasında 106.403 örnek kalmıştır: 79.801 train,
10.640 validation ve 15.962 test. Her çalıştırmanın kesin satır/sınıf sayıları
`sonuclar/veri_kalitesi/data_audit.json` dosyasına yazılır.

11 Temmuz 2026 denetiminde mevcut CSV'lerde iki veya daha az kelimeli örnek sayısı sıfır çıkmıştır. Bunlar eski temizleme sürümünde elendiği için mevcut veriyle eğitilmiş model kısa sosyal medya mesajlarını yeterince temsil etmez. `turkish-toxic-v2` ile ham veriden yeniden üretim ve yeniden eğitim, üretime geçiş için zorunludur.

## İşleme

Belge (3), bölüm 3.1'e uygun `belge3-word2vec-v1` ön işlemesi kullanılır. Metin
küçük harfe çevrilir; URL/kullanıcı adı, noktalama, özel karakter, fazla boşluk ve
Türkçe durak kelimeler temizlenir. Sınıflandırma anlamını tersine çevirebildikleri
için `değil`, `yok` ve `hayır` korunur. Aynı temizlenmiş metne farklı etiket
verilirse varsayılan davranış tüm çelişkili örnekleri eğitim dışında karantinaya
almak ve `etiket_celiskileri.csv` üretmektir. İstenirse `--conflict-policy error`
ile işlem durdurulur.

Görev ikili sınıflandırmadır: `clean=0`; `offensive`, `hate`, `threat`,
`targeted_abuse` ve `sexual_profanity=1`. Açık bir etiketçi uyuşmazlığı bulunan
satırlar reddedilir. Dosyalardaki `placeholder` etiketçi notları kalite raporunda
sayılır; bu satırlar üretim kararı öncesinde gerçek moderatörlerle doğrulanmalıdır.

Split, etiket ve veri kaynağı birleşimine göre katmanlı ve içerik grubu bazlı yapılır. Kelime sırası değiştirilmiş ve tek kelime silinmiş yakın türevler `group_id` ile aynı split'te tutulur. `kaynak` ve `group_id` çıktı dosyalarında korunur; böylece kaynak bazında performans ve yakın-kopya sızıntısı ölçülebilir.

## Kalite kontrolleri

`python tools/audit_dataset.py` şu kontrolleri yapar:

- boş ve yinelenen metinler
- bölümler arası birebir sızıntı
- bölümler arası yakın-kopya grup sızıntısı
- çelişkili etiketler
- kısa mesaj oranı
- sınıf ve kaynak dağılımı
- her split için SHA-256 veri parmak izi

## Sınırlamalar

Etiket politikaları iki kaynak arasında tamamen aynı olmayabilir. Rastgele split aynı kullanıcıya ait veya birbirinin yeniden yazımı olan örnekleri ayırmayabilir. Kullanıcı/konuşma kimliği sağlanabildiğinde grup bazlı; zaman bilgisi sağlanabildiğinde zamansal holdout tercih edilmelidir.

Kamuya açık metinler de kişisel veri içerebilir. Ham veri yeniden dağıtılmadan önce kaynak lisansı, kullanım koşulları, silme talepleri ve kişisel veri yükümlülükleri kontrol edilmelidir.
