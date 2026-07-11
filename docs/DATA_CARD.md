# Veri Kartı

## Kaynaklar

- Hugging Face: `Overfit-GM/turkish-toxic-language`
- Kaggle: `toygarr/turkish-offensive-language-detection`

Mevcut temizlenmiş sürüm 98.358 örnektir: train 73.768, validation 9.836 ve test 14.754. İnceleme sırasında bölümler arasında birebir aynı metin bulunmamıştır.

11 Temmuz 2026 denetiminde mevcut CSV'lerde iki veya daha az kelimeli örnek sayısı sıfır çıkmıştır. Bunlar eski temizleme sürümünde elendiği için mevcut veriyle eğitilmiş model kısa sosyal medya mesajlarını yeterince temsil etmez. `turkish-toxic-v2` ile ham veriden yeniden üretim ve yeniden eğitim, üretime geçiş için zorunludur.

## İşleme

`backend_api.text_processing.normalize_for_model` eğitim ve canlı tahmin için ortak kullanılır. URL ve kullanıcı adları belirtece dönüştürülür; hashtag metni, kısa mesajlar, Türkçe karakterler ve anlamlı noktalama korunur. Aynı normalize metne farklı etiket verilirse pipeline `etiket_celiskileri.csv` üretip durur.

Split, etiket ve veri kaynağı birleşimine göre katmanlı yapılır. `kaynak` sütunu çıktı dosyalarında korunur; böylece kaynak bazında performans ölçülebilir.

## Kalite kontrolleri

`python tools/audit_dataset.py` şu kontrolleri yapar:

- boş ve yinelenen metinler
- bölümler arası birebir sızıntı
- çelişkili etiketler
- kısa mesaj oranı
- sınıf ve kaynak dağılımı
- her split için SHA-256 veri parmak izi

## Sınırlamalar

Etiket politikaları iki kaynak arasında tamamen aynı olmayabilir. Rastgele split aynı kullanıcıya ait veya birbirinin yeniden yazımı olan örnekleri ayırmayabilir. Kullanıcı/konuşma kimliği sağlanabildiğinde grup bazlı; zaman bilgisi sağlanabildiğinde zamansal holdout tercih edilmelidir.

Kamuya açık metinler de kişisel veri içerebilir. Ham veri yeniden dağıtılmadan önce kaynak lisansı, kullanım koşulları, silme talepleri ve kişisel veri yükümlülükleri kontrol edilmelidir.
