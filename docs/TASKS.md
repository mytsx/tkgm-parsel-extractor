# TKGM Parsel Extractor - Optimizasyon Gorev Listesi

## Mevcut Durum
- app.py: Quadtree + Grid + Boundary Walking + Gap Fill hibrit sistem
- tkgm_client.py: Batch destegi
- Worker: /batch endpoint MEVCUT

---

## FAZ 1: Batch + Pruning Hibrit ✅ TAMAMLANDI
**Hedef:** Mevcut koda minimum mudahale ile %90 maliyet dususu

### Gorevler:
- [x] 1.1 app.py - ScanWorker'a batch destegi ekle
  - Noktalari 15'li gruplar halinde `/batch` endpoint'ine gonder
  - Her batch sonrasi donen geometrileri kullanarak pruning yap
  - Progress bar'i batch bazli guncelle

- [x] 1.2 Worker - /batch endpoint optimizasyonu
  - Paralel TKGM sorgulari (Promise.all)
  - Rate limiting icin kucuk delay (STAGGER_DELAY_MS = 50)

- [x] 1.3 Hata yonetimi
  - Batch icinde basarisiz sorgular icin retry mekanizmasi (max 2 retry)
  - Kismi basari durumunda devam etme

**Kazanc:** 10-20x daha az HTTP istegi

---

## FAZ 2: Quadtree Adaptif Ornekleme ✅ TAMAMLANDI
**Hedef:** Akilli ornekleme ile sorgu sayisini minimize etme

### Gorevler:
- [x] 2.1 Quadtree veri yapisi olustur
  - QuadTreeNode sinifi: bounds, children[4], parcel_ids, is_leaf
  - Recursive subdivide fonksiyonu
  - MIN_CELL_SIZE = 20m

- [x] 2.2 Adaptif ornekleme algoritmasi
  - 5 nokta ornekleme (4 kose + merkez)
  - Farkli parseller bulunursa subdivide et
  - Minimum hucre boyutu limiti (20m)

- [x] 2.3 Grid ile hibrit entegrasyon
  - Buyuk hucreler: Quadtree ornekleme
  - Kucuk hucreler: Yogun grid tarama
  - Pruning tum seviyelerde aktif

**Kazanc:** 50-100x daha az sorgu (ozellikle buyuk parselli alanlar)

---

## FAZ 3: Sinir Takibi (Boundary Walking) + Gap Fill ✅ TAMAMLANDI
**Hedef:** Parsel sinirlarindan yuruyerek komsulari bulma + bosluk doldurma

### Gorevler:
- [x] 3.1 Sinir noktasi hesaplama
  - Parsel polygon'unun her kenarinin orta noktalarini bul
  - Kenardan disa dogru 8m adim at (OUTWARD_STEP_METERS)
  - Kenar boyunca 10m aralikla noktalar (EDGE_STEP_METERS)

- [x] 3.2 Flood-fill benzeri yayilma
  - Queue-based BFS algoritmasi
  - Ziyaret edilen parselleri hash ile takip et
  - Max 5 iterasyon (sonsuz dongu korunmasi)
  - Yeni parsel bulundukca kuyruga ekle

- [x] 3.3 Gap Filling (Bosluk Doldurma)
  - Son asamada kalan bosluklari tespit et
  - Daha yogun grid ile tara (step_meters / 2)
  - Pruning ile gereksiz sorgulari ele

**Kazanc:** Tam kapsam garantisi + sorgu sayisi ≈ Parsel sayisi x 4-8

---

## FAZ 4: Istatistiksel Durdurma
**Hedef:** "Ne zaman durmaliyim?" sorusuna matematiksel cevap

### Gorevler:
- [ ] 4.1 Parsel alan dagilimi analizi
  - Bulunan parsellerin alanlarini topla
  - Dagilim modeli olustur (Log-Normal)

- [ ] 4.2 Durdurma kriteri
  - Kalan bosluk < dagilimin %1 persentili ise dur
  - Kullaniciya "Tahmini kapsam: %98" goster

---

## FAZ 5: Computer Vision Entegrasyonu (GELECEK)
**Hedef:** Uydu goruntusu ile on-bilgi

### Gorevler:
- [ ] 5.1 Uydu goruntusu API entegrasyonu
  - Google Maps Static API veya Mapbox
  - Bounding box icin goruntu cek

- [ ] 5.2 Edge detection
  - OpenCV Canny edge detection
  - Veya Segment Anything Model (SAM)

- [ ] 5.3 Aday nokta cikarimi
  - Tespit edilen alanlarin centroid'lerini hesapla
  - Sadece bu noktalara sorgu at

---

## Oncelik Sirasi

```
FAZ 1 ──► FAZ 2 ──► FAZ 3 ──► FAZ 4 ──► FAZ 5
  ✅        ✅        ✅     (Kolay)   (Zor)
 %90       %95      %99+     %99++    %99+++
```

---

## Notlar

- FAZ 1, 2, 3 tamamlandi - hibrit sistem aktif
- FAZ 4 (Istatistiksel Durdurma) opsiyonel optimizasyon
- FAZ 5 ek API maliyeti getirir, ROI hesaplanmali
- Mevcut sistem tam kapsam garantisi sagliyor (Gap Fill sayesinde)
