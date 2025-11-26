# TKGM Parsel Extractor - Optimizasyon Gorev Listesi

## Mevcut Durum
- app.py: Geometri bazli eleme (pruning) VAR, batch YOK
- tkgm_client.py: Batch VAR, pruning YOK
- Worker: /batch endpoint MEVCUT

---

## FAZ 1: Batch + Pruning Hibrit (ONCELIKLI)
**Hedef:** Mevcut koda minimum mudahale ile %90 maliyet dususu

### Gorevler:
- [x] 1.1 app.py - ScanWorker'a batch destegi ekle
  - Noktalari 10-20'li gruplar halinde `/batch` endpoint'ine gonder
  - Her batch sonrasi donen geometrileri kullanarak pruning yap
  - Progress bar'i batch bazli guncelle

- [x] 1.2 Worker - /batch endpoint optimizasyonu
  - Paralel TKGM sorgulari (Promise.all)
  - Rate limiting icin kucuk delay

- [ ] 1.3 Hata yonetimi
  - Batch icinde basarisiz sorgular icin retry mekanizmasi
  - Kismi basari durumunda devam etme

**Beklenen Kazanc:** 10-20x daha az HTTP istegi

---

## FAZ 2: Quadtree Adaptif Ornekleme
**Hedef:** Akilli ornekleme ile sorgu sayisini minimize etme

### Gorevler:
- [ ] 2.1 Quadtree veri yapisi olustur
  - Node: bounds, children[4], parcels, is_complete
  - Recursive subdivide fonksiyonu

- [ ] 2.2 Adaptif ornekleme algoritmasi
  - Kare merkezine sorgu at
  - Donen parsel kareyi ne kadar kapliyor hesapla
  - Kaplama < %80 ise 4'e bol ve recurse
  - Minimum kare boyutu limiti (ornek: 10m)

- [ ] 2.3 UI entegrasyonu
  - "Tarama Modu" secenegi: Grid / Quadtree
  - Quadtree goruntulemesi (opsiyonel)

**Beklenen Kazanc:** 50-100x daha az sorgu (ozellikle buyuk parselli alanlar)

---

## FAZ 3: Sinir Takibi (Boundary Walking)
**Hedef:** Parsel sinirlarindan yuruyerek komsulari bulma

### Gorevler:
- [ ] 3.1 Sinir noktasi hesaplama
  - Parsel polygon'unun her kenarinin orta noktasini bul
  - Kenardan disa dogru 5-10m adim at

- [ ] 3.2 Flood-fill benzeri yayilma
  - Queue-based BFS algoritmasi
  - Ziyaret edilen parselleri takip et
  - Tum komsular bulunana kadar devam et

- [ ] 3.3 Hibrit yaklasim
  - Ilk parsel icin grid/quadtree kullan
  - Sonra sinir takibi ile yayil

**Beklenen Kazanc:** Sorgu sayisi ≈ Parsel sayisi x 4-8

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
(Kolay)   (Orta)   (Orta)   (Kolay)   (Zor)
 %90       %95      %99      %99+     %99++
```

---

## Notlar

- Her faz bagimsiz olarak deploy edilebilir
- FAZ 1 tek basina buyuk fark yaratacak
- FAZ 2 ve 3 birbirinin alternatifi olabilir (ikisini de yapmak sart degil)
- FAZ 5 ek API maliyeti getirir, ROI hesaplanmali
