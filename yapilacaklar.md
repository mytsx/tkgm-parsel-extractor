# Yapilacaklar

## UI Iyilestirmeleri
- [x] Cikti formati varsayilan olarak KML olsun
- [ ] Bekleme suresi ayari "Gelismis Ayarlar" menusune tasinmali

## Tarama Ozellikleri
- [x] Tarama sirasinda sorgulanan/kalan nokta sayisi gibi detayli istatistikler gosterilmeli
- [x] Kullanicinin eksik bolgeleri daha yogun grid ile tekrar tarayabilmesi icin "Detaylandir" ozelligi

## Gelecek Ozellikler (Lokal Veritabani)
- [ ] Indirilen parsel verileri lokal bir veritabaninda (PostGIS + PostgreSQL) saklanmali
  - [ ] Docker Compose ile PostgreSQL kurulumu
  - [ ] Veri guncellik tarihi alani eklenmeli
  - [ ] "Onbellekten kullan" veya "Verileri guncelle" secenekleri
  - [ ] KML/GeoJSON export'a cekilme tarihi bilgisi eklenmeli

## Kod Kalitesi (Gelecek Refactoring)
- [ ] ScanWorker sinifini ayir: QuadtreeScanner, BoundaryWalker, GapFiller
