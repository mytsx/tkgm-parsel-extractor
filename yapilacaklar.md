-- çıktı formatı default kml olsun

-- bekleme süresini biz girmemeliyiz detay ayarlar ekranında vs olabilir oradan ayarlarız çünkü o ince ayar sonuçta

-- işlem kısmında kaç noktanın sorgulandığı kaç noktanın kaldığını vs göstermeli arada artıyor vs ya o durumdada güncellemeli 

-- kulanıcı eksik yerler olduğunu düşündüğünde detaylandır dediğinde halen boş alanlar varsa oraları daha detaylı grid ile tekrar çekmeye çalışmalı

-- daha önceden indirilen alanlar varsa onları local bir db'te tutmalı varsa orada alan oradan çekmeli. bunun sonrasında local değilde tamamen güncel çek gibi veya çektikten sonra local'den gelenleri güncel çek gibi özellikler eklenebilir. local'deki veriler için kml'ye çeklilme tarihi bilgisi eklenmeli (sadece kml değil tabi diğer veri formatları içinde geçerli). local db için geoloc verileri en iyi sanırım postgre'de tutabiliyoruz o yüzden onun kullanalım docker'da ayarlayabiliriz sanırım.