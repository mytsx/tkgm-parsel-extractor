#!/usr/bin/env python3
"""
TKGM Parsel Veri Cekme - Modern Desktop Uygulamasi
PyQt5 + Fluent Design
"""

import sys
import json
import math
import time
from xml.etree import ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings

# Fluent Widgets
from qfluentwidgets import (
    PushButton, PrimaryPushButton, LineEdit, BodyLabel, TitleLabel,
    SubtitleLabel, CardWidget, ProgressBar, TextEdit,
    SpinBox, DoubleSpinBox, InfoBar, InfoBarPosition, FluentIcon,
    setTheme, Theme
)

import requests


# ==================== VERI MODELLERI ====================

@dataclass
class BoundingBox:
    """Cografi sinir kutusu (bounding box)."""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float


# ==================== KML PARSER ====================

class KMLParser:
    """KML dosyalarini parse eden yardimci sinif."""

    @staticmethod
    def parse(kml_path: str) -> List[List[Tuple[float, float]]]:
        """KML dosyasindan polygon koordinatlarini cikarir."""
        tree = ET.parse(kml_path)
        root = tree.getroot()
        polygons = []

        for coordinates in root.iter():
            if coordinates.tag.endswith('coordinates'):
                coords_text = coordinates.text.strip()
                polygon = KMLParser._parse_coordinates(coords_text)
                if polygon:
                    polygons.append(polygon)
        return polygons

    @staticmethod
    def _parse_coordinates(coords_text: str) -> List[Tuple[float, float]]:
        """Koordinat stringini (lat, lon) tuple listesine donusturur."""
        coords = []
        for coord in coords_text.split():
            parts = coord.strip().split(',')
            if len(parts) >= 2:
                lon = float(parts[0])
                lat = float(parts[1])
                coords.append((lat, lon))
        return coords

    @staticmethod
    def get_bounding_box(polygon: List[Tuple[float, float]]) -> BoundingBox:
        """Polygon icin bounding box hesaplar."""
        lats = [p[0] for p in polygon]
        lons = [p[1] for p in polygon]
        return BoundingBox(min(lats), max(lats), min(lons), max(lons))

    @staticmethod
    def point_in_polygon(lat: float, lon: float, polygon: List[Tuple[float, float]]) -> bool:
        """Ray casting algoritmasi ile noktanin polygon icinde olup olmadigini kontrol eder."""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            if ((polygon[i][1] > lon) != (polygon[j][1] > lon)) and \
               (lat < (polygon[j][0] - polygon[i][0]) * (lon - polygon[i][1]) /
                (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
                inside = not inside
            j = i
        return inside


# ==================== TKGM CLIENT ====================

class TKGMClient:
    """Cloudflare Worker uzerinden TKGM API istemcisi."""

    def __init__(self, worker_url: str, api_key: str = None):
        """Worker URL ve opsiyonel API key ile istemci olusturur."""
        self.worker_url = worker_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        })
        if api_key:
            self.session.headers["X-API-Key"] = api_key

    def get_parsel(self, lat: float, lon: float) -> tuple:
        """Returns (data, error_code) tuple"""
        try:
            url = f"{self.worker_url}/parsel/{lat}/{lon}"
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'properties' in data:
                    return data, None
            elif response.status_code == 401:
                return None, 401
            return None, response.status_code
        except requests.RequestException as e:
            return None, str(e)

    def get_batch(self, coordinates: list) -> tuple:
        """
        Batch sorgu - birden fazla koordinati tek istekte sorgula
        Args:
            coordinates: [(lat, lon), (lat, lon), ...] listesi
        Returns:
            (results_list, error_code) tuple
            results_list: [{data}, {data}, ...] veya None'lar
        """
        try:
            url = f"{self.worker_url}/batch"
            payload = {
                "coordinates": [{"lat": c[0], "lon": c[1]} for c in coordinates]
            }
            response = self.session.post(url, json=payload, timeout=120)

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                # Her sonucu kontrol et, gecerli parsel mi?
                processed = []
                for r in results:
                    if r and isinstance(r, dict) and 'properties' in r:
                        processed.append(r)
                    else:
                        processed.append(None)
                return processed, None
            if response.status_code == 401:
                return None, 401
            return None, response.status_code
        except requests.RequestException as e:
            return None, str(e)


# ==================== WORKER THREAD ====================

class ScanWorker(QThread):
    """Arka planda alan taramasi yapan worker thread."""

    BATCH_SIZE = 15  # Her batch'te kac nokta sorgulanacak

    progress = pyqtSignal(int, int)  # current, total
    log = pyqtSignal(str)
    parcel_found = pyqtSignal(str, dict)
    finished_scan = pyqtSignal(int)
    auth_error = pyqtSignal()  # 401 hatasi icin

    def __init__(self, client: TKGMClient, polygons: List, step_meters: int, delay: float):
        """Tarama worker'i olusturur."""
        super().__init__()
        self.client = client
        self.polygons = polygons
        self.step_meters = step_meters
        self.delay = delay
        self.is_running = True

    def run(self):
        """Tarama islemini calistirir."""
        all_points = []
        for polygon in self.polygons:
            points = self.generate_grid_points(polygon)
            all_points.extend(points)

        total_points = len(all_points)
        remaining_points = list(all_points)  # Liste olarak tut (batch icin)
        self.log.emit(f"Toplam {total_points} nokta taranacak (batch size: {self.BATCH_SIZE})")

        found_count = 0
        found_ids = set()
        api_calls = 0
        total_pruned = 0

        while remaining_points and self.is_running:
            # Batch icin noktalari al
            batch_points = remaining_points[:self.BATCH_SIZE]
            remaining_points = remaining_points[self.BATCH_SIZE:]

            api_calls += 1
            self.log.emit(f"Batch #{api_calls}: {len(batch_points)} nokta sorgulanıyor...")

            results, error = self.client.get_batch(batch_points)

            # 401 Unauthorized hatasi
            if error == 401:
                self.log.emit("HATA: API Key hatali veya eksik (401 Unauthorized)")
                self.auth_error.emit()
                break

            if error:
                self.log.emit(f"  Batch hatasi: {error}")
                continue

            if not results:
                continue

            # Batch sonuclarini isle
            new_parcels_in_batch = []
            for i, result in enumerate(results):
                if result and 'properties' in result:
                    parsel_id = result['properties'].get('ozet', f"unknown_{i}")
                    if parsel_id not in found_ids:
                        found_ids.add(parsel_id)
                        found_count += 1
                        new_parcels_in_batch.append(result)
                        self.parcel_found.emit(parsel_id, result)
                        props = result['properties']
                        self.log.emit(f"  ✓ {parsel_id} - {props.get('nitelik', '')} ({props.get('alan', '')})")

            # Batch sonrasi pruning: Bulunan parsellerin icine dusen noktalari ele
            if new_parcels_in_batch:
                # Tum yeni parsellerin poligonlarini topla
                new_polygons = [
                    [(c[1], c[0]) for c in result['geometry'].get('coordinates', [[]])[0]]
                    for result in new_parcels_in_batch
                    if 'geometry' in result and result['geometry'].get('type') == 'Polygon'
                ]

                if new_polygons:
                    initial_count = len(remaining_points)

                    # Tek geciste tum poligonlara karsi kontrol et
                    remaining_points = [
                        p for p in remaining_points
                        if not any(KMLParser.point_in_polygon(p[0], p[1], poly) for poly in new_polygons)
                    ]

                    pruned_in_batch = initial_count - len(remaining_points)
                    if pruned_in_batch > 0:
                        total_pruned += pruned_in_batch
                        self.log.emit(f"  ↳ {pruned_in_batch} nokta elendi (parseller icinde)")

            # Progress guncelle
            processed = total_points - len(remaining_points)
            self.progress.emit(processed, total_points)

            # Batch arasi bekleme
            time.sleep(self.delay)

        if not self.is_running:
            self.log.emit("Tarama durduruldu")

        self.log.emit(f"\n{'='*50}")
        self.log.emit(f"SONUC: {found_count} parsel bulundu")
        self.log.emit(f"API cagrilari: {api_calls} batch ({api_calls * self.BATCH_SIZE} yerine)")
        self.log.emit(f"Elenen noktalar: {total_pruned}")
        savings = ((total_points - api_calls) / total_points * 100) if total_points > 0 else 0.0
        self.log.emit(f"Tasarruf: ~{savings:.1f}%")
        self.finished_scan.emit(found_count)

    def generate_grid_points(self, polygon):
        """Polygon icinde grid noktalari olusturur."""
        bbox = KMLParser.get_bounding_box(polygon)
        center_lat = (bbox.min_lat + bbox.max_lat) / 2
        lat_step = self.step_meters / 111000
        lon_step = self.step_meters / (111000 * math.cos(math.radians(center_lat)))

        points = []
        lat = bbox.min_lat
        while lat <= bbox.max_lat:
            lon = bbox.min_lon
            while lon <= bbox.max_lon:
                if KMLParser.point_in_polygon(lat, lon, polygon):
                    points.append((lat, lon))
                lon += lon_step
            lat += lat_step
        return points

    def stop(self):
        """Taramayi durdurur."""
        self.is_running = False


# ==================== ANA PENCERE ====================

class MainWindow(QMainWindow):
    """Ana uygulama penceresi."""

    def __init__(self):
        """Ana pencereyi olusturur ve yapilandirir."""
        super().__init__()
        self.setWindowTitle("TKGM Parsel Veri Cekme")
        self.setMinimumSize(800, 700)

        # Ayarlari yukle
        self.settings = QSettings("TKGM", "ParselCekme")
        self.polygons = []
        self.parcels = {}
        self.worker = None

        # Tema
        setTheme(Theme.LIGHT)

        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        """Kullanici arayuzunu olusturur."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Baslik
        title = TitleLabel("TKGM Parsel Veri Cekme")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Worker URL Card
        url_card = CardWidget()
        url_layout = QVBoxLayout(url_card)
        url_layout.addWidget(SubtitleLabel("Cloudflare Worker URL"))
        self.worker_url_input = LineEdit()
        self.worker_url_input.setPlaceholderText("https://your-worker.workers.dev")
        url_layout.addWidget(self.worker_url_input)

        # API Key (Opsiyonel)
        url_layout.addWidget(BodyLabel("API Key (Opsiyonel)"))
        self.api_key_input = LineEdit()
        self.api_key_input.setPlaceholderText("Bos birakilabilir - Worker herkese acik olur")
        url_layout.addWidget(self.api_key_input)

        layout.addWidget(url_card)

        # KML Card
        kml_card = CardWidget()
        kml_layout = QVBoxLayout(kml_card)
        kml_layout.addWidget(SubtitleLabel("KML Dosyasi"))

        kml_row = QHBoxLayout()
        self.kml_path_input = LineEdit()
        self.kml_path_input.setPlaceholderText("KML dosyasi secin...")
        self.kml_path_input.setReadOnly(True)
        kml_row.addWidget(self.kml_path_input)

        kml_btn = PushButton("Sec")
        kml_btn.setIcon(FluentIcon.FOLDER)
        kml_btn.clicked.connect(self.select_kml)
        kml_row.addWidget(kml_btn)
        kml_layout.addLayout(kml_row)

        self.kml_info = BodyLabel("")
        kml_layout.addWidget(self.kml_info)
        layout.addWidget(kml_card)

        # Ayarlar Card
        settings_card = CardWidget()
        settings_layout = QHBoxLayout(settings_card)

        # Grid araligi
        grid_layout = QVBoxLayout()
        grid_layout.addWidget(BodyLabel("Grid Araligi (metre)"))
        self.step_spin = SpinBox()
        self.step_spin.setRange(10, 200)
        self.step_spin.setValue(30)
        grid_layout.addWidget(self.step_spin)
        settings_layout.addLayout(grid_layout)

        # Bekleme
        delay_layout = QVBoxLayout()
        delay_layout.addWidget(BodyLabel("Bekleme Suresi (sn)"))
        self.delay_spin = DoubleSpinBox()
        self.delay_spin.setRange(0.1, 5.0)
        self.delay_spin.setValue(0.3)
        self.delay_spin.setSingleStep(0.1)
        delay_layout.addWidget(self.delay_spin)
        settings_layout.addLayout(delay_layout)

        layout.addWidget(settings_card)

        # Butonlar
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.start_btn = PrimaryPushButton("Taramayi Baslat")
        self.start_btn.setIcon(FluentIcon.PLAY)
        self.start_btn.clicked.connect(self.start_scan)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = PushButton("Durdur")
        self.stop_btn.setIcon(FluentIcon.PAUSE)
        self.stop_btn.clicked.connect(self.stop_scan)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        self.save_btn = PushButton("Kaydet")
        self.save_btn.setIcon(FluentIcon.SAVE)
        self.save_btn.clicked.connect(self.save_results)
        btn_layout.addWidget(self.save_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Progress
        progress_card = CardWidget()
        progress_layout = QVBoxLayout(progress_card)

        self.status_label = BodyLabel("Hazir")
        progress_layout.addWidget(self.status_label)

        self.progress_bar = ProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        layout.addWidget(progress_card)

        # Log
        log_card = CardWidget()
        log_layout = QVBoxLayout(log_card)
        log_layout.addWidget(SubtitleLabel("Log"))

        self.log_text = TextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_card)

    def load_settings(self):
        """Kaydedilmis ayarlari yukler."""
        url = self.settings.value("worker_url", "")
        if url:
            self.worker_url_input.setText(url)

        api_key = self.settings.value("api_key", "")
        if api_key:
            self.api_key_input.setText(api_key)

        step = self.settings.value("step_meters", 30, type=int)
        self.step_spin.setValue(step)

        delay = self.settings.value("delay", 0.3, type=float)
        self.delay_spin.setValue(delay)

    def save_settings(self):
        """Mevcut ayarlari kaydeder."""
        self.settings.setValue("worker_url", self.worker_url_input.text())
        self.settings.setValue("api_key", self.api_key_input.text())
        self.settings.setValue("step_meters", self.step_spin.value())
        self.settings.setValue("delay", self.delay_spin.value())

    def log(self, message: str):
        """Log alanina zaman damgali mesaj ekler."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def select_kml(self):
        """KML dosyasi secme dialogunu acar."""
        path, _ = QFileDialog.getOpenFileName(
            self, "KML Dosyasi Sec", "",
            "KML Files (*.kml);;All Files (*)"
        )
        if path:
            self.kml_path_input.setText(path)
            self.load_kml(path)

    def load_kml(self, path: str):
        """KML dosyasini yukler ve parse eder."""
        try:
            self.polygons = KMLParser.parse(path)
            info = f"{len(self.polygons)} polygon bulundu"

            total_points = 0
            for poly in self.polygons:
                bbox = KMLParser.get_bounding_box(poly)
                center_lat = (bbox.min_lat + bbox.max_lat) / 2
                lat_step = self.step_spin.value() / 111000
                lon_step = self.step_spin.value() / (111000 * math.cos(math.radians(center_lat)))
                lat_count = int((bbox.max_lat - bbox.min_lat) / lat_step) + 1
                lon_count = int((bbox.max_lon - bbox.min_lon) / lon_step) + 1
                total_points += lat_count * lon_count

            info += f" (~{total_points} tarama noktasi)"
            self.kml_info.setText(info)
            self.log(f"KML yuklendi: {path}")
            self.log(info)

            InfoBar.success(
                title="Basarili",
                content=info,
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000
            )
        except (ET.ParseError, ValueError, IOError) as e:
            self.log(f"HATA: {e}")
            InfoBar.error(
                title="Hata",
                content=str(e),
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=5000
            )

    def start_scan(self):
        """Tarama islemini baslatir."""
        if not self.polygons:
            InfoBar.warning(
                title="Uyari",
                content="Lutfen once KML dosyasi secin",
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000
            )
            return

        url = self.worker_url_input.text().strip()
        if not url or "your-worker" in url:
            InfoBar.warning(
                title="Uyari",
                content="Lutfen Worker URL girin",
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000
            )
            return

        self.save_settings()
        self.parcels = {}
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        api_key = self.api_key_input.text().strip() or None
        client = TKGMClient(url, api_key)

        # API Key uyarisi
        if not api_key:
            InfoBar.info(
                title="Bilgi",
                content="API Key girilmedi - Worker herkese acik",
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000
            )

        self.worker = ScanWorker(
            client,
            self.polygons,
            self.step_spin.value(),
            self.delay_spin.value()
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.log)
        self.worker.parcel_found.connect(self.on_parcel_found)
        self.worker.finished_scan.connect(self.on_finished)
        self.worker.auth_error.connect(self.on_auth_error)
        self.worker.start()

        self.log("Tarama baslatildi...")

    def stop_scan(self):
        """Devam eden taramayi durdurur."""
        if self.worker:
            self.worker.stop()
            self.log("Durdurma istegi gonderildi...")

    def on_progress(self, current: int, total: int):
        """Ilerleme sinyalini isler."""
        percent = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        self.status_label.setText(f"Taraniyor: {current}/{total} | Bulunan: {len(self.parcels)} parsel")

    def on_parcel_found(self, parsel_id: str, data: dict):
        """Bulunan parseli kaydeder."""
        self.parcels[parsel_id] = data

    def on_auth_error(self):
        """API Key hatasini isler."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Hata - API Key hatali")

        InfoBar.error(
            title="API Key Hatasi",
            content="API Key hatali veya eksik. Worker'da API_KEY tanimliysa ayni key'i girin.",
            parent=self,
            position=InfoBarPosition.TOP_RIGHT,
            duration=8000
        )

    def on_finished(self, count: int):
        """Tarama tamamlandiginda cagrilir."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"Tamamlandi - {count} parsel bulundu")
        self.log(f"Tarama tamamlandi! {count} benzersiz parsel bulundu")

        InfoBar.success(
            title="Tamamlandi",
            content=f"{count} parsel bulundu",
            parent=self,
            position=InfoBarPosition.TOP_RIGHT,
            duration=5000
        )

    def save_results(self):
        """Sonuclari dosyaya kaydetme dialogunu acar."""
        if not self.parcels:
            InfoBar.warning(
                title="Uyari",
                content="Kaydedilecek veri yok",
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Kaydet", "",
            "GeoJSON (*.geojson);;KML (*.kml);;JSON (*.json)"
        )

        if not path:
            return

        try:
            if path.endswith('.kml'):
                self.save_kml(path)
            else:
                self.save_geojson(path)

            self.log(f"Kaydedildi: {path}")
            InfoBar.success(
                title="Kaydedildi",
                content=path,
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000
            )
        except (IOError, OSError) as e:
            InfoBar.error(
                title="Hata",
                content=str(e),
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=5000
            )

    def save_geojson(self, path: str):
        """Parselleri GeoJSON formatinda kaydeder."""
        geojson = {
            "type": "FeatureCollection",
            "features": list(self.parcels.values())
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

    def save_kml(self, path: str):
        """Parselleri KML formatinda kaydeder."""
        kml = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>TKGM Parseller</name>
<Style id="s"><LineStyle><color>ff0000ff</color><width>2</width></LineStyle><PolyStyle><color>4d0000ff</color></PolyStyle></Style>
'''
        for pid, p in self.parcels.items():
            props = p.get('properties', {})
            geom = p.get('geometry', {})
            if geom.get('type') != 'Polygon':
                continue
            coords = geom.get('coordinates', [[]])[0]
            coord_str = ' '.join([f"{c[0]},{c[1]},0" for c in coords])
            name = props.get('ozet', pid)

            # Detayli aciklama
            desc = f"""<![CDATA[
<b>Il:</b> {props.get('ilAd', '-')}<br/>
<b>Ilce:</b> {props.get('ilceAd', '-')}<br/>
<b>Mahalle:</b> {props.get('mahalleAd', '-')}<br/>
<b>Mevkii:</b> {props.get('mevkii', '-')}<br/>
<b>Ada No:</b> {props.get('adaNo', '-')}<br/>
<b>Parsel No:</b> {props.get('parselNo', '-')}<br/>
<b>Pafta:</b> {props.get('pafta', '-')}<br/>
<b>Alan:</b> {props.get('alan', '-')} m²<br/>
<b>Nitelik:</b> {props.get('nitelik', '-')}<br/>
<b>Durum:</b> {props.get('zeminKmdurum', '-')}
]]>"""

            kml += f'<Placemark><name>{name}</name><description>{desc}</description>'
            kml += '<styleUrl>#s</styleUrl>'
            kml += f'<Polygon><outerBoundaryIs><LinearRing><coordinates>{coord_str}</coordinates>'
            kml += '</LinearRing></outerBoundaryIs></Polygon></Placemark>\n'

        kml += '</Document></kml>'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(kml)

    def closeEvent(self, event):  # pylint: disable=invalid-name
        """Pencere kapatilirken ayarlari kaydeder ve worker'i durdurur."""
        self.save_settings()
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()


# ==================== MAIN ====================

def main():
    """Uygulamayi baslatir."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
