// Cloudflare Worker - TKGM API Proxy
// Bu worker'ı Cloudflare Dashboard'da deploy edin

// Sabitler
const MAX_BATCH_COORDS = 20;      // Maksimum koordinat sayisi
const STAGGER_START_INDEX = 5;    // Bu indexten sonra gecikme baslar
const STAGGER_DELAY_MS = 50;      // Her istek arasi gecikme (ms)

// TKGM API Base URL'leri
const TKGM_API_V3 = "https://cbsapi.tkgm.gov.tr/megsiswebapi.v3/api";
const TKGM_API_V31 = "https://cbsapi.tkgm.gov.tr/megsiswebapi.v3.1/api";

// Ortak header'lar
const TKGM_HEADERS = {
  "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
  "Accept": "application/json",
  "Referer": "https://parselsorgu.tkgm.gov.tr/",
  "Origin": "https://parselsorgu.tkgm.gov.tr"
};

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, X-API-Key",
};

// Yardımcı fonksiyon - TKGM'ye istek at
async function fetchTKGM(url) {
  return fetch(url, { headers: TKGM_HEADERS });
}

// JSON yanıt oluştur
function jsonResponse(data, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...CORS_HEADERS, ...extraHeaders }
  });
}

export default {
  async fetch(request, env, ctx) {
    // CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS_HEADERS });
    }

    // Opsiyonel API Key kontrolu
    if (env.API_KEY) {
      const providedKey = request.headers.get("X-API-Key");
      if (providedKey !== env.API_KEY) {
        return jsonResponse({ error: "Unauthorized - Invalid API Key" }, 401);
      }
    }

    const url = new URL(request.url);
    const path = url.pathname;

    try {
      // ============================================
      // İDARİ YAPI ENDPOINTLERİ
      // ============================================

      // GET /iller - Tüm illerin listesi
      if (path === "/iller" || path === "/iller/") {
        const response = await fetchTKGM(`${TKGM_API_V3}/idariYapi/ilListe`);
        const data = await response.json();
        return jsonResponse(data, response.status);
      }

      // GET /ilceler/{ilId} - Belirli ilin ilçeleri
      const ilceMatch = path.match(/^\/ilceler\/(\d+)\/?$/);
      if (ilceMatch) {
        const ilId = ilceMatch[1];
        const response = await fetchTKGM(`${TKGM_API_V3}/idariYapi/ilceListe/${ilId}`);
        const data = await response.json();
        return jsonResponse(data, response.status);
      }

      // GET /mahalleler/{ilceId} - Belirli ilçenin mahalleleri
      const mahalleMatch = path.match(/^\/mahalleler\/(\d+)\/?$/);
      if (mahalleMatch) {
        const ilceId = mahalleMatch[1];
        const response = await fetchTKGM(`${TKGM_API_V3}/idariYapi/mahalleListe/${ilceId}`);
        const data = await response.json();
        return jsonResponse(data, response.status);
      }

      // GET /parsel-by-ada/{mahalleId}/{ada}/{parsel} - Ada/Parsel ile sorgu
      const adaParselMatch = path.match(/^\/parsel-by-ada\/(\d+)\/(\d+)\/(\d+)\/?$/);
      if (adaParselMatch) {
        const [, mahalleId, ada, parsel] = adaParselMatch;
        const response = await fetchTKGM(`${TKGM_API_V31}/parsel/${mahalleId}/${ada}/${parsel}`);
        const data = await response.json();
        return jsonResponse(data, response.status);
      }

      // ============================================
      // KOORDİNAT İLE PARSEL SORGULAMA
      // ============================================

      // GET /parsel/{lat}/{lon} - Koordinat ile parsel sorgu
      if (path.startsWith("/parsel/")) {
        const parts = path.split("/");
        const lat = parts[2];
        const lon = parts[3];

        if (!lat || !lon) {
          return jsonResponse({ error: "lat ve lon parametreleri gerekli" }, 400);
        }

        const response = await fetchTKGM(`${TKGM_API_V31}/parsel/${lat}/${lon}/`);
        const data = await response.text();

        return new Response(data, {
          status: response.status,
          headers: {
            "Content-Type": "application/json",
            ...CORS_HEADERS,
            "X-Proxied-By": "cloudflare-worker"
          }
        });
      }

      // ============================================
      // BATCH ENDPOINT
      // ============================================

      // POST /batch - Toplu koordinat sorgusu
      if (path === "/batch" && request.method === "POST") {
        const body = await request.json();
        const coordinates = body.coordinates;

        if (!coordinates || !Array.isArray(coordinates)) {
          return jsonResponse({ error: "coordinates array gerekli" }, 400);
        }

        const limitedCoords = coordinates.slice(0, MAX_BATCH_COORDS);

        const results = await Promise.all(
          limitedCoords.map(async (coord, index) => {
            if (index > STAGGER_START_INDEX) {
              await new Promise(r => setTimeout(r, (index - STAGGER_START_INDEX) * STAGGER_DELAY_MS));
            }

            try {
              const response = await fetchTKGM(`${TKGM_API_V31}/parsel/${coord.lat}/${coord.lon}/`);

              if (response.ok) {
                const data = await response.json();
                if (data && data.properties) {
                  return { status: "found", data, coord };
                }
                return { status: "empty", data: null, coord };
              }

              return { status: "error", error: response.status, data: null, coord };
            } catch (e) {
              return { status: "error", error: e.message, data: null, coord };
            }
          })
        );

        const stats = {
          total: results.length,
          found: results.filter(r => r.status === "found").length,
          empty: results.filter(r => r.status === "empty").length,
          error: results.filter(r => r.status === "error").length
        };

        return jsonResponse({ results, stats, count: results.length, success: stats.found });
      }

      // ============================================
      // ANA SAYFA - API DOKÜMANTASYONU
      // ============================================

      return jsonResponse({
        name: "TKGM Proxy API",
        version: "2.0",
        endpoints: {
          // İdari Yapı
          iller: "GET /iller - Tüm illerin listesi",
          ilceler: "GET /ilceler/{ilId} - Belirli ilin ilçeleri",
          mahalleler: "GET /mahalleler/{ilceId} - Belirli ilçenin mahalleleri",
          parselByAda: "GET /parsel-by-ada/{mahalleId}/{ada}/{parsel} - Ada/Parsel ile sorgu",
          // Koordinat Sorgu
          parsel: "GET /parsel/{lat}/{lon} - Koordinat ile parsel sorgu",
          batch: "POST /batch - Toplu koordinat sorgusu {coordinates: [{lat, lon}, ...]}"
        },
        examples: {
          iller: "/iller",
          boluIlceleri: "/ilceler/36",
          mudurnuMahalleleri: "/mahalleler/271",
          parselByAda: "/parsel-by-ada/134649/101/5",
          parselByKoordinat: "/parsel/40.123/32.456"
        }
      });

    } catch (error) {
      return jsonResponse({ error: error.message }, 500);
    }
  },
};
