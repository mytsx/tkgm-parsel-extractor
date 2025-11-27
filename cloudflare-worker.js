// Cloudflare Worker - TKGM API Proxy
// Bu worker'ı Cloudflare Dashboard'da deploy edin

// Batch endpoint sabitleri
const MAX_BATCH_COORDS = 20;      // Maksimum koordinat sayisi
const STAGGER_START_INDEX = 5;    // Bu indexten sonra gecikme baslar
const STAGGER_DELAY_MS = 50;      // Her istek arasi gecikme (ms)

export default {
  async fetch(request, env, ctx) {
    // CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, X-API-Key",
        },
      });
    }

    // Opsiyonel API Key kontrolu
    // Cloudflare Dashboard > Worker > Settings > Variables > API_KEY ekleyin
    if (env.API_KEY) {
      const providedKey = request.headers.get("X-API-Key");
      if (providedKey !== env.API_KEY) {
        return new Response(JSON.stringify({ error: "Unauthorized - Invalid API Key" }), {
          status: 401,
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
          }
        });
      }
    }

    const url = new URL(request.url);

    // /parsel/lat/lon endpoint'i
    if (url.pathname.startsWith("/parsel/")) {
      const parts = url.pathname.split("/");
      const lat = parts[2];
      const lon = parts[3];

      if (!lat || !lon) {
        return new Response(JSON.stringify({ error: "lat ve lon parametreleri gerekli" }), {
          status: 400,
          headers: { "Content-Type": "application/json" }
        });
      }

      const tkgmUrl = `https://cbsapi.tkgm.gov.tr/megsiswebapi.v3.1/api/parsel/${lat}/${lon}/`;

      try {
        const response = await fetch(tkgmUrl, {
          headers: {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://parselsorgu.tkgm.gov.tr/",
            "Origin": "https://parselsorgu.tkgm.gov.tr"
          }
        });

        const data = await response.text();

        return new Response(data, {
          status: response.status,
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "X-Proxied-By": "cloudflare-worker"
          }
        });
      } catch (error) {
        return new Response(JSON.stringify({ error: error.message }), {
          status: 500,
          headers: { "Content-Type": "application/json" }
        });
      }
    }

    // /batch endpoint'i - toplu sorgu (optimize edilmis)
    if (url.pathname === "/batch" && request.method === "POST") {
      try {
        const body = await request.json();
        const coordinates = body.coordinates; // [{lat, lon}, ...]

        if (!coordinates || !Array.isArray(coordinates)) {
          return new Response(JSON.stringify({ error: "coordinates array gerekli" }), {
            status: 400,
            headers: { "Content-Type": "application/json" }
          });
        }

        // Koordinat limitini uygula
        const limitedCoords = coordinates.slice(0, MAX_BATCH_COORDS);

        const results = await Promise.all(
          limitedCoords.map(async (coord, index) => {
            // Staggered delay: ilk STAGGER_START_INDEX aninda, sonrakiler STAGGER_DELAY_MS arayla
            if (index >= STAGGER_START_INDEX) {
              await new Promise(r => setTimeout(r, (index - STAGGER_START_INDEX) * STAGGER_DELAY_MS));
            }

            const tkgmUrl = `https://cbsapi.tkgm.gov.tr/megsiswebapi.v3.1/api/parsel/${coord.lat}/${coord.lon}/`;

            try {
              const response = await fetch(tkgmUrl, {
                headers: {
                  "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                  "Accept": "application/json",
                  "Referer": "https://parselsorgu.tkgm.gov.tr/",
                  "Origin": "https://parselsorgu.tkgm.gov.tr"
                }
              });

              if (response.ok) {
                const data = await response.json();
                // Basarili sonucu dondur
                return data;
              }
              // Bos/hata durumunda null dondur
              return null;
            } catch (e) {
              return null;
            }
          })
        );

        // Basarili sonuclari say
        const successCount = results.filter(r => r && r.properties).length;

        return new Response(JSON.stringify({
          results,
          count: results.length,
          success: successCount
        }), {
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
          }
        });
      } catch (error) {
        return new Response(JSON.stringify({ error: error.message }), {
          status: 500,
          headers: { "Content-Type": "application/json" }
        });
      }
    }

    return new Response(JSON.stringify({
      message: "TKGM Proxy API",
      endpoints: {
        parsel: "GET /parsel/{lat}/{lon}",
        batch: "POST /batch with {coordinates: [{lat, lon}, ...]}"
      }
    }), {
      headers: { "Content-Type": "application/json" }
    });
  },
};
