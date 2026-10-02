// Rozšíření pro Voiceflow chat: změří obličej kamerou přímo v prohlížeči
// a do toku vrátí jen čísla. Obraz z kamery zařízení neopouští.

const BASE = new URL(".", import.meta.url).href
const TRACE_NAME = "ext_faceMeasure"

// Průměr lidské duhovky je téměř konstantní, slouží jako měřítko v obraze.
const IRIS_MM = 11.7
const SAMPLES_NEEDED = 60
const MIN_IRIS_PX = 20
const MAX_YAW = 8
const MAX_PITCH = 15
const MAX_ROLL = 8
const MAX_IRIS_MISMATCH = 0.12

// Indexy bodů MediaPipe Face Mesh (478 bodů včetně duhovek).
const IRIS_A = { center: 468, left: 469, right: 471 }
const IRIS_B = { center: 473, left: 474, right: 476 }
const EYE_A = { inner: 133, outer: 33 }
const EYE_B = { inner: 362, outer: 263 }
// Spodní okraj obočí nad středem oka a bod pod okem na dolním okraji očnice:
// mezi nimi sedí těsnění brýlí.
const SOCKET_A = { top: 52, bottom: 230 }
const SOCKET_B = { top: 282, bottom: 450 }
const FACE_EDGE_A = 234
const FACE_EDGE_B = 454

// S parametrem ?mereni_test v adrese se kamera nespustí a vrátí se pevná čísla.
// Slouží k ověření propojení s Voiceflow bez měření.
const TEST_MODE = new URLSearchParams(window.location.search).has("mereni_test")
const TEST_RESULT = {
  pd_mm: 62.4,
  pd_spread_mm: 0.6,
  inner_corners_mm: 33.1,
  outer_corners_mm: 91.8,
  eye_width_mm: 29.3,
  socket_height_mm: 31.5,
  face_width_mm: 138,
  samples: SAMPLES_NEEDED,
}

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y)
const deg = (rad) => (rad * 180) / Math.PI
const round1 = (n) => Math.round(n * 10) / 10

function percentile(values, p) {
  const sorted = [...values].sort((x, y) => x - y)
  const i = (sorted.length - 1) * p
  const lo = Math.floor(i)
  const hi = Math.ceil(i)
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo)
}

function measureError(reason, detail) {
  return Object.assign(new Error(detail || reason), { reason })
}

let landmarkerPromise = null

async function createLandmarker() {
  const { FaceLandmarker, FilesetResolver } = await import(BASE + "mediapipe/vision_bundle.mjs")
  const fileset = await FilesetResolver.forVisionTasks(BASE + "mediapipe/wasm")
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: BASE + "mediapipe/face_landmarker.task", delegate },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
  })
  try {
    return await FaceLandmarker.createFromOptions(fileset, options("GPU"))
  } catch (err) {
    console.warn("FaceMeasure: GPU není k dispozici, používám CPU", err)
    return await FaceLandmarker.createFromOptions(fileset, options("CPU"))
  }
}

// Knihovna a model (asi 13 MB) se stahují až při prvním měření, ne s načtením stránky.
export function preloadFaceMeasurement() {
  if (!landmarkerPromise) {
    landmarkerPromise = createLandmarker()
    landmarkerPromise.catch(() => {
      landmarkerPromise = null
    })
  }
  return landmarkerPromise
}

// Matice je 4x4 po sloupcích; z rotační části vytáhneme natočení hlavy.
function headAngles(matrix) {
  if (!matrix) return { yaw: 0, pitch: 0 }
  const m = matrix.data
  return {
    yaw: deg(Math.atan2(m[8], m[10])),
    pitch: deg(Math.asin(Math.max(-1, Math.min(1, -m[9])))),
  }
}

function measure(px, matrix) {
  const irisA = dist(px[IRIS_A.left], px[IRIS_A.right])
  const irisB = dist(px[IRIS_B.left], px[IRIS_B.right])
  const irisPx = (irisA + irisB) / 2
  const mmPerPx = IRIS_MM / irisPx
  const a = px[IRIS_A.center]
  const b = px[IRIS_B.center]
  const { yaw, pitch } = headAngles(matrix)
  return {
    irisPx,
    irisMismatch: Math.abs(irisA - irisB) / irisPx,
    pd: dist(a, b) * mmPerPx,
    innerCorners: dist(px[EYE_A.inner], px[EYE_B.inner]) * mmPerPx,
    outerCorners: dist(px[EYE_A.outer], px[EYE_B.outer]) * mmPerPx,
    eyeWidth:
      ((dist(px[EYE_A.inner], px[EYE_A.outer]) + dist(px[EYE_B.inner], px[EYE_B.outer])) / 2) *
      mmPerPx,
    socketHeight:
      ((dist(px[SOCKET_A.top], px[SOCKET_A.bottom]) + dist(px[SOCKET_B.top], px[SOCKET_B.bottom])) / 2) *
      mmPerPx,
    faceWidth: dist(px[FACE_EDGE_A], px[FACE_EDGE_B]) * mmPerPx,
    yaw,
    pitch,
    roll: deg(Math.atan(Math.abs(a.y - b.y) / Math.abs(a.x - b.x))),
  }
}

function problem(m) {
  if (m.irisPx < MIN_IRIS_PX) return "Přibližte se ke kameře"
  if (Math.abs(m.yaw) > MAX_YAW) return "Otočte hlavu rovně ke kameře"
  if (Math.abs(m.pitch) > MAX_PITCH) return "Nezaklánějte ani nepředklánějte hlavu"
  if (m.roll > MAX_ROLL) return "Držte hlavu rovně, nenaklánějte ji do strany"
  if (m.irisMismatch > MAX_IRIS_MISMATCH) return "Dívejte se přímo do kamery"
  return null
}

function summarize(samples) {
  const median = (key) => percentile(samples.map((s) => s[key]), 0.5)
  const pds = samples.map((s) => s.pd)
  return {
    pd_mm: round1(median("pd")),
    pd_spread_mm: round1((percentile(pds, 0.9) - percentile(pds, 0.1)) / 2),
    inner_corners_mm: round1(median("innerCorners")),
    outer_corners_mm: round1(median("outerCorners")),
    eye_width_mm: round1(median("eyeWidth")),
    socket_height_mm: round1(median("socketHeight")),
    face_width_mm: Math.round(median("faceWidth")),
    samples: samples.length,
  }
}

const OVERLAY_HTML = `
  <style>
    .fm-overlay { position: fixed; inset: 0; z-index: 2147483647; background: #000; color: #fff;
      font: 16px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
    .fm-overlay video, .fm-overlay canvas { position: absolute; inset: 0; width: 100%; height: 100%;
      object-fit: cover; transform: scaleX(-1); }
    .fm-top { position: absolute; top: 0; left: 0; right: 0; padding: 16px; display: flex;
      justify-content: space-between; align-items: center; gap: 12px;
      background: linear-gradient(rgba(0,0,0,.6), transparent); }
    .fm-tip { font-size: 14px; }
    .fm-cancel { font: inherit; font-weight: 600; color: #fff; background: rgba(0,0,0,.55);
      border: 1px solid rgba(255,255,255,.6); border-radius: 8px; padding: 8px 14px; cursor: pointer; }
    .fm-bottom { position: absolute; left: 16px; right: 16px; bottom: 24px; }
    .fm-hint { background: rgba(0,0,0,.65); padding: 12px 14px; border-radius: 10px;
      text-align: center; font-weight: 600; }
    .fm-hint.fm-ok { background: rgba(26,143,90,.9); }
    .fm-bar { height: 6px; background: rgba(255,255,255,.25); border-radius: 3px; margin-top: 10px; overflow: hidden; }
    .fm-bar > div { height: 100%; width: 0; background: #4fd39a; transition: width .1s linear; }
  </style>
  <video playsinline muted></video>
  <canvas></canvas>
  <div class="fm-top">
    <span class="fm-tip">Sundejte si brýle a dívejte se přímo do kamery.</span>
    <button type="button" class="fm-cancel">Zrušit</button>
  </div>
  <div class="fm-bottom">
    <div class="fm-hint">Načítám…</div>
    <div class="fm-bar"><div></div></div>
  </div>
`

// Otevře měření přes celou obrazovku. Vrací naměřené hodnoty,
// při zrušení nebo chybě odmítne s chybou, která má pole `reason`.
export function openFaceMeasurement() {
  if (TEST_MODE) return Promise.resolve({ ...TEST_RESULT })

  return new Promise((resolve, reject) => {
    const overlay = document.createElement("div")
    overlay.className = "fm-overlay"
    overlay.innerHTML = OVERLAY_HTML
    document.body.appendChild(overlay)

    const video = overlay.querySelector("video")
    const canvas = overlay.querySelector("canvas")
    const ctx = canvas.getContext("2d")
    const hint = overlay.querySelector(".fm-hint")
    const progress = overlay.querySelector(".fm-bar > div")

    let running = true
    let stream = null
    let landmarker = null
    let lastVideoTime = -1
    const samples = []

    const setHint = (text, ok = false) => {
      hint.textContent = text
      hint.classList.toggle("fm-ok", ok)
    }

    const close = () => {
      running = false
      if (stream) stream.getTracks().forEach((track) => track.stop())
      overlay.remove()
    }

    overlay.querySelector(".fm-cancel").addEventListener("click", () => {
      close()
      reject(measureError("cancelled"))
    })

    const draw = (px) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      if (!px) return
      ctx.lineWidth = 2
      ctx.strokeStyle = "#4fd39a"
      for (const iris of [IRIS_A, IRIS_B]) {
        ctx.beginPath()
        ctx.arc(px[iris.center].x, px[iris.center].y, dist(px[iris.left], px[iris.right]) / 2, 0, Math.PI * 2)
        ctx.stroke()
      }
    }

    const tick = () => {
      if (!running) return
      requestAnimationFrame(tick)
      if (video.readyState < 2 || video.currentTime === lastVideoTime) return
      lastVideoTime = video.currentTime

      const res = landmarker.detectForVideo(video, performance.now())
      const landmarks = res.faceLandmarks?.[0]
      if (!landmarks) {
        draw(null)
        setHint("Nevidím obličej")
        return
      }

      const w = video.videoWidth
      const h = video.videoHeight
      const px = landmarks.map((p) => ({ x: p.x * w, y: p.y * h }))
      const m = measure(px, res.facialTransformationMatrixes?.[0])
      draw(px)

      const issue = problem(m)
      if (issue) {
        setHint(issue)
        return
      }
      setHint("Držte, měřím…", true)
      samples.push(m)
      progress.style.width = (samples.length / SAMPLES_NEEDED) * 100 + "%"
      if (samples.length >= SAMPLES_NEEDED) {
        close()
        resolve(summarize(samples))
      }
    }

    ;(async () => {
      try {
        landmarker = await preloadFaceMeasurement()
        if (!running) return
        setHint("Povolte přístup ke kameře…")
        if (!navigator.mediaDevices?.getUserMedia) throw measureError("camera_unavailable")
        const media = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        })
        if (!running) {
          media.getTracks().forEach((track) => track.stop())
          return
        }
        stream = media
        video.srcObject = stream
        await video.play()
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        tick()
      } catch (err) {
        if (!running) return
        console.error("FaceMeasure:", err)
        close()
        if (err.reason) reject(err)
        else if (err.name === "NotAllowedError") reject(measureError("camera_denied"))
        else if (err.name === "NotFoundError") reject(measureError("camera_unavailable"))
        else reject(measureError("error", err.message))
      }
    })()
  })
}

const FAILURE_TEXT = {
  cancelled: "Měření bylo zrušeno. Můžete ho spustit znovu.",
  camera_denied: "Přístup ke kameře byl zamítnut. Povolte ho v prohlížeči, nebo vyberte bez měření.",
  camera_unavailable: "Kameru se nepodařilo najít. Vyberte prosím bez měření.",
  error: "Měření se nepodařilo. Zkuste to znovu, nebo vyberte bez měření.",
}

export const FaceMeasureExtension = {
  name: "FaceMeasure",
  type: "response",
  match: ({ trace }) => trace.type === TRACE_NAME || trace.payload?.name === TRACE_NAME,
  render: ({ element }) => {
    const box = document.createElement("div")
    box.innerHTML = `
      <style>
        .fm-card { font: inherit; display: grid; gap: 8px; min-width: 220px; }
        .fm-card p { margin: 0; font-size: 14px; }
        .fm-card button { font: inherit; font-weight: 600; padding: 10px 14px; border-radius: 8px;
          border: 1px solid #0a7ea4; cursor: pointer; }
        .fm-card .fm-go { background: #0a7ea4; color: #fff; }
        .fm-card .fm-skip { justify-self: center; background: transparent; color: #0a7ea4;
          padding: 6px 12px; font-size: 13px; }
        .fm-card button:disabled { opacity: .5; cursor: default; }
        .fm-card .fm-msg:empty { display: none; }
      </style>
      <div class="fm-card">
        <p>Měření trvá pár vteřin. Obraz z kamery zůstává ve vašem zařízení.</p>
        <button type="button" class="fm-go">Spustit měření</button>
        <button type="button" class="fm-skip">Vybrat bez měření</button>
        <p class="fm-msg"></p>
      </div>
    `
    const go = box.querySelector(".fm-go")
    const skip = box.querySelector(".fm-skip")
    const msg = box.querySelector(".fm-msg")
    let lastFailure = "skipped"

    const finish = (type, payload, text) => {
      go.disabled = true
      skip.disabled = true
      msg.textContent = text
      window.voiceflow.chat.interact({ type, payload })
    }

    go.addEventListener("click", async () => {
      go.disabled = true
      skip.disabled = true
      msg.textContent = ""
      try {
        const result = await openFaceMeasurement()
        finish("complete", result, "Změřeno ✓")
      } catch (err) {
        lastFailure = err.reason || "error"
        msg.textContent = FAILURE_TEXT[lastFailure] || FAILURE_TEXT.error
        go.disabled = false
        skip.disabled = false
      }
    })

    skip.addEventListener("click", () => {
      finish("cancel", { reason: lastFailure }, "")
    })

    element.appendChild(box)
  },
}
