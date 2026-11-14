# 🎬 Hướng Dẫn Sử Dụng ViMax Studio (Tiếng Việt)

Chào mừng bạn đến với **ViMax Studio** – Framework AI Agentic chuyên sản xuất phim ngắn, phim dài tập và chuyển thể tiểu thuyết thành phim điện ảnh tự động.

---

## 🚀 1. Cách Khởi Động Nhanh

### Cách 1: Chạy bằng file Click đúp (Khuyên dùng trên Windows)
1. Bấm đúp vào file **`ViMax_Studio.exe`** hoặc **`run_web.bat`** tại thư mục gốc của ViMax.
2. Mở trình duyệt web và truy cập địa chỉ: **[http://127.0.0.1:4173](http://127.0.0.1:4173)**.

### Cách 2: Khởi động bằng dòng lệnh PowerShell / Terminal
```powershell
cd C:\Users\NoNomt007\.gemini\antigravity\scratch\ViMax\web
npm run dev
```

---

## ⚙️ 2. Cấu Hình API Key & Model

Giao diện ViMax đã được **Việt hóa hoàn toàn**. Bạn có thể cấu hình trực quan tại mục **"Cài đặt Cấu hình"** trên Web UI hoặc sửa trực tiếp file `configs/agent.local.yaml`:

### ⚡ Cấu hình Video - Tối ưu Tốc Độ Tạo Phim với Seedance (Khuyên Dùng):
Tốc độ tạo video thường là nút thắt cổ chai lớn nhất khi làm phim AI. Các model truyền thống như Google Veo 3 / Kling / Sora thường mất từ **1.5 đến 3 phút** cho mỗi shot quay 5 giây. Với phim dài có 50 - 100 shots, thời gian chờ có thể lên tới 2 - 4 tiếng.

**Doubao Seedance (ByteDance)** là giải pháp tối ưu hàng đầu:
* **Tốc độ thần tốc:** Chỉ **~10 - 25 giây** mỗi cú máy (nhanh gấp 5 - 10 lần các model khác).
* **Khả năng nối shot mượt mà (First Frame & Last Frame to Video):** Giữ chuyển động camera liền mạch giữa các cảnh.
* **Chi phí tiết kiệm:** Rẻ hơn 3x - 5x so với Veo 3.

**Cách thiết lập Seedance:**
* **Cách 1: Trực tiếp qua Yunwu API (Khuyên dùng):**
  * `model_provider: seedance` (hoặc để trống)
  * `model: doubao-seedance-1-0-lite-t2v-250428` (hoặc `seedance-2.0-fast`)
  * `base_url: https://yunwu.ai/volc/v1`
  * `api_key: <Key_Yunwu_Cua_Ban>`
* **Cách 2: Qua OpenRouter (Nếu dùng chung cổng OpenRouter):**
  * `model_provider: openrouter`
  * `model: bytedance/seedance-2.0-fast`
  * `base_url: https://openrouter.ai/api/v1`
  * `api_key: <Key_OpenRouter_Cua_Ban>`

### 🎨 Cấu hình Tạo Ảnh (Image) - Nhân Vật & Khung Hình Chính (Keyframe):
Trong ViMax, tạo ảnh đảm nhận 2 nhiệm vụ cốt lõi:
1. **Thiết kế nhân vật (Character Sheet):** Tạo hình ảnh gốc để AI giữ chuẩn khuôn mặt, trang phục nhân vật suốt cả phim.
2. **Khung hình chính (Keyframe):** Tạo các ảnh 16:9 sắc nét cho từng cảnh quay trước khi chuyển cho model video tạo chuyển động.

**Các model & provider khuyên dùng (có thể chỉnh `model_provider` ngay trong Cài đặt Web UI):**
* **Lựa chọn 1 (Miễn phí qua Web Gateway cục bộ / VPS): `muse2api`**
  * `model_provider: muse2api`
  * `model: muse-image`
  * `base_url: http://127.0.0.1:18610/v1` (hoặc `http://muse2api:18610/v1` khi chạy Docker Compose)
  * `api_key: <MUSE2API_KEY>`
  * *(Tùy chọn)* Biến môi trường `VIMAX_MUSE_MEDIA_CONCURRENCY=1` (mặc định `1` để chạy tuần tự ổn định; nâng lên `2`–`3` nếu pool tài khoản muse2api có nhiều acc + proxy).
* **Lựa chọn 2 (Nhanh ~2 - 4 giây/ảnh): Google Gemini Flash Image qua Yunwu**
  * `model_provider: yunwu`
  * `model: gemini-3.1-flash-image-preview` (hoặc `gemini-2.5-flash-image-preview`)
  * `base_url: https://yunwu.ai`
  * `api_key: <YUNWU_API_KEY>` *(Dùng chung 1 key với LLM và Seedance video!)*
* **Lựa chọn 3 (Đẹp nhất cho nhân vật & chân thực): FLUX qua OpenRouter**
  * `model_provider: openrouter`
  * `model: black-forest-labs/flux-1.1-pro` (hoặc `black-forest-labs/flux-schnell`, `openai/gpt-image-2`)
  * `base_url: https://openrouter.ai/api/v1`
  * `api_key: <OPENROUTER_API_KEY>`
* **Lựa chọn 4 (Đồng bộ ByteDance cùng Seedance): Doubao SeaDream qua Yunwu**
  * `model: doubao-seedream-4-0-250828`
  * `base_url: https://yunwu.ai`
  * `api_key: <YUNWU_API_KEY>`

### 🎙️ Cấu hình Âm Thanh (`native` / ElevenLabs / Edge-TTS) & Phụ Đề Vietsub Tự Động:
1. **Chế độ `native` (KHUYÊN DÙNG khi video đã có thoại & SFX gốc từ Veo / Muse / Seedance Audio):**
   - Đặt `audio.model_provider: native` -> ViMax giữ nguyên giọng nói & hiệu ứng gốc của từng shot video, **không cần ElevenLabs**! Vẫn có thể trộn thêm nhạc nền `bgm_file`.
2. **Chế độ lồng tiếng TTS (`elevenlabs` / `edge-tts`):**
   - Tự động bóc tách người nói theo từng shot (`[Speaker] Alice: ...`), ánh xạ Voice ID riêng hoặc tự nhận diện Nam/Nữ, hòa âm khớp Timeline.
3. **Phụ đề `.srt` / `.ass` & Burn-in Vietsub:**
   - Bật mục **Phụ đề (Subtitles)** trong Web UI hoặc `configs/agent.local.yaml`:
     - Trích xuất trực tiếp các dòng `(Vietsub: ...)` trong `script.txt`, hoặc tự động dịch thoại spoken bằng `Agent LLM`.
     - Tạo đồng thời `final_video_vietsub.srt` & `final_video_vietsub.ass` khớp từng miligiây thời lượng thật từng clip và tự động gắn cứng (`libass` FFmpeg) vào `final_video.mp4`.

```yaml
audio:
  model_provider: native        # native, elevenlabs, hoặc edge-tts
  bgm_file: ''
  bgm_volume: 0.15

subtitles:
  enabled: true
  language: vi
  burn_in: true
  font: Arial
  font_size: 34
```

### 🤖 LLM (Biên kịch & Đạo diễn - `AgentLoop`):
* Hỗ trợ các model có **OpenAI Tool Calling / Function Calling** tốt: `gpt-4o`, `gpt-5.x`, `claude`, `gemini-2.5-flash` / `pro`, `deepseek-chat`.
* **Lưu ý quan trọng khi dùng `muse2api`:** Trong `muse2api`, `MUSE2API_TOOL_PROTOCOL` mặc định bằng `0` (chỉ tương thích tham số, không giả lập JSON tool call qua web chat muse.ai). Vì vậy cho Web UI / `AgentLoop` (`main_agent.py`), hãy trỏ `llm` tới endpoint hỗ trợ tool-calling thực thụ (OpenAI / Yunwu / OpenRouter / Gemini), và dùng `muse2api` cho `image` + `video`.

### ⏱️ Quy Hoạch Thời Lượng Phim (~8 giây / Shot):
* Mỗi video clip đầu ra trung bình dài **~8 giây** (~**7–8 shots / phút**):
  * Video 3 phút ≈ **23 shots**
  * Video 5 phút ≈ **38 shots**
  * Video 10 phút ≈ **75 shots**
* Mỗi shot chỉ nên viết **1–2 câu thoại ngắn (≤ 20–25 từ)** để thoại kịp dứt trọn vẹn trong 8 giây.

---

## 🎥 3. Hướng Dẫn 3 Quy Trình Sản Xuất Phim

### 📖 Quy trình 1: Chuyển Thể Tiểu Thuyết Thành Phim (Novel2Video) - Dành cho phim dài
1. Bấm **"Dự án mới"** trên Web UI.
2. Bấm nút dấu cộng **`+`** ở khung chat để tải lên file truyện/tiểu thuyết của bạn (dạng file `.txt`).
3. Nhắn cho ViMax:
   > *"Hãy chuyển thể cuốn tiểu thuyết này thành phim dài tập theo quy trình novel2video."*
4. **ViMax sẽ tự động:**
   - Nén tiểu thuyết và tóm tắt theo các sự kiện lớn.
   - Trích xuất danh sách nhân vật (tính cách, diện mạo) để giữ đồng nhất khuôn mặt xuyên suốt phim.
   - Tạo kịch bản phân cảnh cho từng chương.
   - Lập danh sách cú máy (Storyboard).
   - Render hình ảnh, sinh clip video và tự động lồng tiếng đa nhân vật + ghép nhạc nền vào phim hoàn chỉnh!

### 📝 Quy trình 2: Dựng Phim Từ Kịch Bản Có Sẵn (Script2Video)
1. Dán kịch bản vào khung chat và nhắn lệnh:
   > *"Sử dụng kịch bản này để làm phim theo quy trình script2video."*
2. Bạn có thể xem trước bảng phân cảnh (Storyboard) ở thanh trượt bên phải trước khi bấm Render.

### 💡 Quy trình 3: Tạo Phim Nhanh Từ Ý Tưởng (Idea2Video)
1. Chỉ cần gõ ý tưởng vào khung chat:
   > *"Làm một video ngắn về một phi hành gia cô độc tìm thấy một bông hoa phát sáng trên sao Hỏa theo phong cách điện ảnh viễn tưởng (Cinematic Sci-Fi)."*
2. ViMax sẽ hỏi xác nhận và bắt đầu tự sáng tác kịch bản, thiết kế nhân vật, lồng tiếng và dựng thành phim.

---

## 🛠️ 4. Các Phím Tắt & Tính Năng Hữu Ích

* **Nút chuyển đổi ngôn ngữ (VI / EN):** Nằm ở góc trên bên phải màn hình để bạn dễ dàng chuyển qua lại giữa Tiếng Việt và Tiếng Anh.
* **Bảng Storyboard (Cột bên phải):** Bấm vào biểu tượng bảng phân cảnh để xem chi tiết từng shot quay, mô tả chuyển động camera và khung hình chính trước khi video được dựng.
* **Lệnh `/compact`:** Khi dự án quá dài hoặc nhiều cảnh, gõ `/compact` trong khung chat để nén bộ nhớ ngữ cảnh.

---

## 🌐 5. Triển Khai 24/7 Trên Linux VPS (Docker & systemd)

### Cách A: Chạy bằng Docker Compose (Khuyên dùng - tích hợp sẵn FFmpeg libass + Font tiếng Việt + Node + Python)
```bash
# 1. Tạo file cấu hình cục bộ từ mẫu và điền API keys (tuyệt đối không commit agent.local.yaml)
cp configs/agent.example.yaml configs/agent.local.yaml

# 2. Build Web UI & khởi chạy container chạy nền 24/7
docker compose -f docker-compose.vps.yml up -d --build
```
* Truy cập Web Studio tại: `http://<IP_VPS>:4173` (khuyên đặt sau Nginx/Caddy HTTPS + Basic Auth hoặc SSH Tunnel).
* Nếu chạy cùng `muse2api` trên cùng VPS host, đặt `base_url: http://host.docker.internal:18610/v1` trong `agent.local.yaml` (`docker-compose.vps.yml` đã bật sẵn `host.docker.internal`).

### Cách B: Chạy trực tiếp trên Ubuntu/Debian bằng `systemd`
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg fonts-dejavu-core fonts-noto-core nodejs npm
# Tạo venv Python (>=3.12), cài dependencies từ pyproject.toml & build web/dist:
python3 -m venv .venv && ./.venv/bin/pip install uv && ./.venv/bin/uv pip install --python ./.venv/bin/python -r pyproject.toml
(cd web && npm ci && npm run build)
# Cài service systemd mẫu (chỉnh sửa User / WorkingDirectory nếu cần):
sudo cp deploy/vimax-web.service /etc/systemd/system/vimax-web.service
sudo systemctl daemon-reload && sudo systemctl enable --now vimax-web
```

---

## Nâng cấp chất lượng (section `quality:` trong `agent.local.yaml`)

Các cờ `quality.*` bật/tắt từng cải tiến; mặc định đã bật bản nâng cấp. Xem mô tả
từng cờ trong `configs/agent.example.yaml`. Để lùi một phần nếu thấy kém hơn, đổi
cờ tương ứng về giá trị cũ (ví dụ `canon_gate: false`, `ref_policy: legacy`,
`embeddings: off`, `voice_mode: native`, `shot_planner: false`).

- **Hồ sơ nhân vật** (`profiles`): sau khi đọc truyện, pipeline dựng hồ sơ nháp cho
  từng nhân vật trong `<working_dir>/profiles/`. Duyệt một lần bằng trang HTML:
  `python -m tools.profile_sheet <working_dir>` rồi mở file
  `profiles/profiles_review.html`. Chốt/khoá hồ sơ: `python -m utils.profile_store approve|lock|merge <id> [<id2>] --base <working_dir>`.

- **Duyệt ngoại lệ mỗi chương** (`exceptions_queue`): lỗi (danh tính keyframe, gợi ý
  ghép hồ sơ, mâu thuẫn cốt truyện, canon phải sửa) gom vào `exceptions/chNN.json`.
  Xem một trang: `python -m tools.review_page <working_dir> chNN` → mở
  `exceptions/chNN.review.html`. Ghi quyết định vào `exceptions/chNN.decisions.json`
  rồi áp dụng: `python -m pipelines.apply_decisions <working_dir> chNN` (xoá keyframe
  để làm lại đúng shot, hoặc ghép hồ sơ). Chạy lại chương chỉ sinh lại phần bị ảnh hưởng.

- **Embedding** (`embeddings`, `text_embeddings`): cần nhóm phụ thuộc tuỳ chọn
  `embeddings` (`torch`, `transformers`, `numpy`). Model tải lần đầu về cache mặc định:
  `facebook/dinov2-small` (Apache-2.0) và `intfloat/multilingual-e5-small` (MIT).
  Thiếu thư viện/model → tính năng tự tắt, pipeline vẫn chạy.
