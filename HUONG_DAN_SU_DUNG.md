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

**Các model khuyên dùng:**
* **Lựa chọn 1 (KHUYÊN DÙNG - Nhanh nhất & Rẻ nhất): Google Gemini Flash Image qua Yunwu**
  * `model: gemini-3.1-flash-image-preview` (hoặc `gemini-2.5-flash-image-preview`)
  * `base_url: https://yunwu.ai`
  * `api_key: <Key_Yunwu_Cua_Ban>` *(Dùng chung 1 key với LLM và Seedance video!)*
  * Tốc độ: Cực nhanh, chỉ **~2 - 4 giây/ảnh**.
* **Lựa chọn 2 (Đẹp nhất cho nhân vật & chân thực): FLUX qua OpenRouter**
  * `model: black-forest-labs/flux-1.1-pro` (hoặc `black-forest-labs/flux-schnell`, `openai/gpt-image-2`)
  * `base_url: https://openrouter.ai/api/v1`
  * `api_key: <Key_OpenRouter_Cua_Ban>`
  * Chi tiết khuôn mặt nhân vật, ánh sáng điện ảnh, nếp gấp trang phục hoàn hảo.
* **Lựa chọn 3 (Đồng bộ ByteDance cùng Seedance): Doubao SeaDream qua Yunwu**
  * `model: doubao-seedream-4-0-250828`
  * `base_url: https://yunwu.ai`
  * `api_key: <Key_Yunwu_Cua_Ban>`

### 🎙️ Cấu hình Âm thanh & Lồng tiếng (ElevenLabs / Edge-TTS)
* **Nếu dùng ElevenLabs (Đẳng cấp điện ảnh - Khuyên dùng):**
  * `model_provider: elevenlabs`
  * `model: eleven_multilingual_v2` (model đa ngôn ngữ hỗ trợ tiếng Việt cực kỳ truyền cảm, có nhịp thở, cảm xúc)
  * `base_url: https://api.elevenlabs.io/v1`
  * `api_key: 'sk_...'` (Key tài khoản ElevenLabs của bạn)
  * `voice: 21m00Tcm4TlvDq8ikWAM` (Voice ID trên ElevenLabs - bạn có thể tạo hoặc chọn bất kỳ giọng diễn viên nào trên thư viện Voice Library của ElevenLabs rồi dán Voice ID vào đây)
* **Nếu dùng Edge-TTS (Miễn phí):**
  * `model_provider: edge-tts`
  * `voice: vi-VN-NamMinhNeural` (Nam) hoặc `vi-VN-HoaiMyNeural` (Nữ)
  * `api_key: ''` (để trống vì không cần key)
* **Nhạc nền BGM:**
  * `bgm_file: 'C:/duong_dan_den_nhac.mp3'` (để trống nếu không dùng)
  * `bgm_volume: 0.15` (âm lượng nhỏ vừa phải để làm nền cho giọng nói)

### 🤖 LLM (Biên kịch & Đạo diễn):
* `gpt-4o-mini`, `deepseek-chat`, hoặc `gemini-2.5-flash`

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
   - Render hình ảnh, sinh clip video và tự động lồng tiếng + ghép nhạc nền vào phim hoàn chỉnh!

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
