# 🎬 Hướng Dẫn Sử Dụng ViMax Studio (Tiếng Việt)

Chào mừng bạn đến với **ViMax Studio** – Framework AI Agentic chuyên sản xuất phim ngắn, phim dài tập và chuyển thể tiểu thuyết thành phim điện ảnh tự động.

---

## 🚀 1. Cách Khởi Động Nhanh

### Cách 1: Chạy bằng file Click đúp (Khuyên dùng trên Windows)
1. Bấm đúp vào file **`run_web.bat`** tại thư mục gốc của ViMax.
2. Mở trình duyệt web và truy cập địa chỉ: **[http://127.0.0.1:4173](http://127.0.0.1:4173)**.

### Cách 2: Khởi động bằng dòng lệnh PowerShell / Terminal
```powershell
cd C:\Users\NoNomt007\.gemini\antigravity\scratch\ViMax\web
npm run dev
```

---

## ⚙️ 2. Cấu Hình API Key & Model

Giao diện ViMax đã được **Việt hóa hoàn toàn**. Bạn có 2 cách để thiết lập API:

### Cách A: Thiết lập trực quan trên Trình duyệt (Dễ nhất)
1. Mở Web UI tại [http://127.0.0.1:4173](http://127.0.0.1:4173).
2. Vào mục **"Cài đặt Cấu hình"** (biểu tượng bánh răng ở thanh menu bên trái).
3. Điền API Key và nhấn **"Lưu cấu hình"**.

### Cách B: Chỉnh sửa file `configs/agent.local.yaml`
Mở file `configs/agent.local.yaml` và điền key của các dịch vụ bạn dùng:

* **Mục LLM (Biên kịch & Đạo diễn):**
  * `model_provider: openai`
  * `model: gpt-4o-mini` hoặc `deepseek-chat` hoặc `gemini-2.5-flash`
  * `base_url: https://api.openai.com/v1` (hoặc URL của OpenRouter / DeepSeek / Gemini)
  * `api_key: 'sk-...'`
* **Mục Image (Tạo ảnh nhân vật & Keyframe):**
  * `model: gemini-3.1-flash-image-preview` (hoặc flux, midjourney qua OpenRouter)
  * `base_url: https://openrouter.ai/api/v1`
  * `api_key: 'sk-...'`
* **Mục Video (Tạo chuyển động cú máy):**
  * `model: veo3.1-fast` (hoặc kling, runway qua OpenRouter)
  * `base_url: https://openrouter.ai/api/v1`
  * `api_key: 'sk-...'`

---

## 🎥 3. Hướng Dẫn 3 Quy Trình Sản Xuất Phim

### 📖 Quy trình 1: Chuyển Thể Tiểu Thuyết Thành Phim (Novel2Video) - Dành cho phim dài
1. Bấm **"Dự án mới"** trên Web UI.
2. Bấm nút dấu cộng **`+`** ở khung chat để tải lên file truyện/tiểu thuyết của bạn (dạng file `.txt`).
3. Nhắn cho ViMax:
   > *"Hãy chuyển thể cuốn tiểu thuyết này thành phim dài tập theo quy trình novel2video."*
4. **ViMax sẽ tự động làm:**
   - Nén tiểu thuyết và tóm tắt theo các sự kiện lớn.
   - Trích xuất danh sách nhân vật (tính cách, diện mạo) để giữ đồng nhất khuôn mặt xuyên suốt phim.
   - Tạo kịch bản phân cảnh cho từng chương.
   - Lập danh sách cú máy (Storyboard).
   - Render hình ảnh và xuất video từng phân đoạn.

### 📝 Quy trình 2: Dựng Phim Từ Kịch Bản Có Sẵn (Script2Video)
Nếu bạn đã có kịch bản từng cảnh:
1. Nhắn cho ViMax hoặc dán kịch bản vào khung chat.
2. Nhắn lệnh:
   > *"Sử dụng kịch bản này để làm phim theo quy trình script2video."*
3. Bạn có thể xem trước bảng phân cảnh (Storyboard) ở thanh trượt bên phải trước khi bấm Render.

### 💡 Quy trình 3: Tạo Phim Nhanh Từ Ý Tưởng (Idea2Video)
1. Chỉ cần gõ ý tưởng vào khung chat:
   > *"Làm một video ngắn về một phi hành gia cô độc tìm thấy một bông hoa phát sáng trên sao Hỏa theo phong cách điện ảnh viễn tưởng (Cinematic Sci-Fi)."*
2. ViMax sẽ hỏi xác nhận và bắt đầu tự sáng tác kịch bản, thiết kế nhân vật và dựng thành phim.

---

## 🛠️ 4. Các Phím Tắt & Tính Năng Hữu Ích

* **Nút chuyển đổi ngôn ngữ (VI / EN):** Nằm ở góc trên bên phải màn hình để bạn dễ dàng chuyển qua lại giữa Tiếng Việt và Tiếng Anh.
* **Bảng Storyboard (Cột bên phải):** Bấm vào biểu tượng bảng phân cảnh để xem chi tiết từng shot quay, mô tả chuyển động camera và khung hình chính trước khi video được dựng.
* **Lệnh `/compact`:** Khi dự án quá dài hoặc nhiều cảnh, gõ `/compact` trong khung chat để nén bộ nhớ ngữ cảnh, giúp AI không bị chậm hoặc quên cốt truyện.
