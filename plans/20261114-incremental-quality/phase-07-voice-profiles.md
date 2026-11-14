# Phase 07 — Giọng nói nhất quán (vẫn do Muse lồng tiếng)

Cờ: `quality.voice_mode`: `native` | `native_fixed` | `native_with_reference` (mặc định `native_fixed`)

## Vì sao
- Cấu hình hiện tại `audio.model_provider: native`: giọng trong mỗi clip do model video (qua muse2api) tự tạo từ đoạn mô tả chữ "Vocal Acoustic Profile" (`_format_video_prompt_audio`, `_resolve_character_acoustic_profile` trong `pipelines/script2video_pipeline.py` ~dòng 271–500). Mô tả này thay đổi theo ngữ cảnh shot → giọng trôi.
- Người dùng dùng Muse tự lồng tiếng. **Không** dùng TTS ngoài, **không** dùng chuyển giọng.

## Thay đổi

### 7.1 `native_fixed` (mặc định)
- `VoiceProfile.description` (phase 05; nếu phase 05 chưa xong thì đọc từ `series/<slug>/characters.yaml`) là **một câu cố định** cho mỗi nhân vật: tuổi, giới, âm vực, chất giọng, cách nói. Nếu chưa có, LLM viết **một lần** từ `permanent_features` rồi lưu lại, các lần sau dùng lại nguyên văn.
- `_resolve_character_acoustic_profile` trả đúng câu đó cho mọi shot của nhân vật; không ghép thêm chữ theo ngữ cảnh shot. Cảm xúc của câu thoại đi ở dòng riêng (`Delivery emotion: …`), không đổi câu mô tả giọng.
- Lời dẫn chuyện (Narrator) cũng có một câu mô tả cố định riêng.

### 7.2 `native_with_reference` (tự bật nếu Muse hỗ trợ)
- Hàm `probe_audio_reference_support(video_generator)` trong `tools/`: gửi một request video ngắn có kèm file audio mẫu theo trường mà API có thể nhận; kết quả (có nhận / bị bỏ qua / lỗi) lưu cache vào `<working_dir>/provider_capabilities.json`, chỉ dò một lần.
- Nếu Muse nhận: giọng mẫu của mỗi nhân vật chính lấy tự động từ clip native đầu tiên có nhân vật đó nói một mình, cắt 5–15 giây theo thời điểm phụ đề, lưu vào `VoiceProfile.reference_audio`; các clip sau gửi kèm giọng mẫu.
- Nếu không nhận: tự dùng `native_fixed`, ghi một dòng cảnh báo, không lỗi.

## Test tự động
- `native_fixed` sinh cùng một mô tả giọng cho cùng nhân vật ở mọi shot.
- Provider giả không nhận audio → tự về `native_fixed`, không crash.
