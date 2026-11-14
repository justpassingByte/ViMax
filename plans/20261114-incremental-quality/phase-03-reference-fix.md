# Phase 03 — Sửa giới hạn ảnh mẫu

Cờ: `quality.ref_policy` (`legacy` | `all_characters`), `quality.max_reference_images` (mặc định 4)

## Vì sao (đây có thể là nguyên nhân lớn nhất gây trôi ngoại hình ở shot nhiều nhân vật)
Trong `pipelines/script2video_pipeline.py`:
- ~dòng 1369 và ~1921: danh sách ứng viên bị cắt `[:4]` **trước** khi LLM chọn ("cap refs to avoid muse2api payload hang"). Danh sách được nạp theo thứ tự: mọi góc (front/side/back) của nhân vật thứ nhất, rồi nhân vật thứ hai, rồi panel. Với 2 nhân vật, nhân vật thứ hai chỉ còn 1 ảnh và panel bị loại; với 3 nhân vật, nhân vật thứ ba không còn ảnh nào để LLM chọn.
- ~dòng 1428 và ~1939: `_ensure_character_portraits_and_spatial_anchors(..., limit=2)` → **model ảnh chỉ nhận tối đa 2 ảnh mẫu**. Portrait được đặt trước, nên shot có 2 nhân vật mất ảnh neo cảnh/panel; shot có 3 nhân vật trở lên thiếu portrait của người thứ ba.
- Không file nào ghi lại ảnh mẫu thực sự đã gửi cho model ảnh, nên hiện không đo được điều này.

## Thay đổi

### 3.1 Ghi lại ảnh mẫu thực tế (luôn bật, không cần cờ)
Ngay trước `generate_and_select_best_image` ở mọi chỗ gọi (first frame, last frame, `generate_frame_for_single_shot`), ghi `shots/<idx>/<frame>_refs_used.json`: danh sách `{path, text, role}` với `role ∈ portrait | panel | scene_anchor | continuity | other`, cùng số nhân vật có mặt và số nhân vật có portrait. File này dùng để debug khi người dùng thấy sai nhân vật.

### 3.2 Gom logic ảnh mẫu về một hàm
Ba chỗ dựng danh sách ảnh mẫu (first frame trong `generate_frames_for_single_camera`, last frame trong `_build_shot_last_frame`, và `generate_frame_for_single_shot`) đang lặp code gần giống nhau. Gom về một hàm `_build_reference_pairs(shot, frame_type, vis_chars, registry, panel_pairs, scene_anchor, continuity_pair, policy, limit)` trong `script2video_pipeline.py`, giữ nguyên hành vi khi `ref_policy: legacy` (có test so khớp đầu ra cũ). Phase 05 thêm nhánh `profile` vào hàm này.

### 3.3 Chính sách `all_characters`
- Bỏ việc cắt `[:4]` trước LLM. Thay bằng: mỗi nhân vật có mặt chỉ đưa **một** góc portrait phù hợp nhất vào ứng viên (ưu tiên `front`; `side`/`back` nếu `ff_desc` nói nhìn nghiêng/từ sau, bằng so khớp từ khóa đơn giản), cộng tối đa 2 panel, cộng ảnh neo cảnh.
- `_ensure_character_portraits_and_spatial_anchors(limit=max_reference_images)`: đảm bảo **mọi** nhân vật có mặt có đúng 1 portrait; phần còn trống mới dành cho neo cảnh, rồi panel.
- Nếu số nhân vật có mặt > `max_reference_images - 1`: ghép portrait của các nhân vật thành **một ảnh sheet** (xếp ngang, nhãn tên nhỏ dưới mỗi người, nền trung tính) bằng PIL trong module mới `utils/reference_sheet.py`, rồi gửi sheet như một ảnh. Mô tả kèm theo liệt kê thứ tự trái → phải.
- Giữ nguyên ghi chú "không copy nét chibi/screentone từ panel" (chuyển sang `shot_linter` nếu phase 02 đã xong).

### 3.4 Trần theo provider
- Thêm thuộc tính tùy chọn `max_reference_images: int` cho các generator trong `tools/` (đọc trong `ImageGeneratorOpenRouterAPI.__init__` từ config; mặc định 4). `quality.max_reference_images` ghi đè.
- Dự phòng lúc chạy: nếu request có N ảnh bị timeout/treo với muse2api, tự thử lại một lần với N−1 ảnh (ghép các portrait thành sheet). Ghi N ổn định vào `<working_dir>/provider_capabilities.json` để các shot sau dùng luôn.

## Test
- Unit: 3 nhân vật có mặt, `max_reference_images=4` → 3 portrait + 1 neo cảnh; `max_reference_images=2` → 1 sheet + 1 neo cảnh.
- Unit: `reference_sheet.compose([...])` tạo ảnh đúng kích thước, đúng thứ tự.
- Unit: `*_refs_used.json` được ghi với đúng `role`.
