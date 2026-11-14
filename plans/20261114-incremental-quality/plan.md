# Incremental Quality — Kế hoạch nâng cấp end-to-end (cải tiến tại chỗ, không viết lại)

> Dành cho coding agent. Đọc hết file này, rồi làm **tất cả** các phase theo thứ tự.
> Người dùng sẽ tự đánh giá chất lượng bằng cách tạo video và xem. **Không** chạy thí nghiệm, A/B, chấm điểm hay làm bộ đo riêng.

## Mục tiêu sản phẩm (theo thứ tự ưu tiên)

1. Giữ đúng cốt truyện gốc.
2. Giữ danh tính và ngoại hình nhân vật nhất quán qua các cảnh, các chương.
3. Giữ giọng nói nhân vật nhất quán (vẫn do Muse lồng tiếng).
4. Góc máy đa dạng, đúng ngữ cảnh, có hiệu ứng anime phù hợp.
5. Chạy cả chương, rồi cả bộ truyện, không cần duyệt tay từng cảnh.

Không cần chất lượng studio. Được sáng tạo về góc máy, ánh sáng, biểu cảm, chuyển cảnh, hiệu ứng.
**Không được** bịa sự kiện, thoại, tương tác, nhân vật không thuộc cảnh, đổi thứ tự sự kiện, hay mâu thuẫn canon.

## Luật bắt buộc cho agent

- **Làm hết, end-to-end.** Làm tuần tự phase 00 → 10. Không dừng chờ người dùng giữa các phase, trừ hai chỗ: (a) hỏi đường dẫn ổ đĩa để đặt `HF_HOME` trước khi tải model ở phase 04; (b) phase 10.4 (xóa guard) chỉ làm khi người dùng xác nhận.
- **Không viết lại pipeline.** Mọi thay đổi nằm trong luồng `manga2video` → `script2video` hiện có. Không tạo pipeline thứ hai.
- **Mỗi thay đổi có cờ** trong section `quality:` (bên dưới). **Mặc định bật** (giá trị mới), để lần chạy tiếp theo dùng bản nâng cấp. Tắt cờ = quay về hành vi cũ, để người dùng tự lùi nếu thấy phần nào kém hơn.
- **Không có tên nhân vật / bộ truyện trong code** (`agents/`, `pipelines/`, `utils/`, `tools/`, `interfaces/`, `agent_runtime/`). Dữ liệu riêng của bộ truyện nằm trong `series/<slug>/`.
- **Không sửa** `scratch/`, các script `fix_*`/`inspect_*` ở thư mục gốc, file `.tmp_*`.
- **Test tự động phải xanh** sau mỗi phase: `.venv\Scripts\python.exe -m pytest tests -q`. Mỗi module mới có unit test với model/provider giả (không gọi Muse, không tải model trong test). Đây là test để code không vỡ, không phải đánh giá chất lượng.
- **Muse không phải nguồn sự thật.** Không lưu trạng thái nhân vật/cốt truyện trong hội thoại Muse. Mọi request phải dựng lại được từ file của ViMax.
- **Không lỗi cứng khi thiếu thứ gì:** thiếu thư viện embedding, thiếu model, provider không hỗ trợ → tự tắt tính năng đó, ghi cảnh báo, pipeline vẫn chạy.
- **Không xóa output cũ** trong `.working_dir/`. Lần chạy sau nâng cấp dùng working dir mới để người dùng xem video cũ và mới cạnh nhau.
- Mỗi phase xong: commit riêng, message ghi rõ phase.

## Môi trường của người dùng

- Windows; RTX 3070 8 GB (desktop) và laptop RTX 3050. Python trong `.venv`.
- Sinh ảnh/video qua **muse2api** chạy local tại `127.0.0.1:18610`, mặc định 1 luồng (`VIMAX_MUSE_MEDIA_CONCURRENCY`).
- Giọng: `audio.model_provider: native` — **Muse tự lồng tiếng**. Không dùng TTS ngoài (Edge-TTS, ElevenLabs), không dùng chuyển giọng.
- Ổ C gần đầy: model Hugging Face và `torch` phải đặt sang ổ còn trống (`HF_HOME`). Chỉ tải 2 model: `facebook/dinov2-small` và `intfloat/multilingual-e5-small`.

## Cờ cấu hình (`quality:`) — mặc định sau nâng cấp

```yaml
quality:
  canon_gate: true             # phase 01
  generic_linter: true         # phase 02
  series_slug: ""              # phase 02: tên thư mục trong series/, ví dụ eleceed
  ref_policy: profile          # phase 03/05: legacy | all_characters | profile
  max_reference_images: 4      # phase 03: tự giảm nếu muse2api treo
  embeddings: dinov2           # phase 04: off | dinov2
  text_embeddings: e5          # phase 04: off | e5
  scene_split_threshold: 0.80  # phase 04
  profiles: true               # phase 05
  profile_min_appearances: 5   # phase 05: chỉ sinh sheet cho nhân vật xuất hiện >= N beat
  identity_qa: true            # phase 06
  identity_margin: 0.15        # phase 06
  max_keyframe_retakes: 2      # phase 06
  voice_mode: native_fixed     # phase 07: native | native_fixed | native_with_reference
  shot_planner: true           # phase 08
  story_memory: true           # phase 09
  exceptions_queue: true       # phase 10
  max_clip_retakes: 1          # phase 10
  chapter_budget_images: 0     # phase 10: 0 = không giới hạn
  chapter_budget_clips: 0      # phase 10
```

Thêm accessor trong `agent_runtime/config.py` theo kiểu các hàm `manga_vision_*` hiện có, và ghi chú trong `configs/agent.example.yaml`. Với Eleceed, đặt `series_slug: eleceed` trong `configs/agent.local.yaml`.

## Các phase (làm theo thứ tự)

| Phase | File | Việc chính |
| --- | --- | --- |
| 00 | `phase-00-setup.md` | Thêm cờ cấu hình, giữ output cũ |
| 01 | `phase-01-story-fidelity.md` | Bỏ lệnh bịa nội dung trong prompt storyboard; cổng kiểm tra bám cốt truyện có thử lại và tự sửa |
| 02 | `phase-02-generic-linter.md` | Thay guard bằng bộ kiểm tra dùng chung; chuyển dữ liệu Eleceed từ code sang `series/eleceed/` |
| 03 | `phase-03-reference-fix.md` | Sửa lỗi model ảnh chỉ nhận 2 ảnh mẫu; mọi nhân vật trong shot đều có portrait |
| 04 | `phase-04-embeddings.md` | Module embedding (DINOv2-small, E5-small); làm sạch seed; gom cảnh và địa điểm |
| 05 | `phase-05-character-profiles.md` | Hồ sơ nhân vật (trang phục, trạng thái, alias); chọn ảnh mẫu theo trạng thái; ghép nhân vật qua các chương |
| 06 | `phase-06-identity-qa.md` | Kiểm tra danh tính trên keyframe, làm lại trước khi tạo video |
| 07 | `phase-07-voice-profiles.md` | Giọng cố định cho từng nhân vật (Muse vẫn lồng tiếng) |
| 08 | `phase-08-shot-planner.md` | Lên góc máy theo loại beat, có luật đa dạng, hiệu ứng theo ngữ cảnh |
| 09 | `phase-09-story-memory.md` | Trí nhớ cốt truyện qua các chương |
| 10 | `phase-10-unattended-runs.md` | QA clip, hàng ngoại lệ, ngân sách, chạy nhiều chương không cần người |

Các file phase có thể còn nhắc tới "thí nghiệm", "đo", "so sánh" ở đâu đó: **bỏ qua**, chỉ làm phần "Thay đổi" và "Test".

## Xong khi

- Mọi phase đã làm, test xanh, cờ mặc định như bảng trên.
- Lệnh chạy series hiện có chạy được nhiều chương liên tiếp với các cờ bật, không lỗi cứng.
- Ghi `plans/20261114-incremental-quality/CHANGES.md`: mỗi phase một đoạn ngắn — đã đổi gì, file nào, cờ nào để tắt nếu người dùng thấy kém hơn.
