# Phase 09 — Trí nhớ cốt truyện qua các chương

Cờ: `quality.story_memory`

## Vì sao
- Mỗi tập được lập kế hoạch riêng; giữa các tập chỉ truyền ngân hàng nhân vật (`series_characters`, `series_seed_refs`). Reader chỉ có `chunk_tail_summary` của 3 trang trước.
- Không có nơi lưu "chuyện gì đã xảy ra", nên storyboard chương sau có thể mâu thuẫn chương trước, và reader có thể đặt tên nhân vật khác đi ở chương mới.

## Ba tầng thông tin (đúng như doc kế hoạch)
- **Sự kiện canon** (giữ suốt bộ truyện): chuyện đã xảy ra, ai làm, ý nghĩa lời thoại, quan hệ, năng lực/luật thế giới đã lộ, cái chết. Bắt buộc dẫn beat id.
- **Trạng thái lâu dài** của nhân vật (đồ, vết thương, hình dạng, đồ cầm, vị trí): đã lưu trong hồ sơ nhân vật (phase 05, `states`). Phase này chỉ thêm việc trích xuất mỗi chương.
- **Chi tiết tạm thời của cảnh** (ánh sáng, tư thế, góc máy, hiệu ứng): **không bao giờ** ghi vào trí nhớ.

## Thay đổi

### 9.1 Kho `utils/story_memory.py`
Thư mục `<series_dir>/story_memory/` (một tập: `<working_dir>/story_memory/`):
- `facts.jsonl`: `{fact_id, chapter, kind, text, character_ids, beat_ids}`; `kind ∈ event | dialogue_meaning | relationship | ability | world_rule | death`.
- `chapters/chNN.json`: `{summary (150–300 từ), open_threads: [...], cast: [{character_id, aliases_used}]}`.
- `conflicts.json`: các mâu thuẫn chờ duyệt.
- API: `add_chapter(...)`, `facts_for(character_ids, limit)`, `search(text, k)` (text embedding phase 04; không có embedding thì lọc theo nhân vật + mới nhất), `story_so_far(max_words)`.

### 9.2 Lượt trích xuất mỗi chương `agents/story_memory_extractor.py`
- Chạy sau khi đọc xong chương (sau `build_episode_scenes`), trước storyboard của chương đó.
- Một lời gọi LLM mỗi chương (chia lô nếu dài), đầu vào: beat của chương (id, nhân vật, thoại `vietsub`, action), tóm tắt chương trước. Đầu ra JSON: facts, state_changes, summary, open_threads.
- Kiểm bằng code: fact phải dẫn beat id có thật trong chương, nếu không → bỏ. State change phải có beat bằng chứng → ghi vào `profile_store` (phase 05) qua `add_state`, đóng trạng thái cũ cùng loại.
- Mâu thuẫn: với mỗi fact mới, lấy top-k fact cũ cùng nhân vật; một lời gọi LLM hỏi "có mâu thuẫn không"; có → `conflicts.json`, giữ cả hai, không chặn pipeline.

### 9.3 Đưa trí nhớ vào pipeline
- **Reader** (`agents/manga_reader.py::_invoke_chunk_vision`, human prompt): thêm khối `Story so far` (≤ 300 từ) và `Known cast` (id + alias) để giữ tên nhân vật ổn định giữa các chương. Khi reader gặp tên mới gần giống alias cũ, `profile_store.match` (phase 05) xử lý.
- **Storyboard**: thêm khối `STORY CONTEXT (read-only, do not depict unless in the script)` vào script cảnh trong bộ nhớ (không ghi đè `script.txt` trên đĩa), gồm fact của các nhân vật có mặt + top-k fact liên quan, giới hạn ký tự như `manga_vision.max_context_chars`.
- **Cổng cốt truyện** (phase 01): thêm kiểm tra LLM một lượt mỗi cảnh: "shot nào mâu thuẫn các fact sau?" → coi như issue `fact_contradiction`, đi vòng thử lại như các issue khác.
- **Series**: trong `plan_series`, chạy trích xuất sau mỗi tập theo đúng thứ tự chương trước khi lập kế hoạch tập sau.

## Test
- Fact không dẫn beat bị bỏ; state change đóng trạng thái cũ.
- `story_so_far` không vượt giới hạn từ.
- Kiểm tra mâu thuẫn với LLM giả → ghi `conflicts.json`, pipeline chạy tiếp.
- Prompt reader có khối `Known cast` khi cờ bật, không có khi tắt.
