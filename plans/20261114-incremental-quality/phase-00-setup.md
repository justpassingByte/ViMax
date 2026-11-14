# Phase 00 — Chuẩn bị

## Mục tiêu
Chuẩn bị cờ cấu hình và giữ lại output cũ để người dùng tự so sánh bằng mắt. Không đổi hành vi pipeline.

## Việc cần làm
1. Thêm section `quality:` vào `agent_runtime/config.py` (accessor theo kiểu các hàm `manga_vision_*`), `configs/agent.example.yaml` và `configs/agent.local.yaml`, với giá trị mặc định như trong `plan.md`.
2. **Không xóa** `.working_dir/` cũ. Mọi lần chạy mới sau nâng cấp dùng working dir mới (ví dụ `.working_dir/<ngày>-<truyện>-v2/`) để người dùng mở video cũ và mới cạnh nhau.
3. Không đụng `scratch/` và các script `fix_*`/`inspect_*` ở thư mục gốc. Đảm bảo không code chạy nào import từ chúng.
4. Chạy `.venv\Scripts\python.exe -m pytest tests -q` để chắc chắn đang xanh trước khi bắt đầu.

## Xong khi
Cờ `quality.*` đọc được từ config; test xanh.
