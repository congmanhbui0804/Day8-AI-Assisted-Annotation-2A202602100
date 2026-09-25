# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, chọn năm frame bạn sẽ ưu tiên nếu chỉ có
ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét
ảnh gần trùng hoặc trường hợp model không dự đoán được box:

1. **frame_0182.jpg** — hạng 1, t=72.8s, score=0.9591. Điểm cao nhất toàn danh sách, A=1.0 (gần một
   nửa trong 28 box là ca khó, n_ambiguous=18) — cảnh đông xe, rà một ảnh này sửa được nhiều lỗi.
2. **frame_0369.jpg** — hạng 2, t=147.6s, score=0.9324. Chọn ảnh này thay vì hai khung liền kề
   **frame_0372.jpg** (hạng 6, t=148.8s, cách 1.2s, score=0.9101) và **frame_0368.jpg** (hạng 9,
   t=147.2s, cách 0.4s, score=0.9003) — cả hai đều rất gần về thời gian (dưới `min_gap_s=2.0` xác
   nhận trong `batch.json`) nên gần như cùng một cảnh giao thông; rà một đại diện là đủ, không cần
   tốn ngân sách cho cả ba ảnh gần trùng.
3. **frame_0099.jpg** — hạng 8, t=39.6s, score=0.9063. Đây chính là ảnh mình đã Blind Scan trực
   tiếp (xem `BLIND_SCAN.md`): đếm được 21 xe bằng mắt trước khi xem nhãn AI, trong khi AI chỉ gán
   13 box (đúng cả 13, không có box sai) — bỏ sót 8 xe. Rà ảnh này cho bằng chứng thực tế, đã được
   tự tay kiểm chứng, về loại lỗi phổ biến nhất của model (bỏ sót xe nhỏ/tối/ở rìa ảnh).
4. **frame_0392.jpg** — hạng 15, t=156.8s, score=0.8874. A thấp nhất trong lô được chọn (0.6667)
   nhưng U cao nhất (0.9747) — mô hình rất "phân vân" ở ảnh này dù số box mơ hồ không nhiều, đáng
   xem để hiểu vì sao model bất định ngay cả khi ranh giới box không quá khó.
5. **frame_0227.jpg** — hạng 11, t=90.8s, score=0.8915. Đại diện cho mốc thời gian ~90s, khác hẳn
   bốn frame trên (72.8s, 147.6s, 39.6s, 156.8s), giúp lô 5 ảnh trải đều theo thời gian thay vì dồn
   vào 1–2 đoạn video.

Ba frame thuộc lô 12 ảnh model chọn (`selected=True` trong `selection_round1.csv`, khớp
`to_label/round1/batch.json` và `outputs/selection_round1.jpg`) và bằng chứng:

- **frame_0182.jpg** (hạng 1, score 0.9591, A=1.0, D=1.0) — điểm tổng cao nhất, xuất hiện ở ô đầu
  tiên trong `selection_round1.jpg`, 18/28 box được đánh giá mơ hồ.
- **frame_0331.jpg** (hạng 5, score 0.9154, n_boxes=47) — cảnh đông xe nhất trong toàn bộ lô, theo
  `round1_diff.md` có tới 22 box sau khi rà (20 giữ nguyên, 2 thêm mới) — xác nhận đây thực sự là
  ảnh phức tạp, nhiều đối tượng.
- **frame_0099.jpg** (hạng 8, score 0.9063) — bằng chứng mạnh nhất vì có cả Blind Scan độc lập
  (21 xe đếm bằng mắt) và Review Log (8 box thêm mới) khớp chính xác với nhau.

Một frame có điểm cao nhưng không chọn: **frame_0372.jpg** (hạng 6, t=148.8s, score=0.9101 — cao
hơn cả `frame_0107.jpg` hạng 14, score 0.8876, đã được chọn). Lý do không chọn: nó chỉ cách
`frame_0369.jpg` (đã chọn) 1.2 giây, nhỏ hơn `min_gap_s=2.0` ghi trong `batch.json` — cơ chế D phạt
điểm ảnh gần trùng thời gian, dù bản thân U và A của frame_0372 không hề thấp (U=0.9202, A=0.8333).
Việc loại nó chứng minh hệ thống ưu tiên đa dạng theo thời gian hơn là chỉ chăm chăm theo điểm số.

Điều phép chọn này chưa chứng minh về chất lượng mô hình: điểm số U/A/D chỉ phản ánh model cold
start đang bất định/khó ở đâu và ảnh nào tốn công gán nhãn — nó không đo được model sau khi train
thêm các ảnh này có thực sự cải thiện AP50/recall hay không. Bằng chứng thực tế: dù lô 12 ảnh này
được chọn kỹ theo điểm số và được rà nhãn rất cẩn thận (167/169 box AI giữ nguyên đúng, chỉ 1 box
bị xoá vì sai, xem `round1_diff.md`), AP50 sau khi train vẫn giảm mạnh từ 0.771 xuống 0.420
(`reports/rounds_table.md`) — cho thấy chọn đúng ảnh "khó, bất định" theo công thức không tự động
đảm bảo mô hình học tốt hơn; chất lượng huấn luyện (learning rate, số epoch, số ảnh, và liệu pipeline
có dùng đúng bộ nhãn đã rà xong hay không — xem lưu ý về chênh lệch 177 vs 219 box trong
`round1_diff.md`) vẫn quyết định kết quả cuối cùng nhiều hơn cách chọn mẫu.
