# Quét độc lập trước khi xem pre-label

Frame: frame_0099.jpg

Số xe nhìn thấy bằng mắt: 21 (đếm được toàn bộ 21 xe trên ảnh gốc trước khi mở nhãn AI; sau khi đối
chiếu, 13 xe trùng với box AI đã gán sẵn, còn lại 8 xe mình tự thêm vì AI bỏ sót)

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:

1. Xe ở góc/rìa khung hình: các xe nằm sát mép trên-trái hoặc mép dưới của ảnh, thân xe bị cắt một
   phần bởi biên khung hình nên mô hình dễ không nhận ra đủ đặc trưng để vẽ box.
2. Xe có đèn (pha/hậu) mờ: những xe không bật đèn sáng rõ hoặc đèn bị khuất/mờ hơn các xe xung
   quanh, hoà lẫn vào nền đường tối — mô hình cold start dễ bỏ sót nhóm xe này vì thiếu tín hiệu ánh
   sáng nổi bật để phân biệt với nền.

> Đối chiếu nhanh với dữ liệu: so khớp `to_label/round1/prelabels/frame_0099.txt` (13 box AI) với
> `day_08_export_cvat` (21 box sau khi bạn gán tay trong CVAT) cho thấy đúng 8 box được thêm mới,
> khớp hoàn toàn với số bạn báo cáo ở đây — dấu hiệu tốt cho thấy bước Blind Scan và bước rà nhãn
> thực tế nhất quán với nhau.

Chạy `python3 tools/lock_blind.py` ngay sau khi điền (bạn cần tự chạy lệnh này trên máy mình — mình
không có quyền truy cập terminal của bạn). Sau đó giữ file này nguyên vẹn.
